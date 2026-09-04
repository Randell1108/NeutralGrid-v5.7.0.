"""
Loader that extracts a :class:`SeedState` from replay-exported CSV files.

Reads ``levels.csv`` produced by ``neutralgrid-replay`` under
``data/replay_outputs/<SYMBOL>/`` and reconstructs the active order-book
ladder at the snapshot closest to (but not later than) a target timestamp.

This module has **no imports** from ``src/neutralgrid/replay`` — it reads
the exported CSVs directly, keeping the ``backtest/`` package independent.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import pandas as pd

from backtest.btk_seed_state import SeedOrderLevel, SeedState

logger = logging.getLogger(__name__)


_ACTIVE_ORDER_STATUSES = {"NEW", "PARTIALLY_FILLED"}


def _strategy_id_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    try:
        fval = float(value)
        if pd.notna(fval) and fval.is_integer():
            return str(int(fval))
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _numeric(value: Any) -> float:
    if value is None or pd.isna(value):
        return 0.0
    text = str(value).strip().replace(",", "")
    if not text:
        return 0.0
    token = text.split()[0]
    try:
        return float(token)
    except ValueError:
        return 0.0


def _utc(value: Any) -> datetime | None:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return cast(pd.Timestamp, ts).to_pydatetime()


def _coerce_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _time_provenance(
    *,
    raw_value: Any,
    source: str,
    normalized: datetime | None,
) -> dict[str, str]:
    return {
        "raw_time_text": "" if raw_value is None else str(raw_value),
        "raw_time_source": source,
        "raw_timezone": "UTC",
        "normalized_time_utc": (
            _coerce_utc(normalized).isoformat() if normalized is not None else ""
        ),
    }


def _order_files(exports_dir: Path) -> list[Path]:
    if not exports_dir.exists():
        return []
    return sorted(
        path
        for path in exports_dir.glob("*.csv")
        if "order history" in path.name.lower()
    )


def load_seed_from_replay(
    replay_dir: Path,
    symbol: str,
    t0: datetime,
) -> SeedState | None:
    """Load a :class:`SeedState` from replay exports for *symbol* at *t0*.

    Parameters
    ----------
    replay_dir : Path
        Root replay output directory (e.g. ``data/replay_outputs``).
    symbol : str
        Trading pair (e.g. ``RIVERUSDT``).  Must have a matching sub-directory.
    t0 : datetime
        Target timestamp (UTC).  The loader picks the latest snapshot whose
        ``time_ms`` is ``<= t0``.

    Returns
    -------
    SeedState | None
        Populated seed state, or *None* if no matching data is found.
    """
    levels_path = replay_dir / symbol / "levels.csv"
    if not levels_path.exists():
        logger.debug("No levels.csv found for %s at %s", symbol, levels_path)
        return None

    try:
        df = pd.read_csv(levels_path)
    except Exception as exc:
        logger.warning("Cannot read levels.csv for %s: %s", symbol, exc)
        return None

    if df.empty or "time_ms" not in df.columns:
        return None

    # Ensure t0 is UTC-aware for comparison
    if t0.tzinfo is None:
        t0 = t0.replace(tzinfo=timezone.utc)
    t0_ms = int(t0.timestamp() * 1000)

    # Find the latest snapshot at or before t0
    valid = df[df["time_ms"] <= t0_ms]
    if valid.empty:
        logger.debug("No levels at or before %s for %s", t0, symbol)
        return None

    latest_raw = cast(Any, valid["time_ms"].max())
    latest_ms = int(latest_raw)
    snapshot = cast(pd.DataFrame, valid[valid["time_ms"] == latest_ms])

    buy_levels = []
    sell_levels = []
    for _, row in snapshot.iterrows():
        side = str(row.get("side") or "").upper()
        price = float(row.get("price") or 0)
        qty = float(row.get("total_orig_qty") or 0)
        order_count = int(row.get("order_count") or 1)

        if price <= 0:
            continue

        level = SeedOrderLevel(
            side=side,
            price=price,
            qty_total=qty,
            order_count=order_count,
        )
        if side == "BUY":
            buy_levels.append(level)
        elif side == "SELL":
            sell_levels.append(level)

    snap_dt = datetime.fromtimestamp(latest_ms / 1000, tz=timezone.utc)
    logger.info(
        "Loaded seed for %s at %s: %d buy levels, %d sell levels",
        symbol, snap_dt.isoformat(), len(buy_levels), len(sell_levels),
    )

    return SeedState(
        t0=snap_dt,
        open_buy_levels=buy_levels,
        open_sell_levels=sell_levels,
        symbol=symbol,
        evidence_class="partial",
        source=str(levels_path),
    )


def load_seed_from_order_history_exports(
    exports_dir: Path,
    symbol: str,
    t0: datetime,
    *,
    strategy_id: str | int | None = None,
) -> SeedState | None:
    """Load a seed ladder directly from Binance Order History CSV exports.

    The export directory may contain many symbols and strategies.  This loader
    returns a seed only for the requested bot, identified by exact ``symbol``,
    optional exact ``strategy_id``, and orders active at ``t0``.

    It is intentionally an adapter: missing, partial, or conflicting evidence
    is reflected in the returned ``SeedState`` metadata, not hidden by inferred
    order state.
    """
    files = _order_files(exports_dir)
    if not files:
        logger.debug("No Order history CSV files found in %s", exports_dir)
        return None

    frames: list[pd.DataFrame] = []
    for path in files:
        try:
            df = pd.read_csv(path, dtype=str)
        except Exception as exc:
            logger.warning("Cannot read order history %s: %s", path, exc)
            continue
        if df.empty:
            continue
        df = df.copy()
        df.columns = df.columns.str.strip()
        df["_source_file"] = path.name
        frames.append(df)

    if not frames:
        return None

    orders = pd.concat(frames, ignore_index=True)
    required = {"Time(UTC)", "Order No", "Symbol", "Side", "Price", "Amount", "Status"}
    missing = required.difference(set(orders.columns))
    if missing:
        logger.warning("Order history exports missing required columns: %s", sorted(missing))
        return None

    orders["_symbol"] = orders["Symbol"].astype(str).str.strip().str.upper()
    orders = cast(pd.DataFrame, orders.loc[orders["_symbol"] == symbol.strip().upper()].copy())
    if orders.empty:
        return None

    if strategy_id is not None and str(strategy_id).strip():
        sid = _strategy_id_text(strategy_id)
        if "Strategy Id" not in orders.columns:
            logger.debug(
                "Order history exports for %s do not include Strategy Id; "
                "exact seed for strategy_id=%s is not verifiable",
                symbol,
                sid,
            )
            return None
        strategy_series = cast(pd.Series, orders["Strategy Id"])
        strategy_values = strategy_series.map(_strategy_id_text)
        orders = cast(pd.DataFrame, orders.loc[strategy_values == sid].copy())
        if orders.empty:
            return None

    orders["_order_id"] = orders["Order No"].astype(str).str.strip()
    orders["_create_raw_time_text"] = orders["Time(UTC)"].astype(str)
    orders["_create_raw_time_source"] = "Time(UTC)"
    orders["_create_raw_timezone"] = "UTC"
    orders["_create_dt"] = orders["Time(UTC)"].map(_utc)
    update_raw = (
        cast(pd.Series, orders["Update Time"])
        if "Update Time" in orders.columns
        else cast(pd.Series, orders["Time(UTC)"])
    )
    orders["_update_raw_time_text"] = update_raw.astype(str)
    orders["_update_raw_time_source"] = (
        "Update Time" if "Update Time" in orders.columns else "Time(UTC)"
    )
    orders["_update_raw_timezone"] = "UTC"
    orders["_update_dt"] = update_raw.map(_utc)
    orders["_price"] = orders["Price"].map(_numeric)
    orders["_amount"] = orders["Amount"].map(_numeric)
    executed_raw = (
        cast(pd.Series, orders["Executed Amount"])
        if "Executed Amount" in orders.columns
        else pd.Series(0.0, index=orders.index)
    )
    orders["_executed"] = executed_raw.map(_numeric)
    orders["_status"] = orders["Status"].astype(str).str.strip().str.upper()
    orders["_side"] = orders["Side"].astype(str).str.strip().str.upper()

    orders = cast(
        pd.DataFrame,
        orders.loc[
            orders["_create_dt"].notna()
            & orders["_update_dt"].notna()
            & (orders["_price"] > 0)
            & (orders["_amount"] > 0)
            & orders["_side"].isin(["BUY", "SELL"])
        ].copy(),
    )
    if orders.empty:
        return None

    orders = cast(
        pd.DataFrame,
        orders.sort_values(["_order_id", "_update_dt", "_executed"])
        .drop_duplicates("_order_id", keep="last")
        .reset_index(drop=True),
    )

    target = _coerce_utc(t0)

    def _active_at_start(row: pd.Series) -> bool:
        created = cast(datetime | None, row.get("_create_dt"))
        updated = cast(datetime | None, row.get("_update_dt"))
        if created is None or updated is None:
            return False
        status = str(row.get("_status", "")).upper()
        amount = float(row.get("_amount", 0.0) or 0.0)
        executed = float(row.get("_executed", 0.0) or 0.0)
        remaining = max(amount - executed, 0.0)
        still_open = status in _ACTIVE_ORDER_STATUSES and remaining > 0
        if still_open and updated <= created:
            return created <= target
        return created <= target < updated

    active_mask = orders.apply(_active_at_start, axis=1)
    active = cast(pd.DataFrame, orders.loc[active_mask].copy())
    if active.empty:
        return None

    buy_levels: list[SeedOrderLevel] = []
    sell_levels: list[SeedOrderLevel] = []
    qty_values: list[float] = []
    conflict = False

    for key, group in active.groupby(["_side", "_price"], sort=True):
        side, price = cast(tuple[Any, Any], key)
        side_text = str(side)
        price_f = float(price)
        qty_parts: list[float] = []
        for _, row in group.iterrows():
            amount = float(row.get("_amount", 0.0) or 0.0)
            executed = float(row.get("_executed", 0.0) or 0.0)
            remaining = max(amount - executed, 0.0)
            qty = remaining if remaining > 0 else amount
            if qty > 0:
                qty_parts.append(qty)
                qty_values.append(qty)
        if not qty_parts:
            continue
        level = SeedOrderLevel(
            side=side_text,
            price=price_f,
            qty_total=float(sum(qty_parts)),
            order_count=int(len(group)),
        )
        if len(group) > 1:
            conflict = True
        if side_text == "BUY":
            buy_levels.append(level)
        elif side_text == "SELL":
            sell_levels.append(level)

    if not buy_levels and not sell_levels:
        return None

    rounded_qty = {round(qty, 12) for qty in qty_values if qty > 0}
    qty_per_order = next(iter(rounded_qty)) if len(rounded_qty) == 1 else None
    if len(rounded_qty) > 1:
        conflict = True

    evidence_class = "conflicting" if conflict else "partial"
    source = ";".join(path.name for path in files)
    provenance: list[dict[str, str]] = []
    for _, row in active.sort_values(["_order_id", "_create_dt"]).iterrows():
        create_dt = cast(datetime | None, row.get("_create_dt"))
        update_dt = cast(datetime | None, row.get("_update_dt"))
        provenance.append(
            _time_provenance(
                raw_value=row.get("_create_raw_time_text"),
                source=str(row.get("_create_raw_time_source", "Time(UTC)")),
                normalized=create_dt,
            )
        )
        provenance.append(
            _time_provenance(
                raw_value=row.get("_update_raw_time_text"),
                source=str(row.get("_update_raw_time_source", "Update Time")),
                normalized=update_dt,
            )
        )
    return SeedState(
        t0=target,
        open_buy_levels=buy_levels,
        open_sell_levels=sell_levels,
        symbol=symbol.strip().upper(),
        qty_per_order=qty_per_order,
        qty_source="order_history" if qty_per_order is not None else "none",
        evidence_class=evidence_class,
        source=source,
        time_provenance=provenance,
    )
