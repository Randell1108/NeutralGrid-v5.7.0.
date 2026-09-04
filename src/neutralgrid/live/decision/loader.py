"""Live-bot YAML loader.

Reads active-bot YAML registries and produces a list of `LiveBotSpec`s.
See LIVE_DECISION.md for the schema.

Notes
-----
- Legacy dated registries use the **parsed date in the filename**, not mtime,
  so saving an older file does not silently regress the live registry.
- A whole-file parse error raises `LoaderError`; per-bot validation errors are
  collected as `LoaderWarning`s (the rest of the file still loads).
- `deploy_ts > now` produces a `bot_in_future` warning (skip until the
  timestamp passes); never an error.
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

from neutralgrid.live.decision.l2_risk import L2StreamRef
from neutralgrid.live.decision.private_events import PrivateEventStreamRef

logger = logging.getLogger(__name__)

# DD-MM-YY.yaml or .yml. Year is 2-digit -> 20YY.
_FILENAME_DATE_RE = re.compile(r"^(\d{2})-(\d{2})-(\d{2})\.ya?ml$", re.IGNORECASE)
_YAML_SUFFIXES = (".yaml", ".yml")

_REQUIRED_KEYS: tuple[str, ...] = (
    "symbol",
    "deploy_ts",
    "grid_lower",
    "grid_upper",
    "num_grids",
    "leverage",
    "capital_usdt",
)


@dataclass(frozen=True)
class PnlTelemetry:
    total_profit_usdt: Optional[float] = None
    total_profit_pct: Optional[float] = None
    matched_profit_usdt: Optional[float] = None
    matched_profit_pct: Optional[float] = None
    realized_profit_usdt: Optional[float] = None
    unmatched_pnl_usdt: Optional[float] = None
    unmatched_pnl_pct: Optional[float] = None
    funding_fee_usdt: Optional[float] = None
    funding_fee_pct: Optional[float] = None
    annualized_yield_pct: Optional[float] = None
    transaction_fee_usdt: Optional[float] = None


@dataclass(frozen=True)
class OpenOrderLadderEntry:
    side: str
    level: int
    price: float
    pct_to_fill: Optional[float] = None
    qty_base: Optional[float] = None


@dataclass(frozen=True)
class OpenOrderLadder:
    qty_per_order_base: Optional[float] = None
    last_price: Optional[float] = None
    buy: tuple[OpenOrderLadderEntry, ...] = ()
    sell: tuple[OpenOrderLadderEntry, ...] = ()


@dataclass(frozen=True)
class PositionInventory:
    symbol: Optional[str] = None
    contract: Optional[str] = None
    margin_mode: Optional[str] = None
    isolated_margin_balance_usdt: Optional[float] = None
    maintenance_margin_usdt: Optional[float] = None
    size_usdt: Optional[float] = None
    size_base: Optional[float] = None
    margin_usdt: Optional[float] = None
    entry_price: Optional[float] = None
    position_pnl_usdt: Optional[float] = None
    position_roe_pct: Optional[float] = None
    margin_ratio_pct: Optional[float] = None
    liquidation_price: Optional[float] = None
    mark_price: Optional[float] = None


@dataclass(frozen=True)
class RiskTelemetry:
    risk_ratio: Optional[float] = None
    risk_label: Optional[str] = None
    margin_ratio_pct: Optional[float] = None
    maintenance_margin_usdt: Optional[float] = None
    isolated_margin_balance_usdt: Optional[float] = None
    liquidation_price: Optional[float] = None
    mark_price: Optional[float] = None
    liquidation_distance_to_mark_pct: Optional[float] = None


@dataclass(frozen=True)
class TpSlLeg:
    pnl_usdt: Optional[float] = None
    roi_pct: Optional[float] = None
    price_type: Optional[str] = None


@dataclass(frozen=True)
class TpSlTelemetry:
    stop_loss: Optional[TpSlLeg] = None
    take_profit: Optional[TpSlLeg] = None
    close_all_positions_on_stop: Optional[bool] = None
    close_all_positions_on_tp_sl_stop: Optional[bool] = None


@dataclass(frozen=True)
class GridExecutionTelemetry:
    mode: Optional[str] = None
    price_range_lower: Optional[float] = None
    price_range_upper: Optional[float] = None
    num_grids: Optional[int] = None
    profit_per_grid_pct: Optional[float] = None
    invested_margin_usdt: Optional[float] = None
    qty_per_order_base: Optional[float] = None
    initial_leverage: Optional[float] = None
    current_leverage: Optional[float] = None
    grid_start_price: Optional[float] = None
    position_margin_usdt: Optional[float] = None
    margin_used_by_open_orders_usdt: Optional[float] = None
    total_current_margin_usdt: Optional[float] = None


@dataclass(frozen=True)
class OrderHistoryEntry:
    timestamp: Optional[datetime] = None
    timestamp_raw: Optional[str] = None
    matched_profit_usdt: Optional[float] = None
    status: Optional[str] = None


@dataclass(frozen=True)
class OrderHistoryTelemetry:
    source_scope: Optional[str] = None
    total_matched_profit_usdt: Optional[float] = None
    matched_trades_24h: Optional[int] = None
    total_matched_trades: Optional[int] = None
    entries: tuple[OrderHistoryEntry, ...] = ()


@dataclass(frozen=True)
class LiveExecutionTelemetry:
    source: Optional[str] = None
    captured_at: Optional[datetime] = None
    pnl: Optional[PnlTelemetry] = None
    open_order_ladder: Optional[OpenOrderLadder] = None
    position_inventory: Optional[PositionInventory] = None
    risk: Optional[RiskTelemetry] = None
    tp_sl: Optional[TpSlTelemetry] = None
    grid: Optional[GridExecutionTelemetry] = None
    order_history: Optional[OrderHistoryTelemetry] = None

    def to_dict(self) -> dict[str, Any]:
        return _strip_none(asdict(self))


@dataclass(frozen=True)
class LiveBotSpec:
    symbol: str
    strategy_id: Optional[str]
    deploy_ts: datetime  # tz-aware UTC
    grid_lower: float
    grid_upper: float
    num_grids: int
    leverage: float
    capital_usdt: float
    candidate_id: Optional[str]
    execution_telemetry: Optional[LiveExecutionTelemetry] = None
    l2_stream: Optional[L2StreamRef] = None
    private_event_stream: Optional[PrivateEventStreamRef] = None

    @property
    def grid_width(self) -> float:
        return self.grid_upper - self.grid_lower

    @property
    def state_key(self) -> str:
        """Stable per-bot identifier for state-store filenames."""
        if self.strategy_id:
            return self.strategy_id
        return f"{self.symbol}__{self.deploy_ts.isoformat()}"


@dataclass(frozen=True)
class LoaderWarning:
    code: str
    message: str
    bot_index: Optional[int] = None


class LoaderError(Exception):
    """Raised on whole-file parse / structure errors. Per-bot errors yield warnings."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class _BotInFutureError(Exception):
    pass


def find_latest_yaml(active_bots_dir: Path) -> Optional[Path]:
    """Return the file in *active_bots_dir* whose filename parses to the latest date.

    Filenames must match `DD-MM-YY.yaml` (or `.yml`); other files are ignored.
    Returns ``None`` if the directory does not exist or no matching files are present.
    """
    if not active_bots_dir.is_dir():
        return None

    candidates: list[tuple[date, Path]] = []
    for entry in active_bots_dir.iterdir():
        if not entry.is_file():
            continue
        m = _FILENAME_DATE_RE.match(entry.name)
        if not m:
            continue
        try:
            dd, mm, yy = int(m.group(1)), int(m.group(2)), int(m.group(3))
            parsed = date(2000 + yy, mm, dd)
        except ValueError:
            # Invalid date like 31-02-25 -- silently skip
            continue
        candidates.append((parsed, entry))

    if not candidates:
        return None

    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


def is_dated_registry_yaml(path: Path) -> bool:
    """Return True when *path* is named like a legacy DD-MM-YY registry."""

    return bool(_FILENAME_DATE_RE.match(path.name))


def find_active_yaml_paths(active_bots_dir: Path) -> list[Path]:
    """Return active registry YAML paths in scanner load order.

    Preferred format is one active symbol per file under ``active_bots_dir``
    (for example ``RENDERUSDT.yaml``). If no per-symbol files exist, fall back
    to the previous latest ``DD-MM-YY.yaml`` behavior for compatibility.
    """
    if not active_bots_dir.is_dir():
        return []

    symbol_files = [
        entry
        for entry in active_bots_dir.iterdir()
        if entry.is_file()
        and entry.suffix.lower() in _YAML_SUFFIXES
        and not is_dated_registry_yaml(entry)
    ]
    if symbol_files:
        return sorted(symbol_files, key=lambda p: p.name.upper())

    latest = find_latest_yaml(active_bots_dir)
    return [latest] if latest is not None else []


def load_bot_specs(
    yaml_path: Path,
    *,
    now: Optional[datetime] = None,
) -> tuple[list[LiveBotSpec], list[LoaderWarning]]:
    """Parse a YAML file into `LiveBotSpec`s.

    Returns
    -------
    (specs, warnings)
        ``specs`` contains successfully validated bots (in source order).
        ``warnings`` contains per-bot validation issues (loader_error,
        bot_in_future); these bots are excluded from ``specs``.

    Raises
    ------
    LoaderError
        On whole-file YAML parse failure or missing/invalid top-level structure.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    text = yaml_path.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise LoaderError("yaml_parse_failed", str(e)) from e

    if data is None:
        return [], []
    if isinstance(data, dict) and data.get("data_class") == "live_bot_telemetry":
        try:
            spec = _parse_bot(_live_telemetry_yaml_to_bot_entry(data), now=now)
        except _BotInFutureError as e:
            return [], [LoaderWarning(code="bot_in_future", message=str(e), bot_index=0)]
        except (KeyError, ValueError, TypeError) as e:
            return [], [LoaderWarning(code="loader_error", message=str(e), bot_index=0)]
        return [spec], []

    if not isinstance(data, dict) or "bots" not in data:
        raise LoaderError(
            "yaml_missing_bots_key",
            f"YAML at {yaml_path} must have a top-level 'bots' list "
            "or data_class: live_bot_telemetry",
        )

    bots = data["bots"]
    if not isinstance(bots, list):
        raise LoaderError("yaml_bots_not_list", "'bots' must be a list")

    specs: list[LiveBotSpec] = []
    warnings: list[LoaderWarning] = []

    for i, raw in enumerate(bots):
        try:
            spec = _parse_bot(raw, now=now)
        except _BotInFutureError as e:
            warnings.append(LoaderWarning(code="bot_in_future", message=str(e), bot_index=i))
            continue
        except (KeyError, ValueError, TypeError) as e:
            warnings.append(LoaderWarning(code="loader_error", message=str(e), bot_index=i))
            continue
        specs.append(spec)

    return specs, warnings


def _parse_bot(raw: Any, *, now: datetime) -> LiveBotSpec:
    if not isinstance(raw, dict):
        raise TypeError(f"bot entry must be a mapping, got {type(raw).__name__}")

    missing = [k for k in _REQUIRED_KEYS if k not in raw]
    if missing:
        raise KeyError(f"missing required keys: {sorted(missing)}")

    deploy_ts = _coerce_deploy_ts(raw["deploy_ts"])
    if deploy_ts > now:
        raise _BotInFutureError(
            f"deploy_ts {deploy_ts.isoformat()} is in the future (now={now.isoformat()})"
        )

    grid_lower = float(raw["grid_lower"])
    grid_upper = float(raw["grid_upper"])
    if not (grid_lower < grid_upper):
        raise ValueError(f"grid_lower ({grid_lower}) must be < grid_upper ({grid_upper})")
    if grid_lower <= 0:
        raise ValueError(f"grid_lower must be positive, got {grid_lower}")

    num_grids = int(raw["num_grids"])
    if num_grids < 1:
        raise ValueError(f"num_grids must be >= 1, got {num_grids}")

    leverage = float(raw["leverage"])
    if leverage <= 0:
        raise ValueError(f"leverage must be positive, got {leverage}")

    capital_usdt = float(raw["capital_usdt"])
    if capital_usdt <= 0:
        raise ValueError(f"capital_usdt must be positive, got {capital_usdt}")

    symbol_raw = raw["symbol"]
    if not isinstance(symbol_raw, str) or not symbol_raw.strip():
        raise ValueError("symbol must be a non-empty string")
    symbol = symbol_raw.strip().upper()

    strategy_id_raw = raw.get("strategy_id")
    strategy_id = str(strategy_id_raw) if strategy_id_raw is not None else None

    candidate_id_raw = raw.get("candidate_id")
    candidate_id = str(candidate_id_raw) if candidate_id_raw is not None else None
    execution_telemetry = _parse_execution_telemetry(raw.get("execution_telemetry"))
    l2_stream = _parse_l2_stream_ref(
        raw.get("l2_stream"),
        symbol=symbol,
        strategy_id=strategy_id,
    )
    private_event_stream = _parse_private_event_stream_ref(
        raw.get("private_event_stream"),
        symbol=symbol,
        strategy_id=strategy_id,
    )

    return LiveBotSpec(
        symbol=symbol,
        strategy_id=strategy_id,
        deploy_ts=deploy_ts,
        grid_lower=grid_lower,
        grid_upper=grid_upper,
        num_grids=num_grids,
        leverage=leverage,
        capital_usdt=capital_usdt,
        candidate_id=candidate_id,
        execution_telemetry=execution_telemetry,
        l2_stream=l2_stream,
        private_event_stream=private_event_stream,
    )


def _live_telemetry_yaml_to_bot_entry(data: dict[str, Any]) -> dict[str, Any]:
    """Convert canonical Live/<date>/<symbol>/live_bot_data_scanner.yaml to scanner input."""
    ingestion = _require_mapping(data.get("ingestion"), "ingestion")
    bot = _require_mapping(data.get("bot"), "bot")
    grid = _require_mapping(data.get("grid"), "grid")
    pnl = _require_mapping(data.get("pnl"), "pnl")
    position = _require_mapping(data.get("position"), "position")
    risk = _require_mapping(data.get("risk"), "risk")
    ladder = _require_mapping(data.get("open_order_ladder"), "open_order_ladder")
    advanced = _require_mapping(data.get("advanced"), "advanced")

    deploy_ts = bot.get("deploy_ts")
    if deploy_ts is None:
        raise KeyError("live_bot_telemetry bot.deploy_ts is required for scanner input")

    return {
        "symbol": bot.get("symbol"),
        "strategy_id": bot.get("strategy_id"),
        "deploy_ts": deploy_ts,
        "grid_lower": grid.get("price_range_lower"),
        "grid_upper": grid.get("price_range_upper"),
        "num_grids": grid.get("num_grids"),
        "leverage": bot.get("leverage") or grid.get("current_leverage"),
        "capital_usdt": grid.get("invested_margin_usdt") or pnl.get("invested_margin_usdt"),
        "candidate_id": data.get("candidate_id"),
        "execution_telemetry": {
            "source": "live_bot_data_scanner",
            "captured_at": ingestion.get("latest_part_ingested_at"),
            "pnl": {
                "total_profit_usdt": pnl.get("total_profit_usdt"),
                "total_profit_pct": pnl.get("total_profit_pct"),
                "matched_profit_usdt": pnl.get("matched_profit_usdt"),
                "matched_profit_pct": pnl.get("matched_profit_pct"),
                "realized_profit_usdt": pnl.get("realized_profit_usdt")
                if pnl.get("realized_profit_usdt") is not None
                else grid.get("realized_profit_usdt"),
                "unmatched_pnl_usdt": pnl.get("unmatched_pnl_usdt"),
                "unmatched_pnl_pct": pnl.get("unmatched_pnl_pct"),
                "funding_fee_usdt": pnl.get("funding_fee_usdt"),
                "funding_fee_pct": pnl.get("funding_fee_pct"),
                "annualized_yield_pct": pnl.get("annualized_yield_pct"),
                "transaction_fee_usdt": pnl.get("transaction_fee_usdt"),
            },
            "open_order_ladder": {
                "qty_per_order_base": ladder.get("qty_per_order_base"),
                "last_price": ladder.get("last_price"),
                "buy": ladder.get("buy", []),
                "sell": ladder.get("sell", []),
            },
            "position_inventory": {
                "symbol": position.get("symbol"),
                "contract": position.get("contract"),
                "margin_mode": bot.get("margin_mode"),
                "isolated_margin_balance_usdt": position.get(
                    "isolated_margin_balance_usdt"
                ),
                "maintenance_margin_usdt": position.get("maintenance_margin_usdt"),
                "size_usdt": position.get("size_usdt"),
                "size_base": position.get("size_base"),
                "margin_usdt": position.get("margin_usdt"),
                "entry_price": position.get("entry_price"),
                "position_pnl_usdt": position.get("position_pnl_usdt"),
                "position_roe_pct": position.get("position_roe_pct"),
                "margin_ratio_pct": position.get("margin_ratio_pct"),
                "liquidation_price": position.get("liquidation_price"),
                "mark_price": position.get("mark_price"),
            },
            "risk": {
                "risk_ratio": risk.get("risk_ratio"),
                "risk_label": risk.get("risk_label"),
                "margin_ratio_pct": risk.get("margin_ratio_pct"),
                "maintenance_margin_usdt": risk.get("maintenance_margin_usdt"),
                "isolated_margin_balance_usdt": risk.get(
                    "isolated_margin_balance_usdt"
                ),
                "liquidation_price": risk.get("liquidation_price"),
                "mark_price": risk.get("mark_price"),
                "liquidation_distance_to_mark_pct": risk.get(
                    "liquidation_distance_to_mark_pct"
                ),
            },
            "tp_sl": {
                "stop_loss": advanced.get("stop_loss"),
                "take_profit": advanced.get("take_profit"),
                "close_all_positions_on_stop": advanced.get(
                    "close_all_positions_on_stop"
                ),
                "close_all_positions_on_tp_sl_stop": advanced.get(
                    "close_all_positions_on_tp_sl_stop"
                ),
            },
            "grid": {
                "mode": grid.get("mode"),
                "price_range_lower": grid.get("price_range_lower"),
                "price_range_upper": grid.get("price_range_upper"),
                "num_grids": grid.get("num_grids"),
                "profit_per_grid_pct": grid.get("profit_per_grid_pct"),
                "invested_margin_usdt": grid.get("invested_margin_usdt"),
                "qty_per_order_base": grid.get("qty_per_order_base"),
                "initial_leverage": grid.get("initial_leverage"),
                "current_leverage": grid.get("current_leverage"),
                "grid_start_price": grid.get("grid_start_price"),
                "position_margin_usdt": grid.get("position_margin_usdt"),
                "margin_used_by_open_orders_usdt": grid.get(
                    "margin_used_by_open_orders_usdt"
                ),
                "total_current_margin_usdt": grid.get(
                    "total_current_margin_usdt"
                ),
            },
        },
    }


def _coerce_deploy_ts(raw: Any) -> datetime:
    if isinstance(raw, datetime):
        dt = raw
    elif isinstance(raw, str):
        # Accept trailing Z (Python <3.11 compat would matter elsewhere; harmless here)
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError as e:
            raise ValueError(f"deploy_ts is not ISO-8601: {raw!r}") from e
    else:
        raise TypeError(
            f"deploy_ts must be ISO-8601 string or datetime, got {type(raw).__name__}"
        )

    if dt.tzinfo is None:
        # Naive timestamps are interpreted as UTC -- surfaces in the .yaml as a
        # plain "2026-05-06T14:30:00" without a tz suffix.
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def _parse_execution_telemetry(raw: Any) -> Optional[LiveExecutionTelemetry]:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise TypeError(
            "execution_telemetry must be a mapping when provided, "
            f"got {type(raw).__name__}"
        )

    return LiveExecutionTelemetry(
        source=_optional_str(raw, "source"),
        captured_at=_optional_datetime(raw, "captured_at"),
        pnl=_parse_pnl_telemetry(raw.get("pnl")),
        open_order_ladder=_parse_open_order_ladder(raw.get("open_order_ladder")),
        position_inventory=_parse_position_inventory(raw.get("position_inventory")),
        risk=_parse_risk_telemetry(raw.get("risk")),
        tp_sl=_parse_tp_sl_telemetry(raw.get("tp_sl")),
        grid=_parse_grid_execution_telemetry(raw.get("grid")),
        order_history=_parse_order_history_telemetry(raw.get("order_history")),
    )


def _parse_l2_stream_ref(
    raw: Any,
    *,
    symbol: str,
    strategy_id: Optional[str],
) -> Optional[L2StreamRef]:
    if raw is None:
        return None
    data = _require_mapping(raw, "l2_stream")
    feature_path = _optional_str(data, "feature_path")
    if feature_path is None:
        raise KeyError("l2_stream.feature_path is required")
    stream_symbol = _optional_str(data, "symbol") or symbol
    if stream_symbol.upper() != symbol:
        raise ValueError(
            f"l2_stream.symbol mismatch: bot={symbol}, stream={stream_symbol.upper()}"
        )
    manifest_path = _optional_str(data, "manifest_path")
    public_trade_path = _optional_str(data, "public_trade_path")
    stream_strategy = _optional_str(data, "strategy_id")
    if stream_strategy is not None and stream_strategy != strategy_id:
        raise ValueError(
            "l2_stream.strategy_id mismatch: "
            f"bot={strategy_id or 'missing'}, stream={stream_strategy}"
        )
    if public_trade_path is not None and (
        strategy_id is None or stream_strategy is None
    ):
        raise ValueError(
            "l2_stream public trades require an exact bot strategy_id"
        )
    max_age_seconds = _optional_float(data, "max_age_seconds")
    history_window_seconds = _optional_float(data, "history_window_seconds")
    deterioration_min_duration_seconds = _optional_float(
        data,
        "deterioration_min_duration_seconds",
    )
    deterioration_min_observations = _optional_int(
        data,
        "deterioration_min_observations",
    )
    deterioration_fraction = _optional_float(data, "deterioration_fraction")
    max_age = max_age_seconds if max_age_seconds is not None else 15.0
    history_window = (
        history_window_seconds if history_window_seconds is not None else 300.0
    )
    if max_age <= 0 or history_window <= 0:
        raise ValueError("l2_stream age/window settings must be positive")
    min_duration = (
        deterioration_min_duration_seconds
        if deterioration_min_duration_seconds is not None
        else 60.0
    )
    min_observations = (
        deterioration_min_observations
        if deterioration_min_observations is not None
        else 3
    )
    fraction = deterioration_fraction if deterioration_fraction is not None else 0.80
    if min_duration < 0 or min_observations < 2 or not 0.5 <= fraction <= 1.0:
        raise ValueError(
            "l2_stream deterioration settings require duration >= 0, "
            "observations >= 2, and fraction in [0.5, 1.0]"
        )
    return L2StreamRef(
        feature_path=Path(feature_path),
        symbol=symbol,
        run_id=_optional_str(data, "run_id"),
        strategy_id=stream_strategy,
        manifest_path=Path(manifest_path) if manifest_path is not None else None,
        public_trade_path=(
            Path(public_trade_path) if public_trade_path is not None else None
        ),
        max_age_seconds=max_age,
        history_window_seconds=history_window,
        deterioration_min_duration_seconds=min_duration,
        deterioration_min_observations=min_observations,
        deterioration_fraction=fraction,
    )


def _parse_private_event_stream_ref(
    raw: Any,
    *,
    symbol: str,
    strategy_id: Optional[str],
) -> Optional[PrivateEventStreamRef]:
    if raw is None:
        return None
    if strategy_id is None:
        raise ValueError("private_event_stream requires an exact bot strategy_id")
    data = _require_mapping(raw, "private_event_stream")
    event_path = _optional_str(data, "event_path")
    manifest_path = _optional_str(data, "manifest_path")
    if event_path is None or manifest_path is None:
        raise KeyError(
            "private_event_stream.event_path and manifest_path are required"
        )
    stream_symbol = (_optional_str(data, "symbol") or symbol).upper()
    stream_strategy = _optional_str(data, "strategy_id") or strategy_id
    if stream_symbol != symbol or stream_strategy != strategy_id:
        raise ValueError(
            "private_event_stream identity mismatch: "
            f"bot={symbol}/{strategy_id}, stream={stream_symbol}/{stream_strategy}"
        )
    max_age_seconds = _optional_float(data, "max_age_seconds")
    history_window_seconds = _optional_float(data, "history_window_seconds")
    max_age = max_age_seconds if max_age_seconds is not None else 600.0
    history_window = (
        history_window_seconds if history_window_seconds is not None else 86400.0
    )
    if max_age <= 0 or history_window <= 0:
        raise ValueError("private_event_stream age/window settings must be positive")
    return PrivateEventStreamRef(
        event_path=Path(event_path),
        manifest_path=Path(manifest_path),
        symbol=symbol,
        strategy_id=strategy_id,
        run_id=_optional_str(data, "run_id"),
        max_age_seconds=max_age,
        history_window_seconds=history_window,
    )


def _parse_pnl_telemetry(raw: Any) -> Optional[PnlTelemetry]:
    if raw is None:
        return None
    data = _require_mapping(raw, "execution_telemetry.pnl")
    return PnlTelemetry(
        total_profit_usdt=_optional_float(data, "total_profit_usdt"),
        total_profit_pct=_optional_float(data, "total_profit_pct"),
        matched_profit_usdt=_optional_float(data, "matched_profit_usdt"),
        matched_profit_pct=_optional_float(data, "matched_profit_pct"),
        realized_profit_usdt=_optional_float(data, "realized_profit_usdt"),
        unmatched_pnl_usdt=_optional_float(data, "unmatched_pnl_usdt"),
        unmatched_pnl_pct=_optional_float(data, "unmatched_pnl_pct"),
        funding_fee_usdt=_optional_float(data, "funding_fee_usdt"),
        funding_fee_pct=_optional_float(data, "funding_fee_pct"),
        annualized_yield_pct=_optional_float(data, "annualized_yield_pct"),
        transaction_fee_usdt=_optional_float(data, "transaction_fee_usdt"),
    )


def _parse_open_order_ladder(raw: Any) -> Optional[OpenOrderLadder]:
    if raw is None:
        return None
    data = _require_mapping(raw, "execution_telemetry.open_order_ladder")
    return OpenOrderLadder(
        qty_per_order_base=_optional_float(data, "qty_per_order_base"),
        last_price=_optional_float(data, "last_price"),
        buy=_parse_ladder_entries(data.get("buy"), side="buy"),
        sell=_parse_ladder_entries(data.get("sell"), side="sell"),
    )


def _parse_ladder_entries(raw: Any, *, side: str) -> tuple[OpenOrderLadderEntry, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise TypeError(
            f"execution_telemetry.open_order_ladder.{side} must be a list, "
            f"got {type(raw).__name__}"
        )

    out: list[OpenOrderLadderEntry] = []
    for idx, item in enumerate(raw):
        data = _require_mapping(
            item, f"execution_telemetry.open_order_ladder.{side}[{idx}]"
        )
        if "level" not in data:
            raise KeyError(
                f"execution_telemetry.open_order_ladder.{side}[{idx}] missing 'level'"
            )
        if "price" not in data:
            raise KeyError(
                f"execution_telemetry.open_order_ladder.{side}[{idx}] missing 'price'"
            )
        level = int(data["level"])
        if level < 1:
            raise ValueError(
                f"execution_telemetry.open_order_ladder.{side}[{idx}].level "
                f"must be >= 1, got {level}"
            )
        price = float(data["price"])
        if price <= 0:
            raise ValueError(
                f"execution_telemetry.open_order_ladder.{side}[{idx}].price "
                f"must be positive, got {price}"
            )
        out.append(
            OpenOrderLadderEntry(
                side=side,
                level=level,
                price=price,
                pct_to_fill=_optional_float(data, "pct_to_fill"),
                qty_base=_optional_float(data, "qty_base"),
            )
        )
    return tuple(out)


def _parse_position_inventory(raw: Any) -> Optional[PositionInventory]:
    if raw is None:
        return None
    data = _require_mapping(raw, "execution_telemetry.position_inventory")
    return PositionInventory(
        symbol=_optional_str(data, "symbol"),
        contract=_optional_str(data, "contract"),
        margin_mode=_optional_str(data, "margin_mode"),
        isolated_margin_balance_usdt=_optional_float(
            data, "isolated_margin_balance_usdt"
        ),
        maintenance_margin_usdt=_optional_float(data, "maintenance_margin_usdt"),
        size_usdt=_optional_float(data, "size_usdt"),
        size_base=_optional_float(data, "size_base"),
        margin_usdt=_optional_float(data, "margin_usdt"),
        entry_price=_optional_float(data, "entry_price"),
        position_pnl_usdt=_optional_float(data, "position_pnl_usdt"),
        position_roe_pct=_optional_float(data, "position_roe_pct"),
        margin_ratio_pct=_optional_float(data, "margin_ratio_pct"),
        liquidation_price=_optional_float(data, "liquidation_price"),
        mark_price=_optional_float(data, "mark_price"),
    )


def _parse_risk_telemetry(raw: Any) -> Optional[RiskTelemetry]:
    if raw is None:
        return None
    data = _require_mapping(raw, "execution_telemetry.risk")
    return RiskTelemetry(
        risk_ratio=_optional_float(data, "risk_ratio"),
        risk_label=_optional_str(data, "risk_label"),
        margin_ratio_pct=_optional_float(data, "margin_ratio_pct"),
        maintenance_margin_usdt=_optional_float(data, "maintenance_margin_usdt"),
        isolated_margin_balance_usdt=_optional_float(
            data, "isolated_margin_balance_usdt"
        ),
        liquidation_price=_optional_float(data, "liquidation_price"),
        mark_price=_optional_float(data, "mark_price"),
        liquidation_distance_to_mark_pct=_optional_float(
            data, "liquidation_distance_to_mark_pct"
        ),
    )


def _parse_tp_sl_telemetry(raw: Any) -> Optional[TpSlTelemetry]:
    if raw is None:
        return None
    data = _require_mapping(raw, "execution_telemetry.tp_sl")
    return TpSlTelemetry(
        stop_loss=_parse_tp_sl_leg(data.get("stop_loss"), "stop_loss"),
        take_profit=_parse_tp_sl_leg(data.get("take_profit"), "take_profit"),
        close_all_positions_on_stop=_optional_bool(
            data, "close_all_positions_on_stop"
        ),
        close_all_positions_on_tp_sl_stop=_optional_bool(
            data, "close_all_positions_on_tp_sl_stop"
        ),
    )


def _parse_tp_sl_leg(raw: Any, key: str) -> Optional[TpSlLeg]:
    if raw is None:
        return None
    data = _require_mapping(raw, f"execution_telemetry.tp_sl.{key}")
    return TpSlLeg(
        pnl_usdt=_optional_float(data, "pnl_usdt"),
        roi_pct=_optional_float(data, "roi_pct"),
        price_type=_optional_str(data, "price_type"),
    )


def _parse_grid_execution_telemetry(raw: Any) -> Optional[GridExecutionTelemetry]:
    if raw is None:
        return None
    data = _require_mapping(raw, "execution_telemetry.grid")
    return GridExecutionTelemetry(
        mode=_optional_str(data, "mode"),
        price_range_lower=_optional_float(data, "price_range_lower"),
        price_range_upper=_optional_float(data, "price_range_upper"),
        num_grids=_optional_int(data, "num_grids"),
        profit_per_grid_pct=_optional_float(data, "profit_per_grid_pct"),
        invested_margin_usdt=_optional_float(data, "invested_margin_usdt"),
        qty_per_order_base=_optional_float(data, "qty_per_order_base"),
        initial_leverage=_optional_float(data, "initial_leverage"),
        current_leverage=_optional_float(data, "current_leverage"),
        grid_start_price=_optional_float(data, "grid_start_price"),
        position_margin_usdt=_optional_float(data, "position_margin_usdt"),
        margin_used_by_open_orders_usdt=_optional_float(
            data, "margin_used_by_open_orders_usdt"
        ),
        total_current_margin_usdt=_optional_float(
            data, "total_current_margin_usdt"
        ),
    )


def _parse_order_history_telemetry(raw: Any) -> Optional[OrderHistoryTelemetry]:
    if raw is None:
        return None
    data = _require_mapping(raw, "execution_telemetry.order_history")
    entries_raw = data.get("entries")
    if entries_raw is None:
        entries_raw = []
    if not isinstance(entries_raw, list):
        raise TypeError(
            "execution_telemetry.order_history.entries must be a list, "
            f"got {type(entries_raw).__name__}"
        )
    entries: list[OrderHistoryEntry] = []
    for index, item in enumerate(entries_raw):
        entry = _require_mapping(
            item, f"execution_telemetry.order_history.entries[{index}]"
        )
        entries.append(
            OrderHistoryEntry(
                timestamp=_optional_datetime(entry, "timestamp"),
                timestamp_raw=_optional_str(entry, "timestamp_raw"),
                matched_profit_usdt=_optional_float(entry, "matched_profit_usdt"),
                status=_optional_str(entry, "status"),
            )
        )
    return OrderHistoryTelemetry(
        source_scope=_optional_str(data, "source_scope"),
        total_matched_profit_usdt=_optional_float(
            data, "total_matched_profit_usdt"
        ),
        matched_trades_24h=_optional_int(data, "matched_trades_24h"),
        total_matched_trades=_optional_int(data, "total_matched_trades"),
        entries=tuple(entries),
    )


def _require_mapping(raw: Any, name: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise TypeError(f"{name} must be a mapping, got {type(raw).__name__}")
    return raw


def _optional_float(data: dict[str, Any], key: str) -> Optional[float]:
    value = data.get(key)
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{key} must be finite when provided, got {value!r}")
    return number


def _optional_int(data: dict[str, Any], key: str) -> Optional[int]:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"{key} must be an integer when provided, got bool")
    number = int(value)
    if float(number) != float(value):
        raise ValueError(f"{key} must be an integer when provided, got {value!r}")
    return number


def _optional_str(data: dict[str, Any], key: str) -> Optional[str]:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string when provided, got {type(value).__name__}")
    cleaned = value.strip()
    return cleaned or None


def _optional_bool(data: dict[str, Any], key: str) -> Optional[bool]:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise TypeError(f"{key} must be a bool when provided, got {type(value).__name__}")
    return value


def _optional_datetime(data: dict[str, Any], key: str) -> Optional[datetime]:
    value = data.get(key)
    if value is None:
        return None
    return _coerce_deploy_ts(value)


def _strip_none(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, child in value.items():
            cleaned = _strip_none(child)
            if cleaned is not None:
                out[str(key)] = cleaned
        return out
    if isinstance(value, (list, tuple)):
        out_list: list[Any] = []
        for child in value:
            cleaned = _strip_none(child)
            if cleaned is not None:
                out_list.append(cleaned)
        return out_list
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    return value
