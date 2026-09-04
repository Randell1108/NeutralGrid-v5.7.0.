"""Strict parser for Binance Futures Grid drawer telemetry.

The Chrome collector stores the drawer text verbatim.  This module converts
that UI snapshot into the existing live-decision telemetry schema without
silently inventing missing values.  Order-history rows remain labelled as UI
snapshot evidence; they are not a substitute for event-complete private API
history.
"""
from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any, Iterable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


class PrivateTelemetryParseError(ValueError):
    """The drawer contains inconsistent or structurally invalid telemetry."""


_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
_DATETIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}$")


def _lima_timezone() -> timezone | ZoneInfo:
    try:
        return ZoneInfo("America/Lima")
    except ZoneInfoNotFoundError:  # pragma: no cover - Windows tzdata fallback
        from datetime import timedelta

        return timezone(-timedelta(hours=5))


def _clean_lines(text: str) -> list[str]:
    return [
        line.replace("\xa0", " ").strip()
        for line in text.splitlines()
        if line.replace("\xa0", " ").strip()
    ]


def _find_index(lines: list[str], label: str, *, start: int = 0) -> int | None:
    expected = label.casefold()
    for index in range(start, len(lines)):
        if lines[index].casefold() == expected:
            return index
    return None


def _find_last_index(lines: list[str], label: str) -> int | None:
    expected = label.casefold()
    for index in range(len(lines) - 1, -1, -1):
        if lines[index].casefold() == expected:
            return index
    return None


def _section(
    lines: list[str],
    start_label: str,
    end_labels: Iterable[str],
    *,
    last_start: bool = False,
) -> list[str]:
    start = (
        _find_last_index(lines, start_label)
        if last_start
        else _find_index(lines, start_label)
    )
    if start is None:
        return []
    end = len(lines)
    for label in end_labels:
        candidate = _find_index(lines, label, start=start + 1)
        if candidate is not None:
            end = min(end, candidate)
    return lines[start + 1 : end]


def _label_value(lines: list[str], label: str) -> str | None:
    index = _find_index(lines, label)
    if index is None or index + 1 >= len(lines):
        return None
    return lines[index + 1]


def _numbers(value: str | None) -> tuple[float, ...]:
    if value is None:
        return ()
    parsed: list[float] = []
    for match in _NUMBER_RE.finditer(value):
        number = float(match.group(0).replace(",", ""))
        if not math.isfinite(number):
            raise PrivateTelemetryParseError(f"non-finite numeric value: {value!r}")
        parsed.append(number)
    return tuple(parsed)


def _first_number(value: str | None) -> float | None:
    values = _numbers(value)
    return values[0] if values else None


def _put(target: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        target[key] = value


def _parse_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    lowered = value.casefold()
    if lowered == "enabled":
        return True
    if lowered == "disabled":
        return False
    return None


def _parse_pnl(lines: list[str], grid_lines: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    pairs = (
        ("Total Profit (USDT)", "total_profit_usdt", "total_profit_pct"),
        ("Matched Profit (USDT)", "matched_profit_usdt", "matched_profit_pct"),
        ("Unmatched PNL (USDT)", "unmatched_pnl_usdt", "unmatched_pnl_pct"),
        ("Funding Fee (USDT)", "funding_fee_usdt", "funding_fee_pct"),
    )
    for label, amount_key, pct_key in pairs:
        values = _numbers(_label_value(lines, label))
        if values:
            result[amount_key] = values[0]
        if len(values) >= 2:
            result[pct_key] = values[1]
    _put(
        result,
        "annualized_yield_pct",
        _first_number(_label_value(lines, "Annualized Yield")),
    )
    _put(
        result,
        "transaction_fee_usdt",
        _first_number(_label_value(lines, "Transaction Fee (USDT)")),
    )
    _put(
        result,
        "realized_profit_usdt",
        _first_number(_label_value(grid_lines, "Realized Profit")),
    )
    return result


def _parse_position(lines: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not lines or any(line.casefold() == "no position" for line in lines):
        return {}, {}

    position: dict[str, Any] = {}
    risk: dict[str, Any] = {}
    if lines and re.fullmatch(r"[A-Za-z0-9]+USDT", lines[0], re.IGNORECASE):
        position["symbol"] = lines[0].upper()
    if len(lines) > 1 and lines[1].casefold() in {"perp", "perpetual"}:
        position["contract"] = lines[1]

    simple_fields = (
        ("Isolated Margin Balance", "isolated_margin_balance_usdt"),
        ("Maintenance Margin", "maintenance_margin_usdt"),
        ("Margin (USDT)", "margin_usdt"),
        ("Entry Price (USDT)", "entry_price"),
        ("Margin Ratio", "margin_ratio_pct"),
        ("Est. Liq. Price", "liquidation_price"),
        ("Mark Price (USDT)", "mark_price"),
    )
    for label, key in simple_fields:
        _put(position, key, _first_number(_label_value(lines, label)))

    size_values = _numbers(_label_value(lines, "Size"))
    if size_values:
        position["size_usdt"] = size_values[0]
    if len(size_values) >= 2:
        position["size_base"] = size_values[1]

    pnl_values = _numbers(_label_value(lines, "PNL(ROE)"))
    if pnl_values:
        position["position_pnl_usdt"] = pnl_values[0]
    if len(pnl_values) >= 2:
        position["position_roe_pct"] = pnl_values[1]

    risk_value = _label_value(lines, "Bots Risk Ratio")
    risk_values = _numbers(risk_value)
    if risk_values:
        risk["risk_ratio"] = risk_values[0]
    if risk_value is not None:
        label = _NUMBER_RE.sub("", risk_value, count=1).strip()
        if label:
            risk["risk_label"] = label

    for key in (
        "margin_ratio_pct",
        "maintenance_margin_usdt",
        "isolated_margin_balance_usdt",
        "liquidation_price",
        "mark_price",
    ):
        if key in position:
            risk[key] = position[key]

    mark = position.get("mark_price")
    liquidation = position.get("liquidation_price")
    size = position.get("size_usdt")
    if (
        isinstance(mark, float)
        and isinstance(liquidation, float)
        and isinstance(size, float)
        and mark > 0
    ):
        if size >= 0:
            distance = (mark - liquidation) / mark * 100.0
        else:
            distance = (liquidation - mark) / mark * 100.0
        risk["liquidation_distance_to_mark_pct"] = distance
    return position, risk


def _parse_grid(lines: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    mode = _label_value(lines, "Mode")
    if mode is not None:
        result["mode"] = mode.casefold()
    price_range = _numbers(_label_value(lines, "Price Range"))
    if len(price_range) >= 2:
        result["price_range_lower"] = price_range[0]
        result["price_range_upper"] = price_range[1]
    simple_fields = (
        ("Number of Grids", "num_grids"),
        ("Profit Per Grid", "profit_per_grid_pct"),
        ("Invested Margin", "invested_margin_usdt"),
        ("Qty Per Order", "qty_per_order_base"),
        ("Initial Leverage", "initial_leverage"),
        ("Current Leverage", "current_leverage"),
        ("Grid Start Price", "grid_start_price"),
        ("Position Margin", "position_margin_usdt"),
        ("Margin used by open orders", "margin_used_by_open_orders_usdt"),
        ("Total Current Margin", "total_current_margin_usdt"),
    )
    for label, key in simple_fields:
        _put(result, key, _first_number(_label_value(lines, label)))
    return result


def _parse_ladder(
    lines: list[str],
    *,
    grid_lower: float | None,
    grid_upper: float | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    qty = _first_number(_label_value(lines, "Qty Per Order"))
    last_price = _first_number(_label_value(lines, "Last Price"))
    _put(result, "qty_per_order_base", qty)
    _put(result, "last_price", last_price)

    buy_count = None
    sell_count = None
    for line in lines:
        buy_match = re.fullmatch(r"Buy\((\d+)\)", line, re.IGNORECASE)
        sell_match = re.fullmatch(r"Sell\((\d+)\)", line, re.IGNORECASE)
        if buy_match:
            buy_count = int(buy_match.group(1))
        if sell_match:
            sell_count = int(sell_match.group(1))
    if buy_count is None or sell_count is None:
        return result

    price_header = _find_index(lines, "Price (USDT)")
    if price_header is None:
        raise PrivateTelemetryParseError("pending-order ladder is missing Price (USDT)")
    candidates: list[float] = []
    max_level = max(buy_count, sell_count)
    for line in lines[price_header + 1 :]:
        if "%" in line or re.fullmatch(r"\d+", line):
            continue
        values = _numbers(line)
        if len(values) != 1:
            continue
        value = values[0]
        if value <= 0 or (value.is_integer() and value <= max_level):
            continue
        if grid_lower is not None and value < grid_lower:
            continue
        if grid_upper is not None and value > grid_upper:
            continue
        candidates.append(value)

    if last_price is None:
        raise PrivateTelemetryParseError("pending-order ladder is missing Last Price")
    buy_prices = sorted((value for value in candidates if value < last_price), reverse=True)
    sell_prices = sorted(value for value in candidates if value >= last_price)
    if len(buy_prices) != buy_count or len(sell_prices) != sell_count:
        raise PrivateTelemetryParseError(
            "pending-order ladder count mismatch: "
            f"declared buy={buy_count}/sell={sell_count}, "
            f"parsed buy={len(buy_prices)}/sell={len(sell_prices)}"
        )

    def entries(side: str, values: list[float]) -> list[dict[str, Any]]:
        return [
            {
                "side": side,
                "level": level,
                "price": price,
                "pct_to_fill": (price / last_price - 1.0) * 100.0,
                **({"qty_base": qty} if qty is not None else {}),
            }
            for level, price in enumerate(values, start=1)
        ]

    result["buy"] = entries("buy", buy_prices)
    result["sell"] = entries("sell", sell_prices)
    return result


def _parse_tp_sl(lines: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label, key in (("Stop Loss", "stop_loss"), ("Take Profit", "take_profit")):
        raw = _label_value(lines, label)
        values = _numbers(raw)
        if values:
            leg: dict[str, Any] = {"pnl_usdt": values[0]}
            if len(values) >= 2:
                leg["roi_pct"] = values[1]
            if raw is not None:
                price_type = re.search(r"\(([^()]+)\)\s*$", raw)
                if price_type:
                    leg["price_type"] = price_type.group(1)
            result[key] = leg
    _put(
        result,
        "close_all_positions_on_stop",
        _parse_bool(_label_value(lines, "Close all positions on stop")),
    )
    _put(
        result,
        "close_all_positions_on_tp_sl_stop",
        _parse_bool(_label_value(lines, "Close all positions on TP/SL stop")),
    )
    return result


def _parse_order_history(lines: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {"source_scope": "binance_ui_snapshot"}
    _put(
        result,
        "total_matched_profit_usdt",
        _first_number(_label_value(lines, "Total Matched Profit")),
    )
    _put(
        result,
        "matched_trades_24h",
        _first_number(_label_value(lines, "24H Matched Trades")),
    )
    _put(
        result,
        "total_matched_trades",
        _first_number(_label_value(lines, "Total Matched Trades")),
    )

    entries: list[dict[str, Any]] = []
    for index, line in enumerate(lines[:-1]):
        if not _DATETIME_RE.fullmatch(line):
            continue
        try:
            timestamp = (
                datetime.strptime(line, "%Y-%m-%d %H:%M:%S")
                .replace(tzinfo=_lima_timezone())
                .astimezone(timezone.utc)
            )
        except ValueError as exc:  # pragma: no cover - regex guards structure
            raise PrivateTelemetryParseError(f"invalid order-history time: {line}") from exc
        detail = lines[index + 1]
        entry: dict[str, Any] = {"timestamp_raw": line, "timestamp": timestamp.isoformat()}
        values = _numbers(detail)
        if values and "USDT" in detail.upper():
            entry["matched_profit_usdt"] = values[0]
        else:
            entry["status"] = detail
        entries.append(entry)
    result["entries"] = entries
    return result


def parse_private_telemetry_text(text: str) -> dict[str, Any]:
    """Parse a complete Binance Futures Grid drawer into scanner telemetry."""

    lines = _clean_lines(text)
    if not lines:
        raise PrivateTelemetryParseError("private telemetry drawer is empty")

    position_lines = _section(lines, "Positions", ("Pending Order",), last_start=True)
    pending_lines = _section(lines, "Pending Order", ("Grid Details",), last_start=True)
    grid_lines = _section(lines, "Grid Details", ("Advanced (Optional)",), last_start=True)
    advanced_lines = _section(lines, "Advanced (Optional)", ("History",), last_start=True)
    history_lines = _section(lines, "History", (), last_start=True)

    grid = _parse_grid(grid_lines)
    position, risk = _parse_position(position_lines)
    if any(line.casefold() == "isolated" for line in lines[:20]):
        position["margin_mode"] = "Isolated"
    ladder = _parse_ladder(
        pending_lines,
        grid_lower=grid.get("price_range_lower"),
        grid_upper=grid.get("price_range_upper"),
    )
    return {
        "pnl": _parse_pnl(lines, grid_lines),
        "position_inventory": position,
        "risk": risk,
        "open_order_ladder": ladder,
        "grid": grid,
        "tp_sl": _parse_tp_sl(advanced_lines),
        "order_history": _parse_order_history(history_lines),
    }
