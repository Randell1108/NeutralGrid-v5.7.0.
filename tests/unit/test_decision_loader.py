"""Tests for live.decision.loader (Phase A)."""
from __future__ import annotations

import textwrap
from datetime import datetime, timezone
from pathlib import Path

import pytest

from neutralgrid.live.decision.loader import (
    LiveBotSpec,
    LoaderError,
    find_active_yaml_paths,
    find_latest_yaml,
    load_bot_specs,
)


def _write(path: Path, content: str) -> Path:
    path.write_text(textwrap.dedent(content).lstrip("\n"), encoding="utf-8")
    return path


_VALID_YAML = """
bots:
  - symbol: BTCUSDT
    strategy_id: "410472444"
    deploy_ts: 2026-05-01T14:30:00Z
    grid_lower: 60000.0
    grid_upper: 70000.0
    num_grids: 50
    leverage: 5
    capital_usdt: 200.0
    candidate_id: cand-abc
  - symbol: ethusdt
    deploy_ts: 2026-05-02T09:00:00+00:00
    grid_lower: 2500
    grid_upper: 3000
    num_grids: 30
    leverage: 3
    capital_usdt: 100.0
"""


def test_load_valid_yaml_two_bots(tmp_path: Path) -> None:
    yaml_path = _write(tmp_path / "06-05-26.yaml", _VALID_YAML)

    now = datetime(2026, 5, 6, tzinfo=timezone.utc)
    specs, warnings = load_bot_specs(yaml_path, now=now)

    assert warnings == []
    assert len(specs) == 2

    btc, eth = specs
    assert isinstance(btc, LiveBotSpec)
    assert btc.symbol == "BTCUSDT"
    assert btc.strategy_id == "410472444"
    assert btc.candidate_id == "cand-abc"
    assert btc.grid_lower == 60000.0
    assert btc.grid_upper == 70000.0
    assert btc.deploy_ts == datetime(2026, 5, 1, 14, 30, tzinfo=timezone.utc)
    assert btc.grid_width == pytest.approx(10000.0)
    assert btc.state_key == "410472444"

    # ETH — no strategy_id, no candidate_id, lowercase symbol → upper
    assert eth.symbol == "ETHUSDT"
    assert eth.strategy_id is None
    assert eth.candidate_id is None
    assert eth.state_key.startswith("ETHUSDT__")


def test_missing_required_key_yields_warning(tmp_path: Path) -> None:
    bad = """
    bots:
      - symbol: BTCUSDT
        # deploy_ts missing
        grid_lower: 60000
        grid_upper: 70000
        num_grids: 50
        leverage: 5
        capital_usdt: 200
    """
    yaml_path = _write(tmp_path / "06-05-26.yaml", bad)

    specs, warnings = load_bot_specs(yaml_path, now=datetime(2026, 5, 6, tzinfo=timezone.utc))
    assert specs == []
    assert len(warnings) == 1
    assert warnings[0].code == "loader_error"
    assert "deploy_ts" in warnings[0].message
    assert warnings[0].bot_index == 0


def test_invalid_grid_bounds_yields_warning(tmp_path: Path) -> None:
    bad = """
    bots:
      - symbol: BTCUSDT
        deploy_ts: 2026-05-01T00:00:00Z
        grid_lower: 70000
        grid_upper: 60000
        num_grids: 50
        leverage: 5
        capital_usdt: 200
    """
    yaml_path = _write(tmp_path / "06-05-26.yaml", bad)

    specs, warnings = load_bot_specs(yaml_path, now=datetime(2026, 5, 6, tzinfo=timezone.utc))
    assert specs == []
    assert len(warnings) == 1
    assert warnings[0].code == "loader_error"
    assert "grid_lower" in warnings[0].message


def test_deploy_ts_in_future_yields_bot_in_future_warning(tmp_path: Path) -> None:
    yaml_path = _write(tmp_path / "06-05-26.yaml", _VALID_YAML)
    # Now is BEFORE the BTC deploy_ts (2026-05-01)
    now = datetime(2026, 4, 30, tzinfo=timezone.utc)

    specs, warnings = load_bot_specs(yaml_path, now=now)
    # Both bots have deploy_ts > now
    assert specs == []
    assert len(warnings) == 2
    assert all(w.code == "bot_in_future" for w in warnings)


def test_whole_file_yaml_parse_failure_raises(tmp_path: Path) -> None:
    yaml_path = _write(tmp_path / "06-05-26.yaml", "bots: [unterminated")

    with pytest.raises(LoaderError) as exc_info:
        load_bot_specs(yaml_path)
    assert exc_info.value.code == "yaml_parse_failed"


def test_missing_top_level_bots_key_raises(tmp_path: Path) -> None:
    yaml_path = _write(tmp_path / "06-05-26.yaml", "other_key: 1")

    with pytest.raises(LoaderError) as exc_info:
        load_bot_specs(yaml_path)
    assert exc_info.value.code == "yaml_missing_bots_key"


def test_empty_yaml_file_returns_empty(tmp_path: Path) -> None:
    yaml_path = _write(tmp_path / "06-05-26.yaml", "")

    specs, warnings = load_bot_specs(yaml_path)
    assert specs == []
    assert warnings == []


def test_find_latest_yaml_picks_by_parsed_date_not_mtime(tmp_path: Path) -> None:
    # Write three files; the middle one (06-05-26) is the latest by date
    older = _write(tmp_path / "01-05-26.yaml", "bots: []")
    latest = _write(tmp_path / "06-05-26.yaml", "bots: []")
    earlier_year = _write(tmp_path / "31-12-25.yaml", "bots: []")

    # Touch them in reverse-date order so mtime != date
    import os
    import time

    base_t = time.time()
    os.utime(latest, (base_t - 100, base_t - 100))  # latest by name, oldest mtime
    os.utime(earlier_year, (base_t - 50, base_t - 50))
    os.utime(older, (base_t, base_t))  # older by name, newest mtime

    chosen = find_latest_yaml(tmp_path)
    assert chosen == latest


def test_find_latest_yaml_ignores_non_matching_filenames(tmp_path: Path) -> None:
    _write(tmp_path / "notes.txt", "ignored")
    _write(tmp_path / "2026-05-06.yaml", "bots: []")  # wrong format (YYYY-MM-DD, not DD-MM-YY)
    valid = _write(tmp_path / "06-05-26.yaml", "bots: []")

    assert find_latest_yaml(tmp_path) == valid


def test_find_latest_yaml_empty_dir(tmp_path: Path) -> None:
    assert find_latest_yaml(tmp_path) is None


def test_find_latest_yaml_missing_dir(tmp_path: Path) -> None:
    assert find_latest_yaml(tmp_path / "does-not-exist") is None


def test_find_latest_yaml_invalid_date_in_filename_skipped(tmp_path: Path) -> None:
    _write(tmp_path / "31-02-26.yaml", "bots: []")  # 31 Feb does not exist
    valid = _write(tmp_path / "01-01-26.yaml", "bots: []")

    assert find_latest_yaml(tmp_path) == valid


def test_find_active_yaml_paths_prefers_symbol_files(tmp_path: Path) -> None:
    _write(tmp_path / "25-05-26.yaml", "bots: []")
    zec = _write(tmp_path / "ZECUSDT.yaml", "bots: []")
    render = _write(tmp_path / "RENDERUSDT.yaml", "bots: []")
    _write(tmp_path / "notes.txt", "ignored")

    assert find_active_yaml_paths(tmp_path) == [render, zec]


def test_find_active_yaml_paths_falls_back_to_latest_dated_yaml(tmp_path: Path) -> None:
    _write(tmp_path / "24-05-26.yaml", "bots: []")
    latest = _write(tmp_path / "25-05-26.yaml", "bots: []")

    assert find_active_yaml_paths(tmp_path) == [latest]


def test_find_active_yaml_paths_empty_or_missing_dir(tmp_path: Path) -> None:
    assert find_active_yaml_paths(tmp_path) == []
    assert find_active_yaml_paths(tmp_path / "missing") == []


def test_naive_deploy_ts_treated_as_utc(tmp_path: Path) -> None:
    content = """
    bots:
      - symbol: BTCUSDT
        deploy_ts: "2026-05-01T14:30:00"
        grid_lower: 60000
        grid_upper: 70000
        num_grids: 50
        leverage: 5
        capital_usdt: 200
    """
    yaml_path = _write(tmp_path / "06-05-26.yaml", content)
    specs, warnings = load_bot_specs(yaml_path, now=datetime(2026, 5, 6, tzinfo=timezone.utc))
    assert warnings == []
    assert len(specs) == 1
    assert specs[0].deploy_ts.tzinfo is timezone.utc


def test_state_key_falls_back_to_symbol_and_deploy_ts(tmp_path: Path) -> None:
    content = """
    bots:
      - symbol: SOLUSDT
        deploy_ts: 2026-05-01T00:00:00Z
        grid_lower: 100
        grid_upper: 200
        num_grids: 10
        leverage: 2
        capital_usdt: 50
    """
    yaml_path = _write(tmp_path / "06-05-26.yaml", content)
    specs, _ = load_bot_specs(yaml_path, now=datetime(2026, 5, 6, tzinfo=timezone.utc))
    assert specs[0].state_key == "SOLUSDT__2026-05-01T00:00:00+00:00"


def test_execution_telemetry_schema_is_parsed(tmp_path: Path) -> None:
    content = """
    bots:
      - symbol: COSUSDT
        strategy_id: "411987721"
        deploy_ts: "2026-05-13T16:03:55+00:00"
        grid_lower: 0.001275
        grid_upper: 0.002047
        num_grids: 30
        leverage: 10
        capital_usdt: 400
        candidate_id: null
        execution_telemetry:
          source: user_provided_binance_ui
          captured_at: null
          pnl:
            realized_profit_usdt: 24.03
            matched_profit_usdt: 24.05
            matched_profit_pct: 6.01
          open_order_ladder:
            qty_per_order_base: 63284
            last_price: 0.00173
            buy:
              - level: 1
                price: 0.001693
                pct_to_fill: -2.13
            sell:
              - level: 1
                price: 0.001748
                pct_to_fill: 1.04
          position_inventory:
            symbol: COSUSDT
            contract: perp
            margin_mode: isolated
            size_usdt: -109.291468
            size_base: -63284
            liquidation_price: 0.0080195
            mark_price: 0.0017282
          risk:
            risk_ratio: 2.5
            risk_label: Low Risk
            margin_ratio_pct: 1.29
            liquidation_distance_to_mark_pct: 364.0377
          tp_sl:
            stop_loss:
              pnl_usdt: -40
              roi_pct: -10
              price_type: mark
            take_profit:
              pnl_usdt: 60
              roi_pct: 15
              price_type: mark
            close_all_positions_on_stop: true
            close_all_positions_on_tp_sl_stop: true
    """
    yaml_path = _write(tmp_path / "13-05-26.yaml", content)

    specs, warnings = load_bot_specs(
        yaml_path, now=datetime(2026, 5, 13, 18, tzinfo=timezone.utc)
    )

    assert warnings == []
    telemetry = specs[0].execution_telemetry
    assert telemetry is not None
    assert telemetry.pnl is not None
    assert telemetry.pnl.realized_profit_usdt == pytest.approx(24.03)
    assert telemetry.pnl.matched_profit_usdt == pytest.approx(24.05)
    assert telemetry.open_order_ladder is not None
    assert telemetry.open_order_ladder.qty_per_order_base == pytest.approx(63284)
    assert len(telemetry.open_order_ladder.buy) == 1
    assert telemetry.open_order_ladder.buy[0].price == pytest.approx(0.001693)
    assert telemetry.position_inventory is not None
    assert telemetry.position_inventory.size_base == pytest.approx(-63284)
    assert telemetry.risk is not None
    assert telemetry.risk.risk_label == "Low Risk"
    assert telemetry.risk.liquidation_distance_to_mark_pct == pytest.approx(364.0377)
    assert telemetry.tp_sl is not None
    assert telemetry.tp_sl.stop_loss is not None
    assert telemetry.tp_sl.stop_loss.roi_pct == pytest.approx(-10)
    assert telemetry.tp_sl.close_all_positions_on_stop is True


def test_load_canonical_live_telemetry_yaml(tmp_path: Path) -> None:
    content = """
    schema_version: 1
    data_class: live_bot_telemetry
    status: partial_more_parts_expected
    ingestion:
      ingestion_date: "2026-05-13"
      latest_part_ingested_at: "2026-05-13T20:24:23-05:00"
      symbol: DOGEUSDT
    bot:
      symbol: DOGEUSDT
      status: Working
      contract: Perp
      product: Futures Grid
      margin_mode: Isolated
      strategy_mode: Neutral
      leverage: 10
      strategy_id: "411991896"
      deploy_ts: "2026-05-13T22:12:25+00:00"
      duration: "2h 10m"
    pnl:
      total_profit_usdt: 0.05
      total_profit_pct: 0.98
      invested_margin_usdt: 6.00
      matched_profit_usdt: 0.06
      matched_profit_pct: 1.11
      realized_profit_usdt: 0.04
      unmatched_pnl_usdt: -0.01
      unmatched_pnl_pct: -0.12
      funding_fee_usdt: 0.00
      funding_fee_pct: 0.00
      annualized_yield_pct: 3978.83
    position:
      symbol: DOGEUSDT
      contract: Perp
      isolated_margin_balance_usdt: 6.0588
      maintenance_margin_usdt: 0.0348
      risk_ratio: 2.3
      risk_label: Low Risk
      size_usdt: -5.351420
      size_base: -47
      margin_usdt: 0.54
      entry_price: 0.114070
      position_pnl_usdt: 0.00
      position_roe_pct: 1.84
      margin_ratio_pct: 0.57
      liquidation_price: 0.241178
      mark_price: 0.113834
    risk:
      risk_ratio: 2.3
      risk_label: Low Risk
      margin_ratio_pct: 0.57
      maintenance_margin_usdt: 0.0348
      isolated_margin_balance_usdt: 6.0588
      liquidation_price: 0.241178
      mark_price: 0.113834
    open_order_ladder:
      qty_per_order_base: 47
      last_price: 0.11382
      buy:
        - level: 1
          price: 0.11294
          pct_to_fill: -0.77
      sell:
        - level: 1
          price: 0.11445
          pct_to_fill: 0.55
    grid:
      price_range_lower: 0.10924
      price_range_upper: 0.11522
      num_grids: 8
      invested_margin_usdt: 6.00
      current_leverage: 10
    advanced:
      stop_loss:
        pnl_usdt: -0.60
        roi_pct: -10.0
        price_type: Mark
      take_profit:
        pnl_usdt: 0.90
        roi_pct: 15.0
        price_type: Mark
      close_all_positions_on_stop: true
      close_all_positions_on_tp_sl_stop: true
    """
    yaml_path = _write(tmp_path / "live_bot_data_scanner.yaml", content)

    specs, warnings = load_bot_specs(
        yaml_path, now=datetime(2026, 5, 14, 1, 0, tzinfo=timezone.utc)
    )

    assert warnings == []
    assert len(specs) == 1
    spec = specs[0]
    assert spec.symbol == "DOGEUSDT"
    assert spec.strategy_id == "411991896"
    assert spec.grid_lower == pytest.approx(0.10924)
    assert spec.grid_upper == pytest.approx(0.11522)
    assert spec.num_grids == 8
    assert spec.capital_usdt == pytest.approx(6.0)
    assert spec.execution_telemetry is not None
    assert spec.execution_telemetry.pnl is not None
    assert spec.execution_telemetry.pnl.matched_profit_usdt == pytest.approx(0.06)
    assert spec.execution_telemetry.open_order_ladder is not None
    assert spec.execution_telemetry.open_order_ladder.buy[0].price == pytest.approx(0.11294)
