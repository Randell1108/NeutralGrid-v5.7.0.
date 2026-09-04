"""
Tests for the offline order-book replay pipeline.

Tests cover all 7 pipeline stages using synthetic deterministic data.
No network calls, no external dependencies beyond the replay package.
"""
from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

import pytest

from neutralgrid.replay.schema import (
    OrderRow,
    TradeRow,
    OrderLifecycle,
    Event,
    BookSnapshot,
    Anomaly,
    OPEN_SENTINEL_MS,
)
from neutralgrid.replay.io_scan import scan_directory, ScanResult
from neutralgrid.replay.normalize import (
    load_orders,
    load_trades,
    apply_window_filter,
    load_sessions,
    detect_overlaps,
    filter_overlapping_orders,
)
from neutralgrid.replay.join_trades import build_lifecycles
from neutralgrid.replay.events import build_events, split_events_by_symbol
from neutralgrid.replay.sweep import replay_symbol, replay_all
from neutralgrid.replay.export import export_all
from neutralgrid.replay.cli import run_pipeline


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_exports(tmp_path: Path) -> Path:
    """Create a temporary directory with minimal order and trade CSVs."""
    # Order CSV
    order_file = tmp_path / "orders.csv"
    with open(order_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "Order No", "Symbol", "Side", "Type", "Price", "Amount",
            "Executed Amount", "Status", "Time(UTC)", "Update Time",
            "Client Order Id",
        ])
        # Order 1: FILLED, has matching trades
        w.writerow([
            "1001", "BTCUSDT", "BUY", "LIMIT", "50000.0", "0.1",
            "0.1", "FILLED", "2026-01-15 10:00:00", "2026-01-15 10:05:00",
            "grid_buy_1",
        ])
        # Order 2: CANCELED
        w.writerow([
            "1002", "BTCUSDT", "SELL", "LIMIT", "51000.0", "0.1",
            "0.0", "CANCELED", "2026-01-15 10:00:00", "2026-01-15 12:00:00",
            "grid_sell_1",
        ])
        # Order 3: NEW (still open)
        w.writerow([
            "1003", "ETHUSDT", "BUY", "LIMIT", "3000.0", "1.0",
            "0.0", "NEW", "2026-01-15 11:00:00", "2026-01-15 11:00:00",
            "grid_buy_eth",
        ])
        # Order 4: FILLED, no matching trades
        w.writerow([
            "1004", "ETHUSDT", "SELL", "LIMIT", "3200.0", "1.0",
            "1.0", "FILLED", "2026-01-15 11:00:00", "2026-01-15 14:00:00",
            "grid_sell_eth",
        ])

    # Trade CSV
    trade_file = tmp_path / "trades.csv"
    with open(trade_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Order Id", "Symbol", "Side", "Price", "Quantity",
                     "Time(UTC)"])
        # Two fills for order 1001
        w.writerow(["1001", "BTCUSDT", "BUY", "50000.0", "0.05",
                     "2026-01-15 10:02:00"])
        w.writerow(["1001", "BTCUSDT", "BUY", "50000.0", "0.05",
                     "2026-01-15 10:04:00"])

    return tmp_path


@pytest.fixture
def tmp_sessions(tmp_path: Path) -> Path:
    """Create a sessions CSV for overlap testing."""
    sess_file = tmp_path / "sessions.csv"
    with open(sess_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "start_time_utc", "end_time_utc", "strategy_id"])
        # Two overlapping BTCUSDT bots
        w.writerow(["BTCUSDT", "2026-01-15 08:00:00",
                     "2026-01-15 14:00:00", "bot_a"])
        w.writerow(["BTCUSDT", "2026-01-15 12:00:00",
                     "2026-01-15 20:00:00", "bot_b"])
        # Non-overlapping ETHUSDT bots
        w.writerow(["ETHUSDT", "2026-01-15 08:00:00",
                     "2026-01-15 12:00:00", "bot_c"])
        w.writerow(["ETHUSDT", "2026-01-15 14:00:00",
                     "2026-01-15 20:00:00", "bot_d"])
    return sess_file


# ── Test: io_scan ────────────────────────────────────────────────────────

class TestIOScan:
    def test_classifies_orders_and_trades(self, tmp_exports: Path):
        result = scan_directory(tmp_exports)
        assert len(result.order_files) == 1
        assert len(result.trade_files) == 1
        assert len(result.unknown_files) == 0

    def test_unknown_csv_ignored(self, tmp_path: Path):
        # write a CSV with unrecognized columns
        bad = tmp_path / "random.csv"
        with open(bad, "w", newline="") as f:
            csv.writer(f).writerow(["col_a", "col_b"])
        result = scan_directory(tmp_path)
        assert len(result.unknown_files) == 1
        assert len(result.order_files) == 0

    def test_missing_dir_raises(self):
        with pytest.raises(FileNotFoundError):
            scan_directory(Path("/nonexistent_dir_xyz"))


# ── Test: normalize ──────────────────────────────────────────────────────

class TestNormalize:
    def test_load_orders(self, tmp_exports: Path):
        orders = load_orders([tmp_exports / "orders.csv"])
        assert len(orders) == 4
        assert orders[0].symbol == "BTCUSDT"
        assert orders[0].side == "BUY"
        assert orders[0].price == 50000.0
        assert orders[0].t_create_ms > 0

    def test_load_trades(self, tmp_exports: Path):
        trades = load_trades([tmp_exports / "trades.csv"])
        assert len(trades) == 2
        assert trades[0].order_id == 1001
        assert trades[0].qty == 0.05

    def test_load_orders_deduplicates_by_order_id_prefers_latest_row(self, tmp_path: Path):
        first = tmp_path / "orders_a.csv"
        second = tmp_path / "orders_b.csv"

        with open(first, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "Order No", "Symbol", "Side", "Type", "Price", "Amount",
                "Executed Amount", "Status", "Time(UTC)", "Update Time",
                "Client Order Id",
            ])
            w.writerow([
                "2001", "APTUSDT", "BUY", "LIMIT", "1.00", "10",
                "0", "NEW", "2026-02-25 22:53:12", "2026-02-25 22:53:12",
                "grid_buy_apt",
            ])

        with open(second, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "Order No", "Symbol", "Side", "Type", "Price", "Amount",
                "Executed Amount", "Status", "Time(UTC)", "Update Time",
                "Client Order Id",
            ])
            w.writerow([
                "2001", "APTUSDT", "BUY", "LIMIT", "1.00", "10",
                "10", "FILLED", "2026-02-25 22:53:12", "2026-02-27 22:34:29",
                "grid_buy_apt",
            ])

        orders = load_orders([first, second])
        assert len(orders) == 1
        assert orders[0].order_id == 2001
        assert orders[0].status == "FILLED"
        assert orders[0].executed_qty == 10.0

    def test_load_trades_deduplicates_exact_rows_only(self, tmp_path: Path):
        first = tmp_path / "trades_a.csv"
        second = tmp_path / "trades_b.csv"

        header = [
            "Order Id", "Symbol", "Side", "Price", "Quantity", "Time(UTC)",
            "Fee", "Realized Profit", "Maker",
        ]
        with open(first, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerow([
                "3001", "KAITOUSDT", "SELL", "0.7226", "27.7",
                "2026-01-13 23:00:16", "-0.00400320 USDT", "0E-8", "true",
            ])

        with open(second, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerow([
                "3001", "KAITOUSDT", "SELL", "0.7226", "27.7",
                "2026-01-13 23:00:16", "-0.00400320 USDT", "0E-8", "true",
            ])
            w.writerow([
                "3001", "KAITOUSDT", "SELL", "0.7226", "4.3",
                "2026-01-13 23:00:16", "-0.00062143 USDT", "0.02107000", "true",
            ])

        trades = load_trades([first, second])
        assert len(trades) == 2
        qtys = sorted(t.qty for t in trades)
        assert qtys == [4.3, 27.7]

    def test_window_filter(self, tmp_exports: Path):
        orders = load_orders([tmp_exports / "orders.csv"])
        trades = load_trades([tmp_exports / "trades.csv"])
        fo, ft, t_start, t_end = apply_window_filter(orders, trades, 90)
        assert len(fo) == 4  # all within 90 days of each other
        assert t_end > t_start

    def test_window_filter_empty(self):
        fo, ft, ts, te = apply_window_filter([], [], 90)
        assert fo == []
        assert ft == []


# ── Test: overlap detection ──────────────────────────────────────────────

class TestOverlap:
    def test_detect_overlaps(self, tmp_sessions: Path):
        sessions = load_sessions(tmp_sessions)
        assert len(sessions) == 4

        overlaps = detect_overlaps(sessions)
        # Only BTCUSDT should have an overlap
        assert len(overlaps) == 1
        assert overlaps[0].symbol == "BTCUSDT"

    def test_filter_overlapping_orders(self, tmp_exports: Path,
                                       tmp_sessions: Path):
        orders = load_orders([tmp_exports / "orders.csv"])
        sessions = load_sessions(tmp_sessions)
        overlaps = detect_overlaps(sessions)

        filtered, discarded = filter_overlapping_orders(orders, overlaps)
        # BTC orders that overlap the 12:00-14:00 window should be removed
        assert discarded > 0
        # ETH orders should be untouched
        eth_orders = [o for o in filtered if o.symbol == "ETHUSDT"]
        assert len(eth_orders) == 2


# ── Test: join_trades ────────────────────────────────────────────────────

class TestJoinTrades:
    def test_lifecycle_close_sources(self, tmp_exports: Path):
        orders = load_orders([tmp_exports / "orders.csv"])
        trades = load_trades([tmp_exports / "trades.csv"])
        lifecycles, anomalies = build_lifecycles(orders, trades)

        lc_map = {lc.order.order_id: lc for lc in lifecycles}

        # Order 1001: FILLED with trades -> close from trade
        assert lc_map[1001].close_source == "trade"
        assert lc_map[1001].trade_count == 2

        # Order 1002: CANCELED -> close from update
        assert lc_map[1002].close_source == "update"

        # Order 1003: NEW -> still open
        assert lc_map[1003].is_still_open

        # Order 1004: FILLED without trades -> close from update
        assert lc_map[1004].close_source == "update"

    def test_no_anomalies_when_qty_matches(self, tmp_exports: Path):
        orders = load_orders([tmp_exports / "orders.csv"])
        trades = load_trades([tmp_exports / "trades.csv"])
        _, anomalies = build_lifecycles(orders, trades)
        assert len(anomalies) == 0  # 0.05 + 0.05 == 0.1


# ── Test: events ─────────────────────────────────────────────────────────

class TestEvents:
    def test_event_count(self, tmp_exports: Path):
        orders = load_orders([tmp_exports / "orders.csv"])
        trades = load_trades([tmp_exports / "trades.csv"])
        lifecycles, _ = build_lifecycles(orders, trades)

        _, _, _, t_end = apply_window_filter(orders, trades, 90)
        events = build_events(lifecycles, t_end_ms=t_end)

        # 4 orders -> 4 ADD + 4 REMOVE (including capped NEW)
        assert len(events) == 8

    def test_remove_before_add_at_same_time(self, tmp_exports: Path):
        orders = load_orders([tmp_exports / "orders.csv"])
        trades = load_trades([tmp_exports / "trades.csv"])
        lifecycles, _ = build_lifecycles(orders, trades)
        events = build_events(lifecycles, t_end_ms=None)

        # Check ordering: at any timestamp, REMOVE should precede ADD
        for i in range(len(events) - 1):
            if events[i].time_ms == events[i + 1].time_ms:
                if events[i].event_type == "ADD":
                    assert events[i + 1].event_type == "ADD"

    def test_split_by_symbol(self, tmp_exports: Path):
        orders = load_orders([tmp_exports / "orders.csv"])
        trades = load_trades([tmp_exports / "trades.csv"])
        lifecycles, _ = build_lifecycles(orders, trades)
        events = build_events(lifecycles, t_end_ms=99999999999999)

        by_sym = split_events_by_symbol(events)
        assert "BTCUSDT" in by_sym
        assert "ETHUSDT" in by_sym


# ── Test: sweep ──────────────────────────────────────────────────────────

class TestSweep:
    def test_replay_produces_snapshots(self, tmp_exports: Path):
        orders = load_orders([tmp_exports / "orders.csv"])
        trades = load_trades([tmp_exports / "trades.csv"])
        lifecycles, _ = build_lifecycles(orders, trades)
        events = build_events(lifecycles, t_end_ms=99999999999999)
        by_sym = split_events_by_symbol(events)

        result = replay_all(by_sym)
        assert "BTCUSDT" in result
        assert len(result["BTCUSDT"].snapshots) > 0

    def test_snapshot_counts_correct(self, tmp_exports: Path):
        orders = load_orders([tmp_exports / "orders.csv"])
        trades = load_trades([tmp_exports / "trades.csv"])
        lifecycles, _ = build_lifecycles(orders, trades)
        events = build_events(lifecycles, t_end_ms=99999999999999)
        by_sym = split_events_by_symbol(events)

        btc_result = replay_all(by_sym)["BTCUSDT"]
        # Check first snapshot (after first events): should have orders open
        first = btc_result.snapshots[0]
        assert first.open_order_count >= 1


# ── Test: export ─────────────────────────────────────────────────────────

class TestExport:
    def test_full_export(self, tmp_exports: Path, tmp_path: Path):
        orders = load_orders([tmp_exports / "orders.csv"])
        trades = load_trades([tmp_exports / "trades.csv"])
        lifecycles, anomalies = build_lifecycles(orders, trades)
        events = build_events(lifecycles, t_end_ms=99999999999999)
        by_sym = split_events_by_symbol(events)
        results = replay_all(by_sym)

        out_dir = tmp_path / "output"
        symbol_dirs = export_all(results, trades, anomalies, out_dir)

        assert "BTCUSDT" in symbol_dirs
        btc_dir = symbol_dirs["BTCUSDT"]
        assert (btc_dir / "events.csv").exists()
        assert (btc_dir / "snapshots.csv").exists()
        assert (btc_dir / "levels.csv").exists()
        assert (btc_dir / "trades.csv").exists()


# ── Test: full pipeline (CLI) ────────────────────────────────────────────

class TestCLI:
    def test_run_pipeline_end_to_end(self, tmp_exports: Path, tmp_path: Path):
        out_dir = tmp_path / "pipeline_out"
        result = run_pipeline(
            input_dir=tmp_exports,
            output_dir=out_dir,
            window_days=90,
            cap_open=True,
        )
        assert len(result) == 2  # BTCUSDT + ETHUSDT
        assert (out_dir / "BTCUSDT" / "events.csv").exists()
        assert (out_dir / "ETHUSDT" / "snapshots.csv").exists()

    def test_run_pipeline_with_sessions(self, tmp_exports: Path,
                                         tmp_sessions: Path,
                                         tmp_path: Path):
        out_dir = tmp_path / "pipeline_sess_out"
        result = run_pipeline(
            input_dir=tmp_exports,
            output_dir=out_dir,
            sessions_path=tmp_sessions,
            window_days=90,
            cap_open=True,
        )
        # BTCUSDT may have fewer or zero orders after overlap filtering
        # ETHUSDT should still be present
        assert "ETHUSDT" in result
