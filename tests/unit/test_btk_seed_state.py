"""Tests for Phase 3: Replay seed state integration.

Covers:
- SeedOrderLevel frozen dataclass construction and immutability
- SeedState construction with default empty lists
- seed_from_state() snapping logic with 2% tolerance
- seed_from_state() ignoring levels outside tolerance
- load_seed_from_replay() reading from a mock levels.csv
- load_seed_from_replay() returning None for missing/empty data
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

# Ensure the top-level backtest package is importable
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backtest.btk_seed_state import (
    CANDIDATE_TIME_GEOMETRIC_PROFILE,
    CANDIDATE_TIME_GEOMETRY_SOURCE,
    LEGACY_REALISM_PROFILE,
    SeedOrderLevel,
    SeedState,
    build_candidate_time_geometry_seed,
    validate_realism_profile,
)
from backtest.btk_replay_seed_loader import (
    load_seed_from_order_history_exports,
    load_seed_from_replay,
)
from backtest.backtest_realistic import GridConfig, RealisticGridBacktester


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_backtester(lower: float = 100.0, upper: float = 110.0, num_grids: int = 10) -> RealisticGridBacktester:
    """Create a minimal RealisticGridBacktester for seed tests.

    Binance interval semantics use num_grids as the displayed interval count.
    Default 10 intervals over [100, 110] yields integer levels [100..110],
    preserving the level set used by the seed-snapping assertions below.
    """
    cfg = GridConfig(
        symbol="TESTUSDT",
        lower=lower,
        upper=upper,
        num_grids=num_grids,
        mode="arithmetic",
        capital=400.0,
        leverage=10,
    )
    return RealisticGridBacktester(cfg)


# ═══════════════════════════════════════════════════════════════════════════
# SeedOrderLevel
# ═══════════════════════════════════════════════════════════════════════════


class TestSeedOrderLevel:
    """Tests for the frozen SeedOrderLevel dataclass."""

    def test_construction(self):
        level = SeedOrderLevel(side="BUY", price=105.5, qty_total=1.2, order_count=3)
        assert level.side == "BUY"
        assert level.price == 105.5
        assert level.qty_total == 1.2
        assert level.order_count == 3

    def test_default_order_count(self):
        level = SeedOrderLevel(side="SELL", price=106.0, qty_total=0.5)
        assert level.order_count == 1

    def test_frozen_immutability(self):
        level = SeedOrderLevel(side="BUY", price=100.0, qty_total=1.0)
        with pytest.raises(AttributeError):
            level.price = 200.0  # type: ignore[misc]
        with pytest.raises(AttributeError):
            level.side = "SELL"  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════════════
# SeedState
# ═══════════════════════════════════════════════════════════════════════════


class TestSeedState:
    """Tests for the SeedState dataclass."""

    def test_construction_defaults(self):
        t0 = datetime(2026, 2, 20, 12, 0, 0, tzinfo=timezone.utc)
        state = SeedState(t0=t0)
        assert state.t0 == t0
        assert state.open_buy_levels == []
        assert state.open_sell_levels == []
        assert state.open_positions == []
        assert state.symbol is None

    def test_construction_with_levels(self):
        t0 = datetime(2026, 2, 20, 12, 0, 0, tzinfo=timezone.utc)
        buy = SeedOrderLevel(side="BUY", price=100.0, qty_total=1.0)
        sell = SeedOrderLevel(side="SELL", price=110.0, qty_total=0.5)
        state = SeedState(
            t0=t0,
            open_buy_levels=[buy],
            open_sell_levels=[sell],
            symbol="TESTUSDT",
        )
        assert len(state.open_buy_levels) == 1
        assert len(state.open_sell_levels) == 1
        assert state.symbol == "TESTUSDT"

    def test_open_positions_default_empty(self):
        """open_positions defaults to empty list (no fabricated inventory)."""
        state = SeedState(t0=datetime(2026, 1, 1, tzinfo=timezone.utc))
        assert state.open_positions == []

    def test_separate_default_lists(self):
        """Each SeedState instance gets its own default lists (no shared mutable default)."""
        s1 = SeedState(t0=datetime(2026, 1, 1, tzinfo=timezone.utc))
        s2 = SeedState(t0=datetime(2026, 1, 2, tzinfo=timezone.utc))
        s1.open_buy_levels.append(
            SeedOrderLevel(side="BUY", price=100.0, qty_total=1.0)
        )
        assert len(s2.open_buy_levels) == 0


class TestRealismProfileSeed:
    """Tests for explicit realism profile seed helpers."""

    def test_validate_realism_profile_accepts_known_profiles(self):
        assert validate_realism_profile(LEGACY_REALISM_PROFILE) == "legacy"
        assert (
            validate_realism_profile(CANDIDATE_TIME_GEOMETRIC_PROFILE.upper())
            == CANDIDATE_TIME_GEOMETRIC_PROFILE
        )

    def test_validate_realism_profile_rejects_unknown_profile(self):
        with pytest.raises(ValueError):
            validate_realism_profile("silent_auto_tune")

    def test_candidate_time_geometry_seed_classifies_ladder_sides(self):
        t0 = datetime(2026, 2, 20, 12, 0, 0, tzinfo=timezone.utc)
        seed = build_candidate_time_geometry_seed(
            symbol="TESTUSDT",
            grid_levels=[100.0, 101.0, 102.0, 103.0],
            start_price=101.5,
            t0=t0,
        )

        assert seed is not None
        assert seed.source == CANDIDATE_TIME_GEOMETRY_SOURCE
        assert seed.evidence_class == "partial"
        assert seed.qty_per_order is None
        assert seed.qty_source == "model"
        assert [level.price for level in seed.open_buy_levels] == [100.0, 101.0]
        assert [level.price for level in seed.open_sell_levels] == [102.0, 103.0]

    def test_candidate_time_geometry_seed_invalid_price_returns_none(self):
        seed = build_candidate_time_geometry_seed(
            symbol="TESTUSDT",
            grid_levels=[100.0, 101.0],
            start_price=0.0,
            t0=datetime(2026, 2, 20, tzinfo=timezone.utc),
        )
        assert seed is None


# ═══════════════════════════════════════════════════════════════════════════
# seed_from_state()
# ═══════════════════════════════════════════════════════════════════════════


class TestSeedFromState:
    """Tests for RealisticGridBacktester.seed_from_state()."""

    def test_buy_levels_snapped_to_grid(self):
        """Buy levels within 2% of a grid level populate _level_available_bar."""
        bt = _make_backtester(lower=100.0, upper=110.0, num_grids=10)
        # Grid spacing = 1.0, levels = [100, 101, ..., 110]
        # Seed a buy level at 101.005 -- within 2% of 101.0
        seed = SeedState(
            t0=datetime(2026, 2, 20, tzinfo=timezone.utc),
            open_buy_levels=[
                SeedOrderLevel(side="BUY", price=101.005, qty_total=1.0),
                SeedOrderLevel(side="BUY", price=103.0, qty_total=0.8),
            ],
        )
        bt.seed_from_state(seed, bar0_idx=0)

        assert 101.0 in bt._level_available_bar
        assert bt._level_available_bar[101.0] == 0
        assert 103.0 in bt._level_available_bar
        assert bt._level_available_bar[103.0] == 0
        assert bt._is_level_available(101.0, 0) is True
        assert bt._is_level_available(101.0, 0, side="BUY") is True
        assert bt._is_level_available(101.0, 0, side="SELL") is False
        assert bt._is_level_available(102.0, 0) is False

    def test_unseeded_levels_remain_available_by_default(self):
        """Unseeded runs preserve legacy level availability semantics."""
        bt = _make_backtester(lower=100.0, upper=110.0, num_grids=10)
        assert bt._is_level_available(101.0, 0) is True
        assert bt._is_level_available(101.0, 0, side="BUY") is True
        assert bt._is_level_available(101.0, 0, side="SELL") is True

    def test_sell_levels_snapped_to_grid(self):
        """Sell levels within 2% of a grid level populate _level_available_bar."""
        bt = _make_backtester(lower=100.0, upper=110.0, num_grids=10)
        seed = SeedState(
            t0=datetime(2026, 2, 20, tzinfo=timezone.utc),
            open_sell_levels=[
                SeedOrderLevel(side="SELL", price=107.01, qty_total=0.5),
            ],
        )
        bt.seed_from_state(seed, bar0_idx=5)

        assert 107.0 in bt._level_available_bar
        assert bt._level_available_bar[107.0] == 5
        assert bt._is_level_available(107.0, 5, side="SELL") is True
        assert bt._is_level_available(107.0, 5, side="BUY") is False

    def test_levels_outside_tolerance_ignored(self):
        """Seed levels >2% away from any grid level are not added."""
        bt = _make_backtester(lower=100.0, upper=110.0, num_grids=10)
        # Grid levels: 100, 101, ..., 110
        # 100.5 is exactly between 100 and 101; nearest is 100 or 101 (0.5 away)
        # 0.5 / 100.5 = 0.00498 = ~0.5%, within tolerance
        # Use a price far from any grid level: e.g. 95.0 (nearest grid = 100.0, delta = 5%)
        seed = SeedState(
            t0=datetime(2026, 2, 20, tzinfo=timezone.utc),
            open_buy_levels=[
                SeedOrderLevel(side="BUY", price=95.0, qty_total=1.0),
            ],
        )
        bt.seed_from_state(seed, bar0_idx=0)
        assert len(bt._level_available_bar) == 0

    def test_clears_existing_state(self):
        """seed_from_state clears positions and _level_available_bar before seeding."""
        bt = _make_backtester(lower=100.0, upper=110.0, num_grids=10)
        # Pre-populate some state
        bt.positions[101.0] = {"entry_price": 101.0, "qty": 1.0, "entry_bar": 0}
        bt._level_available_bar[102.0] = 5

        seed = SeedState(
            t0=datetime(2026, 2, 20, tzinfo=timezone.utc),
            open_buy_levels=[
                SeedOrderLevel(side="BUY", price=105.0, qty_total=1.0),
            ],
        )
        bt.seed_from_state(seed, bar0_idx=0)

        # Old state should be gone
        assert len(bt.positions) == 0
        assert 102.0 not in bt._level_available_bar
        # Only the new seeded level should be present
        assert 105.0 in bt._level_available_bar

    def test_custom_bar0_idx(self):
        """bar0_idx is correctly used as the availability bar."""
        bt = _make_backtester(lower=100.0, upper=110.0, num_grids=10)
        seed = SeedState(
            t0=datetime(2026, 2, 20, tzinfo=timezone.utc),
            open_buy_levels=[
                SeedOrderLevel(side="BUY", price=104.0, qty_total=1.0),
            ],
        )
        bt.seed_from_state(seed, bar0_idx=42)
        assert bt._level_available_bar[104.0] == 42
        assert bt._is_level_available(104.0, 41) is False
        assert bt._is_level_available(104.0, 42) is True

    def test_seed_qty_override_is_used_as_position_size(self):
        """Verified live order quantity overrides model sizing only in seeded mode."""
        bt = _make_backtester(lower=100.0, upper=110.0, num_grids=10)
        seed = SeedState(
            t0=datetime(2026, 2, 20, tzinfo=timezone.utc),
            open_buy_levels=[
                SeedOrderLevel(side="BUY", price=104.0, qty_total=7.0),
            ],
            qty_per_order=7.0,
            qty_source="order_history",
            evidence_class="partial",
        )
        bt.seed_from_state(seed)
        assert bt._get_position_size() == 7.0
        assert bt._seed_qty_source == "order_history"


# ═══════════════════════════════════════════════════════════════════════════
# load_seed_from_replay()
# ═══════════════════════════════════════════════════════════════════════════


class TestLoadSeedFromReplay:
    """Tests for the replay seed loader."""

    def test_loads_mock_levels_csv(self, tmp_path: Path):
        """Loader reads a temp levels.csv and constructs correct SeedState."""
        symbol = "TESTUSDT"
        symbol_dir = tmp_path / symbol
        symbol_dir.mkdir()

        # Create a mock levels.csv
        t0_ms = 1708430400000  # 2024-02-20 12:00:00 UTC
        data = pd.DataFrame({
            "time_ms": [t0_ms - 1000, t0_ms - 1000, t0_ms + 5000],
            "side": ["BUY", "SELL", "BUY"],
            "price": [100.5, 105.0, 101.0],
            "total_orig_qty": [1.0, 0.5, 2.0],
            "order_count": [1, 2, 1],
        })
        data.to_csv(symbol_dir / "levels.csv", index=False)

        t0 = datetime.fromtimestamp(t0_ms / 1000, tz=timezone.utc)
        result = load_seed_from_replay(tmp_path, symbol, t0)

        assert result is not None
        assert result.symbol == symbol
        # Only the first snapshot (time_ms = t0_ms - 1000) should be included
        # The third row (time_ms = t0_ms + 5000) is after t0
        assert len(result.open_buy_levels) == 1
        assert len(result.open_sell_levels) == 1
        assert result.open_buy_levels[0].price == 100.5
        assert result.open_sell_levels[0].price == 105.0
        assert result.open_sell_levels[0].order_count == 2

    def test_returns_none_for_missing_file(self, tmp_path: Path):
        """Returns None when levels.csv does not exist."""
        result = load_seed_from_replay(tmp_path, "NOSYMBOL", datetime.now(timezone.utc))
        assert result is None

    def test_returns_none_for_empty_csv(self, tmp_path: Path):
        """Returns None when levels.csv is empty."""
        symbol_dir = tmp_path / "EMPTYUSDT"
        symbol_dir.mkdir()
        # Write header only
        pd.DataFrame(columns=["time_ms", "side", "price", "total_orig_qty"]).to_csv(
            symbol_dir / "levels.csv", index=False
        )
        result = load_seed_from_replay(
            tmp_path, "EMPTYUSDT", datetime.now(timezone.utc)
        )
        assert result is None

    def test_returns_none_when_no_data_before_t0(self, tmp_path: Path):
        """Returns None when all rows have time_ms > t0."""
        symbol = "FUTUREUSDT"
        symbol_dir = tmp_path / symbol
        symbol_dir.mkdir()

        t0_ms = 1708430400000
        data = pd.DataFrame({
            "time_ms": [t0_ms + 10000, t0_ms + 20000],
            "side": ["BUY", "SELL"],
            "price": [100.0, 105.0],
            "total_orig_qty": [1.0, 0.5],
            "order_count": [1, 1],
        })
        data.to_csv(symbol_dir / "levels.csv", index=False)

        t0 = datetime.fromtimestamp(t0_ms / 1000, tz=timezone.utc)
        result = load_seed_from_replay(tmp_path, symbol, t0)
        assert result is None

    def test_picks_latest_snapshot_before_t0(self, tmp_path: Path):
        """When multiple snapshots exist before t0, picks the latest one."""
        symbol = "MULTIUSDT"
        symbol_dir = tmp_path / symbol
        symbol_dir.mkdir()

        t0_ms = 1708430400000
        data = pd.DataFrame({
            "time_ms": [
                t0_ms - 60000,  # older snapshot
                t0_ms - 60000,
                t0_ms - 1000,   # latest snapshot (should be picked)
                t0_ms - 1000,
            ],
            "side": ["BUY", "SELL", "BUY", "SELL"],
            "price": [99.0, 111.0, 100.5, 105.5],
            "total_orig_qty": [1.0, 0.5, 2.0, 1.5],
            "order_count": [1, 1, 3, 2],
        })
        data.to_csv(symbol_dir / "levels.csv", index=False)

        t0 = datetime.fromtimestamp(t0_ms / 1000, tz=timezone.utc)
        result = load_seed_from_replay(tmp_path, symbol, t0)

        assert result is not None
        # Should pick the latest snapshot (t0_ms - 1000)
        assert len(result.open_buy_levels) == 1
        assert result.open_buy_levels[0].price == 100.5
        assert result.open_buy_levels[0].order_count == 3
        assert len(result.open_sell_levels) == 1
        assert result.open_sell_levels[0].price == 105.5

    def test_skips_zero_price_rows(self, tmp_path: Path):
        """Rows with price <= 0 are skipped."""
        symbol = "ZEROUSDT"
        symbol_dir = tmp_path / symbol
        symbol_dir.mkdir()

        t0_ms = 1708430400000
        data = pd.DataFrame({
            "time_ms": [t0_ms - 1000, t0_ms - 1000],
            "side": ["BUY", "BUY"],
            "price": [0.0, 100.0],
            "total_orig_qty": [1.0, 1.0],
            "order_count": [1, 1],
        })
        data.to_csv(symbol_dir / "levels.csv", index=False)

        t0 = datetime.fromtimestamp(t0_ms / 1000, tz=timezone.utc)
        result = load_seed_from_replay(tmp_path, symbol, t0)

        assert result is not None
        assert len(result.open_buy_levels) == 1
        assert result.open_buy_levels[0].price == 100.0

    def test_naive_t0_treated_as_utc(self, tmp_path: Path):
        """A timezone-naive t0 is treated as UTC."""
        symbol = "NAIVEUSDT"
        symbol_dir = tmp_path / symbol
        symbol_dir.mkdir()

        t0_ms = 1708430400000
        data = pd.DataFrame({
            "time_ms": [t0_ms - 1000],
            "side": ["BUY"],
            "price": [100.0],
            "total_orig_qty": [1.0],
            "order_count": [1],
        })
        data.to_csv(symbol_dir / "levels.csv", index=False)

        # Pass naive datetime (no tzinfo)
        t0_naive = datetime(2024, 2, 20, 12, 0, 0)
        result = load_seed_from_replay(tmp_path, symbol, t0_naive)
        assert result is not None


class TestLoadSeedFromOrderHistoryExports:
    """Tests for direct Binance Order History seed loading."""

    def test_filters_exact_symbol_strategy_and_active_window(self, tmp_path: Path):
        path = tmp_path / "Order history 8.csv"
        pd.DataFrame(
            [
                {
                    "Time(UTC)": "2026-05-13 22:12:25",
                    "Order No": "1",
                    "Symbol": "DOGEUSDT",
                    "Side": "BUY",
                    "Strategy Id": "411991896",
                    "Price": "0.10924",
                    "Amount": "47",
                    "Executed Amount": "0",
                    "Status": "FILLED",
                    "Update Time": "2026-05-13 23:01:03",
                },
                {
                    "Time(UTC)": "2026-05-13 22:12:25",
                    "Order No": "2",
                    "Symbol": "DOGEUSDT",
                    "Side": "SELL",
                    "Strategy Id": "411991896",
                    "Price": "0.11369",
                    "Amount": "47",
                    "Executed Amount": "0",
                    "Status": "FILLED",
                    "Update Time": "2026-05-13 23:46:01",
                },
                {
                    "Time(UTC)": "2026-05-13 23:46:01",
                    "Order No": "3",
                    "Symbol": "DOGEUSDT",
                    "Side": "SELL",
                    "Strategy Id": "411991896",
                    "Price": "0.11445",
                    "Amount": "47",
                    "Executed Amount": "0",
                    "Status": "NEW",
                    "Update Time": "2026-05-13 23:46:01",
                },
                {
                    "Time(UTC)": "2026-05-13 22:12:25",
                    "Order No": "4",
                    "Symbol": "ADAUSDT",
                    "Side": "BUY",
                    "Strategy Id": "411991896",
                    "Price": "0.50",
                    "Amount": "10",
                    "Executed Amount": "0",
                    "Status": "NEW",
                    "Update Time": "2026-05-13 22:12:25",
                },
            ]
        ).to_csv(path, index=False)

        result = load_seed_from_order_history_exports(
            tmp_path,
            "DOGEUSDT",
            datetime(2026, 5, 13, 22, 12, 25, tzinfo=timezone.utc),
            strategy_id="411991896",
        )

        assert result is not None
        assert result.symbol == "DOGEUSDT"
        assert result.evidence_class == "partial"
        assert result.qty_per_order == 47
        assert result.qty_source == "order_history"
        assert [level.price for level in result.open_buy_levels] == [0.10924]
        assert [level.price for level in result.open_sell_levels] == [0.11369]
        assert result.time_provenance
        first_time = result.time_provenance[0]
        assert first_time["raw_time_text"] == "2026-05-13 22:12:25"
        assert first_time["raw_time_source"] == "Time(UTC)"
        assert first_time["raw_timezone"] == "UTC"
        assert first_time["normalized_time_utc"] == "2026-05-13T22:12:25+00:00"

    def test_returns_none_without_exact_strategy_match(self, tmp_path: Path):
        path = tmp_path / "Order history 8.csv"
        pd.DataFrame(
            [
                {
                    "Time(UTC)": "2026-05-13 22:12:25",
                    "Order No": "1",
                    "Symbol": "DOGEUSDT",
                    "Side": "BUY",
                    "Strategy Id": "OTHER",
                    "Price": "0.10924",
                    "Amount": "47",
                    "Executed Amount": "0",
                    "Status": "NEW",
                    "Update Time": "2026-05-13 22:12:25",
                },
            ]
        ).to_csv(path, index=False)

        result = load_seed_from_order_history_exports(
            tmp_path,
            "DOGEUSDT",
            datetime(2026, 5, 13, 22, 12, 25, tzinfo=timezone.utc),
            strategy_id="411991896",
        )

        assert result is None

    def test_missing_strategy_id_column_cannot_seed_exact_strategy(self, tmp_path: Path):
        path = tmp_path / "Order History 1.csv"
        pd.DataFrame(
            [
                {
                    "Time(UTC)": "2026-05-13 22:12:25",
                    "Order No": "1",
                    "Symbol": "DOGEUSDT",
                    "Side": "BUY",
                    "Price": "0.10924",
                    "Amount": "47",
                    "Executed Amount": "0",
                    "Status": "NEW",
                    "Update Time": "2026-05-13 22:12:25",
                },
            ]
        ).to_csv(path, index=False)

        result = load_seed_from_order_history_exports(
            tmp_path,
            "DOGEUSDT",
            datetime(2026, 5, 13, 22, 12, 25, tzinfo=timezone.utc),
            strategy_id="411991896",
        )

        assert result is None

    def test_conflicting_quantities_do_not_override_position_size(self, tmp_path: Path):
        path = tmp_path / "Order history 8.csv"
        pd.DataFrame(
            [
                {
                    "Time(UTC)": "2026-05-13 22:12:25",
                    "Order No": "1",
                    "Symbol": "DOGEUSDT",
                    "Side": "BUY",
                    "Strategy Id": "411991896",
                    "Price": "0.10924",
                    "Amount": "47",
                    "Executed Amount": "0",
                    "Status": "NEW",
                    "Update Time": "2026-05-13 22:12:25",
                },
                {
                    "Time(UTC)": "2026-05-13 22:12:25",
                    "Order No": "2",
                    "Symbol": "DOGEUSDT",
                    "Side": "SELL",
                    "Strategy Id": "411991896",
                    "Price": "0.11369",
                    "Amount": "48",
                    "Executed Amount": "0",
                    "Status": "NEW",
                    "Update Time": "2026-05-13 22:12:25",
                },
            ]
        ).to_csv(path, index=False)

        result = load_seed_from_order_history_exports(
            tmp_path,
            "DOGEUSDT",
            datetime(2026, 5, 13, 22, 12, 25, tzinfo=timezone.utc),
            strategy_id="411991896",
        )

        assert result is not None
        assert result.evidence_class == "conflicting"
        assert result.qty_per_order is None
        assert result.qty_source == "none"
