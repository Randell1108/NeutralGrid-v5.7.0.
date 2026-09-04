from __future__ import annotations

import importlib.util
import shutil
import uuid
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from neutralgrid.backtest.candidate_pipeline import (
    attach_mark_close,
    extract_exchange_filter_settings,
    filter_backtest_candidates,
    funding_rate_series_from_rows,
    load_all_scanner_csvs,
    run_single_backtest,
)
from neutralgrid.models.meta_labeler import ACTIVE_SNAPSHOT_META_FEATURES


_CLI_PATH = Path(__file__).resolve().parents[2] / "backtest_candidates.py"
_CLI_SPEC = importlib.util.spec_from_file_location("backtest_candidates", _CLI_PATH)
assert _CLI_SPEC is not None and _CLI_SPEC.loader is not None
backtest_cli = importlib.util.module_from_spec(_CLI_SPEC)
_CLI_SPEC.loader.exec_module(backtest_cli)


def _make_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": "STDUSDT",
                "score": 60.0,
                "grid_is_valid": True,
                "grid_lower": 95.0,
                "grid_upper": 105.0,
                "num_grids": 10,
                "micro_osc_score": 0.20,
                "survival_prob": 0.80,
                "candidate_id": "std_keep",
            },
            {
                "symbol": "BYPASSUSDT",
                "score": 60.0,
                "grid_is_valid": False,
                "grid_lower": 95.0,
                "grid_upper": 105.0,
                "num_grids": 10,
                "micro_osc_score": 0.55,
                "survival_prob": 0.70,
                "candidate_id": "bypass_keep",
            },
            {
                "symbol": "NOPARAMSUSDT",
                "score": 60.0,
                "grid_is_valid": False,
                "grid_lower": None,
                "grid_upper": None,
                "num_grids": None,
                "micro_osc_score": 0.55,
                "survival_prob": 0.70,
                "candidate_id": "bypass_drop",
            },
            {
                "symbol": "PLAININVALIDUSDT",
                "score": 60.0,
                "grid_is_valid": False,
                "grid_lower": 90.0,
                "grid_upper": 110.0,
                "num_grids": 12,
                "micro_osc_score": 0.30,
                "survival_prob": 0.80,
                "candidate_id": "plain_invalid_drop",
            },
            {
                "symbol": "LOWSCOREUSDT",
                "score": 40.0,
                "grid_is_valid": False,
                "grid_lower": 90.0,
                "grid_upper": 110.0,
                "num_grids": 12,
                "micro_osc_score": 0.55,
                "survival_prob": 0.80,
                "candidate_id": "low_drop",
            },
        ]
    )


def test_standard_mode_keeps_only_grid_valid_rows() -> None:
    df = _make_candidates()

    with patch(
        "neutralgrid.backtest.candidate_pipeline._load_deployed_candidate_ids",
        return_value=(set(), set()),
    ), patch(
        "neutralgrid.backtest.candidate_pipeline._meta_labeler_needs_bootstrap",
        return_value=False,
    ):
        result = filter_backtest_candidates(
            df,
            linkage_path=Path("ignored.csv"),
            min_score=45.0,
        )

    assert result["symbol"].tolist() == ["STDUSDT"]


def test_standard_mode_keeps_discovery_grid_valid_rows_below_score() -> None:
    df = pd.DataFrame(
        [
            {
                "symbol": "DISCOVERYUSDT",
                "score": 12.0,
                "grid_is_valid": True,
                "discovery_mode": True,
                "below_threshold_tag": True,
                "grid_lower": 95.0,
                "grid_upper": 105.0,
                "num_grids": 10,
                "candidate_id": "discovery_keep",
            },
            {
                "symbol": "LOWSTANDARDUSDT",
                "score": 12.0,
                "grid_is_valid": True,
                "discovery_mode": False,
                "below_threshold_tag": True,
                "grid_lower": 95.0,
                "grid_upper": 105.0,
                "num_grids": 10,
                "candidate_id": "standard_drop",
            },
        ]
    )

    with patch(
        "neutralgrid.backtest.candidate_pipeline._load_deployed_candidate_ids",
        return_value=(set(), set()),
    ), patch(
        "neutralgrid.backtest.candidate_pipeline._meta_labeler_needs_bootstrap",
        return_value=False,
    ):
        result = filter_backtest_candidates(
            df,
            linkage_path=Path("ignored.csv"),
            min_score=20.0,
        )

    assert result["symbol"].tolist() == ["DISCOVERYUSDT"]


def test_bypass_only_keeps_bypass_rows_with_grid_params() -> None:
    df = _make_candidates()

    with patch(
        "neutralgrid.backtest.candidate_pipeline._load_deployed_candidate_ids",
        return_value=(set(), set()),
    ):
        result = filter_backtest_candidates(
            df,
            linkage_path=Path("ignored.csv"),
            min_score=45.0,
            bypass_only=True,
        )

    assert result["symbol"].tolist() == ["BYPASSUSDT"]
    assert "NOPARAMSUSDT" not in set(result["symbol"])
    assert "PLAININVALIDUSDT" not in set(result["symbol"])
    assert "STDUSDT" not in set(result["symbol"])


def test_bypass_only_excludes_deployed_candidates() -> None:
    df = _make_candidates().loc[lambda frame: frame["symbol"].isin(["STDUSDT", "BYPASSUSDT"])].copy()

    with patch(
        "neutralgrid.backtest.candidate_pipeline._load_deployed_candidate_ids",
        return_value=({"bypass_keep"}, set()),
    ):
        result = filter_backtest_candidates(
            df,
            linkage_path=Path("ignored.csv"),
            min_score=45.0,
            bypass_only=True,
        )

    assert result.empty


def test_bypass_only_honors_exact_thresholds() -> None:
    df = pd.DataFrame(
        [
            {
                "symbol": "EXACTPASSUSDT",
                "score": 45.0,
                "grid_is_valid": False,
                "grid_lower": 95.0,
                "grid_upper": 105.0,
                "num_grids": 10,
                "micro_osc_score": 0.45,
                "survival_prob": 0.60,
                "candidate_id": "exact_pass",
            },
            {
                "symbol": "JUSTBELOWUSDT",
                "score": 45.0,
                "grid_is_valid": False,
                "grid_lower": 95.0,
                "grid_upper": 105.0,
                "num_grids": 10,
                "micro_osc_score": 0.4499,
                "survival_prob": 0.5999,
                "candidate_id": "just_below",
            },
        ]
    )

    with patch(
        "neutralgrid.backtest.candidate_pipeline._load_deployed_candidate_ids",
        return_value=(set(), set()),
    ):
        result = filter_backtest_candidates(
            df,
            linkage_path=Path("ignored.csv"),
            min_score=45.0,
            bypass_only=True,
        )

    assert result["symbol"].tolist() == ["EXACTPASSUSDT"]


def test_parse_args_accepts_bypass_only_flag(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["backtest_candidates.py", "--bypass-only"],
    )

    args = backtest_cli.parse_args()

    assert args.bypass_only is True


def test_authoritative_pool_loader_reads_csv_and_applies_full_date_window(
    tmp_path: Path,
) -> None:
    pool_path = tmp_path / "authoritative.csv"
    pd.DataFrame(
        {
            "candidate_id": [
                "BTCUSDT_20260608_000000_deadbeef",
                "币安人生USDT_20260817_235959_e6e12c06",
                "ETHUSDT_20260818_000000_cafebabe",
            ]
        }
    ).to_csv(pool_path, index=False)

    ids = backtest_cli._load_authoritative_candidate_ids(
        pool_path,
        datetime.fromisoformat("2026-06-08T00:00:00+00:00"),
        datetime.fromisoformat("2026-08-17T23:59:59.999999+00:00"),
    )

    assert ids == {
        "BTCUSDT_20260608_000000_deadbeef",
        "币安人生USDT_20260817_235959_e6e12c06",
    }


def test_authoritative_pool_loader_selects_general_sheet_from_canonical_workbook(
    tmp_path: Path,
) -> None:
    pool_path = tmp_path / "authoritative.xlsx"
    with pd.ExcelWriter(pool_path, engine="openpyxl") as writer:
        pd.DataFrame(
            {"candidate_id": ["BTCUSDT_20260608_000000_deadbeef"]}
        ).to_excel(writer, sheet_name="General", index=False)
        pd.DataFrame({"candidate_id": ["wrong_sheet"]}).to_excel(
            writer, sheet_name="Meta Features", index=False
        )

    ids = backtest_cli._load_authoritative_candidate_ids(
        pool_path,
        datetime.fromisoformat("2026-06-08T00:00:00+00:00"),
        datetime.fromisoformat("2026-08-17T23:59:59.999999+00:00"),
    )

    assert ids == {"BTCUSDT_20260608_000000_deadbeef"}


def test_parse_args_defaults_realism_profile_to_legacy(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["backtest_candidates.py"])

    args = backtest_cli.parse_args()

    assert args.realism_profile == "legacy"


def test_parse_args_accepts_candidate_time_realism_profile(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "backtest_candidates.py",
            "--realism-profile",
            "candidate_time_geometric_v1",
            "--output",
            "outputs/audits/candidate_time_shadow",
        ],
    )

    args = backtest_cli.parse_args()

    assert args.realism_profile == "candidate_time_geometric_v1"


def test_parse_args_accepts_public_market_realism_profile(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "backtest_candidates.py",
            "--realism-profile",
            "candidate_time_public_market_v1",
            "--output",
            "outputs/audits/public_market_shadow",
        ],
    )

    args = backtest_cli.parse_args()

    assert args.realism_profile == "candidate_time_public_market_v1"


@pytest.mark.parametrize(
    "profile",
    ["candidate_time_geometric_v1", "candidate_time_public_market_v1"],
)
def test_parse_args_rejects_shadow_profile_in_canonical_output(
    monkeypatch,
    capsys,
    profile: str,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["backtest_candidates.py", "--realism-profile", profile],
    )

    with pytest.raises(SystemExit) as exc_info:
        backtest_cli.parse_args()

    assert exc_info.value.code == 2
    assert "shadow-only" in capsys.readouterr().err


def test_parse_args_rejects_relative_canonical_output_from_subdirectory(
    monkeypatch,
    capsys,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(repo_root / "tests")
    monkeypatch.setattr(
        "sys.argv",
        [
            "backtest_candidates.py",
            "--realism-profile",
            "candidate_time_geometric_v1",
            "--output",
            "../data/backtest_candidates",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        backtest_cli.parse_args()

    assert exc_info.value.code == 2
    assert "shadow-only" in capsys.readouterr().err


def test_parse_args_rejects_shadow_output_below_canonical_directory(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "backtest_candidates.py",
            "--realism-profile",
            "candidate_time_public_market_v1",
            "--output",
            "data/backtest_candidates/shadow",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        backtest_cli.parse_args()

    assert exc_info.value.code == 2
    assert "shadow-only" in capsys.readouterr().err


def test_exchange_filter_extraction_uses_filter_values_not_precision() -> None:
    exchange_info = {
        "symbols": [
            {
                "symbol": "ABCUSDT",
                "pricePrecision": 2,
                "quantityPrecision": 3,
                "filters": [
                    {"filterType": "PRICE_FILTER", "tickSize": "0.0005"},
                    {"filterType": "LOT_SIZE", "stepSize": "0.25"},
                    {"filterType": "MIN_NOTIONAL", "notional": "5"},
                ],
            }
        ]
    }

    settings = extract_exchange_filter_settings(exchange_info, "ABCUSDT")

    assert settings["exchange_filter_validation_status"] == "valid"
    assert settings["tick_size"] == pytest.approx(0.0005)
    assert settings["step_size"] == pytest.approx(0.25)
    assert settings["min_notional"] == pytest.approx(5.0)


def test_public_market_profile_fails_closed_without_exchange_filters() -> None:
    start = datetime(2026, 1, 1)
    klines = pd.DataFrame(
        {
            "timestamp": [start + timedelta(minutes=i) for i in range(4)],
            "open": [100.0, 101.0, 99.0, 100.5],
            "high": [101.0, 102.0, 100.0, 101.0],
            "low": [99.0, 100.0, 98.0, 100.0],
            "close": [100.0, 101.0, 99.0, 100.5],
            "volume": [1.0, 1.0, 1.0, 1.0],
        }
    )
    candidate = {
        "symbol": "ABCUSDT",
        "candidate_id": "ABCUSDT_20260101_000000",
        "grid_lower": 95.0,
        "grid_upper": 105.0,
        "num_grids": 5,
        "mode": "geometric",
    }

    with pytest.raises(ValueError, match="exchange filters invalid"):
        run_single_backtest(
            candidate_row=candidate,
            klines_df=klines,
            realism_profile="candidate_time_public_market_v1",
        )


def test_mark_and_funding_helpers_classify_public_evidence() -> None:
    ts = pd.Timestamp("2026-01-01T00:00:00Z")
    last = pd.DataFrame({"timestamp": [ts], "close": [100.0]})
    mark = pd.DataFrame({"timestamp": [ts], "close": [100.5]})

    merged = attach_mark_close(last, mark)
    funding_series, funding_status, funding_rows = funding_rate_series_from_rows([])

    assert merged["mark_close"].tolist() == [100.5]
    assert funding_series is None
    assert funding_status == "no_event_in_window"
    assert funding_rows == 0


def test_load_all_scanner_csvs_avoids_fragmentation_warning(monkeypatch) -> None:
    tmp_path = Path.cwd() / ".tmp_test_load_all_scanner_csvs" / str(uuid.uuid4())
    tmp_path.mkdir(parents=True, exist_ok=True)
    try:
        csv_path = tmp_path / "deployment_ready_20260331_203529.csv"
        csv_path.write_text("symbol\nBTCUSDT\n", encoding="utf-8")

        fragmented = pd.DataFrame({"symbol": ["BTCUSDT"]})
        for col in ACTIVE_SNAPSHOT_META_FEATURES:
            if col == "grid_spacing_pct":
                continue
            fragmented[col] = [0.0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
            for idx in range(130):
                fragmented.insert(len(fragmented.columns), f"feature_{idx}", [float(idx)])

        def _fake_read_csv(_: Path) -> pd.DataFrame:
            return fragmented.copy()

        monkeypatch.setattr(
            "neutralgrid.backtest.candidate_pipeline.pd.read_csv",
            _fake_read_csv,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", pd.errors.PerformanceWarning)
            result = load_all_scanner_csvs(tmp_path)

        assert result["scan_file"].tolist() == [csv_path.name]
        assert result["candidate_id"].tolist() == ["BTCUSDT_20260331_203529"]
    finally:
        shutil.rmtree(tmp_path.parent, ignore_errors=True)


def test_load_all_scanner_csvs_uses_active_profile_gate_with_alias_support(monkeypatch) -> None:
    tmp_path = Path.cwd() / ".tmp_test_load_all_scanner_csvs_active_profile" / str(uuid.uuid4())
    tmp_path.mkdir(parents=True, exist_ok=True)
    try:
        keep_path = tmp_path / "deployment_ready_20260331_203529.csv"
        drop_path = tmp_path / "deployment_ready_20260331_203630.csv"
        keep_path.write_text("symbol\nBTCUSDT\n", encoding="utf-8")
        drop_path.write_text("symbol\nETHUSDT\n", encoding="utf-8")

        # FASTWIN-01: the active profile gate now requires the 20-feature
        # fast-winner profile (minus grid_spacing_pct, the one derivable
        # exception). keep_df supplies the full required set, exercising alias
        # support via scan_adx_15m -> adx_15m. drop_df omits adx_1h (which has
        # no alias) so it fails the completeness gate and is dropped.
        keep_df = pd.DataFrame(
            [
                {
                    "symbol": "BTCUSDT",
                    "ev_score": 0.5,
                    "adx_1h": 18.0,
                    "scan_adx_15m": 20.0,  # alias -> adx_15m
                    "adx_5m": 22.0,
                    "atr_pct_15m": 0.01,
                    "ou_halflife": 12.0,
                    "rsi_15m": 50.0,
                    "ema_slope_1h": 0.0,
                    "ema_crosses_5m": 1.0,
                    "vwap_crosses_5m": 1.0,
                    "hurst_exponent": 0.5,
                    "range_size_pct": 5.0,
                    "bb_width": 0.03,
                    "profit_per_grid_pct": 0.45,
                    "num_grids": 20,
                    "quote_volume_24h": 5.0e7,
                    "open_interest": 5.0e7,
                    "micro_round_trip_cost_pct": 0.15,
                    "funding_rate": 0.0001,
                }
            ]
        )
        drop_df = keep_df.drop(columns=["adx_1h"]).copy()
        drop_df["symbol"] = "ETHUSDT"

        def _fake_read_csv(path: Path) -> pd.DataFrame:
            if path == keep_path:
                return keep_df.copy()
            if path == drop_path:
                return drop_df.copy()
            raise AssertionError(f"Unexpected path: {path}")

        monkeypatch.setattr(
            "neutralgrid.backtest.candidate_pipeline.pd.read_csv",
            _fake_read_csv,
        )

        result = load_all_scanner_csvs(tmp_path)

        assert result["symbol"].tolist() == ["BTCUSDT"]
        assert result["candidate_id"].tolist() == ["BTCUSDT_20260331_203529"]
    finally:
        shutil.rmtree(tmp_path.parent, ignore_errors=True)
