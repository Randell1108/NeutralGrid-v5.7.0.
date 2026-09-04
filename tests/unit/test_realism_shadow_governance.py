from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd
import pytest

import backtest_candidates as backtest_cli
import optimize_thresholds_v20260311 as threshold_optimizer
from neutralgrid.backtest.realism_governance import (
    CANDIDATE_TIME_GEOMETRIC_PROFILE,
    CANDIDATE_TIME_PUBLIC_MARKET_PROFILE,
    LEGACY_REALISM_PROFILE,
    validate_realism_output_path,
)
from neutralgrid.scanner import canonical_fastwin_profile as canonical_profile
from neutralgrid.scanner import fastwin_logistic_profile as logistic_profile
from neutralgrid.scanner.empirical_profile_v20260302 import (
    collect_authoritative_backtest_rows,
)
from scripts import audit_canonical_fastwin_profile as profile_audit
from scripts import backtest_matched_candidate_comparison as matched_comparison
from scripts import build_depth_shadow_outcomes as depth_shadow
from scripts import build_fastwin_holdout as fastwin_holdout
from scripts import generate_fastwin_dataset as fastwin_generator
from scripts.validate_backtest_live_reconciliation import run_validation


@pytest.mark.parametrize(
    "profile",
    [
        CANDIDATE_TIME_GEOMETRIC_PROFILE,
        CANDIDATE_TIME_PUBLIC_MARKET_PROFILE,
    ],
)
@pytest.mark.parametrize(
    "relative_output",
    [
        "data/backtest_candidates",
        "data/backtest_candidates/shadow/run.csv",
        "data/fastwin_dataset",
        "data/fastwin_dataset/shadow",
    ],
)
def test_shadow_output_guard_rejects_canonical_roots_and_descendants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
    relative_output: str,
) -> None:
    base_dir = tmp_path / "checkout"
    work_dir = base_dir / "tests"
    work_dir.mkdir(parents=True)
    monkeypatch.chdir(work_dir)
    output = Path("..") / Path(relative_output)

    with pytest.raises(ValueError, match="shadow-only"):
        validate_realism_output_path(profile, output, base_dir=base_dir)


def test_legacy_output_guard_preserves_canonical_authority(tmp_path: Path) -> None:
    canonical = tmp_path / "data" / "backtest_candidates"

    resolved = validate_realism_output_path(
        LEGACY_REALISM_PROFILE,
        canonical,
        base_dir=tmp_path,
    )

    assert resolved == canonical.resolve()


def test_unknown_profile_fails_before_path_admission(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported realism_profile"):
        validate_realism_output_path(
            "unregistered_profile",
            tmp_path / "outputs" / "audit",
            base_dir=tmp_path,
        )


def test_primary_cli_rejects_shadow_fastwin_pool(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "backtest_candidates.py",
            "--realism-profile",
            CANDIDATE_TIME_PUBLIC_MARKET_PROFILE,
            "--output",
            "data/fastwin_dataset/shadow",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        backtest_cli.parse_args()

    assert exc_info.value.code == 2
    assert "shadow-only" in capsys.readouterr().err


def test_fastwin_generator_defaults_to_isolated_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["generate_fastwin_dataset.py"])

    args = fastwin_generator.parse_args()

    assert args.output == "outputs/audits/fastwin_shadow"


def test_fastwin_generator_rejects_authoritative_pool(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_fastwin_dataset.py",
            "--output",
            "data/fastwin_dataset",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        fastwin_generator.parse_args()

    assert exc_info.value.code == 2
    assert "shadow-only" in capsys.readouterr().err


def test_matched_comparison_defaults_to_isolated_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["backtest_matched_candidate_comparison.py"])

    args = matched_comparison.parse_args()

    assert args.output.startswith("outputs/audits/")


def test_matched_comparison_rejects_canonical_descendant(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "backtest_matched_candidate_comparison.py",
            "--output",
            "data/backtest_candidates/matched.csv",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        matched_comparison.parse_args()

    assert exc_info.value.code == 2
    assert "shadow-only" in capsys.readouterr().err


def test_matched_comparison_contains_invalid_row_and_close_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = pd.DataFrame(
        [
            {
                "strategy_id": "413491078",
                "candidate_id": "ADAUSDT_20260801_000000_fixture",
                "observed_symbol": "ADAUSDT",
                "observed_start_time_utc": "not-a-timestamp",
                "observed_duration_hours": 6.0,
                "observed_margin_usdt": 400.0,
                "observed_leverage": 10,
                "observed_total_profit_usdt": 8.0,
                "observed_pnl_pct": 2.0,
            }
        ]
    )
    candidates = pd.DataFrame(
        [
            {
                "candidate_id": "ADAUSDT_20260801_000000_fixture",
                "symbol": "ADAUSDT",
                "candidate_source_file": "deployment_ready_fixture.csv",
            }
        ]
    )

    class FailingCloseClient:
        closed = False

        async def close(self) -> None:
            self.closed = True
            raise OSError("close fixture")

    client = FailingCloseClient()
    monkeypatch.setattr(matched_comparison, "load_observed_rows", lambda *_: observed)
    monkeypatch.setattr(
        matched_comparison,
        "load_exact_candidate_rows",
        lambda *_: candidates,
    )
    monkeypatch.setattr(matched_comparison, "BinanceClient", lambda: client)
    args = argparse.Namespace(
        expired_bots_path="unused.xlsx",
        intake_manifest="unused.csv",
        results_dir="unused",
        delay=0.0,
    )

    result = asyncio.run(matched_comparison.run_comparison(args))

    assert len(result) == 1
    assert result.iloc[0]["backtest_status"] == "skipped"
    assert result.iloc[0]["backtest_error_type"] == "ValueError"
    assert result.iloc[0]["observed_start_time_utc"] == "not-a-timestamp"
    assert client.closed is True


def test_fastwin_holdout_rejects_canonical_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_fastwin_holdout.py",
            "freeze",
            "--output-dir",
            "data/backtest_candidates/holdout",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        fastwin_holdout.parse_args()

    assert exc_info.value.code == 2
    assert "shadow-only" in capsys.readouterr().err


def test_fastwin_holdout_function_guard_precedes_input_reads() -> None:
    with pytest.raises(ValueError, match="shadow-only"):
        fastwin_holdout.freeze_holdout(
            output_dir=Path("data/fastwin_dataset/holdout"),
            results_dir=Path("unused-results"),
            linkage_path=Path("unused-linkage.csv"),
            fastwin_dir=Path("unused-fastwin"),
            evaluated_dir=Path("unused-evaluated"),
            as_of=datetime.now(timezone.utc),
            min_rows=150,
            min_scan_groups=10,
        )


def test_depth_shadow_rejects_canonical_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        depth_shadow.parse_args(
            [
                "--input",
                "unused.csv",
                "--output-dir",
                "data/fastwin_dataset/depth",
            ]
        )

    assert exc_info.value.code == 2
    assert "shadow-only" in capsys.readouterr().err


def test_depth_shadow_function_guard_precedes_input_read() -> None:
    args = argparse.Namespace(
        input="unused.csv",
        output_dir="data/backtest_candidates/depth",
        realism_profile=CANDIDATE_TIME_GEOMETRIC_PROFILE,
    )

    with pytest.raises(ValueError, match="shadow-only"):
        asyncio.run(depth_shadow.build_outcomes(args))


def test_validator_rejects_shadow_output_before_loading_inputs() -> None:
    args = argparse.Namespace(
        realism_profile=CANDIDATE_TIME_GEOMETRIC_PROFILE,
        output_dir="data/backtest_candidates/validator",
    )

    with pytest.raises(ValueError, match="shadow-only"):
        run_validation(args)


def test_canonical_profile_freeze_guard_precedes_source_reads() -> None:
    with pytest.raises(ValueError, match="shadow-only"):
        canonical_profile.freeze_experiment(
            output_dir=Path("data/fastwin_dataset/profile_experiment"),
            fastwin_dir=Path("unused-fastwin"),
            results_dir=Path("unused-results"),
            incumbent_model_path=Path("unused-model.json"),
            incumbent_pattern_path=Path("unused-pattern.json"),
            incumbent_workbook_path=Path("unused-workbook.xlsx"),
        )


@pytest.mark.parametrize(
    "operation",
    [
        lambda path: canonical_profile.evaluate_development(path),
        lambda path: canonical_profile.evaluate_holdout(path),
        lambda path: canonical_profile.promote_from_holdout(
            path,
            profile_dir=Path("unused-profile"),
        ),
    ],
)
def test_canonical_profile_lifecycle_rejects_canonical_experiment_tree(
    operation: Callable[[Path], object],
) -> None:
    with pytest.raises(ValueError, match="shadow-only"):
        operation(Path("data/backtest_candidates/profile_experiment"))


def test_logistic_profile_guard_precedes_preregistration_read() -> None:
    with pytest.raises(ValueError, match="shadow-only"):
        logistic_profile.evaluate_development(
            preregistration_path=Path("unused-preregistration.json"),
            output_dir=Path("data/fastwin_dataset/logistic"),
        )


def test_profile_audit_atomic_writer_rejects_canonical_tree() -> None:
    output = Path("data/backtest_candidates/profile_audit.json")

    with pytest.raises(ValueError, match="shadow-only"):
        profile_audit._atomic_write_json(output, {"status": "shadow"})

    assert not output.exists()


def test_empirical_profile_filters_shadow_before_candidate_dedup(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    backtest_dir = tmp_path / "backtest_candidates"
    backtest_dir.mkdir()
    pd.DataFrame(
        [
            {
                "candidate_id": "historical_only",
                "net_pnl_pct": 1.0,
            }
        ]
    ).to_csv(backtest_dir / "backtest_results_01_historical.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate_id": "legacy_preserved",
                "net_pnl_pct": 2.0,
                "realism_profile": LEGACY_REALISM_PROFILE,
            }
        ]
    ).to_csv(backtest_dir / "backtest_results_02_legacy.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate_id": "legacy_preserved",
                "net_pnl_pct": 99.0,
                "realism_profile": CANDIDATE_TIME_GEOMETRIC_PROFILE,
            },
            {
                "candidate_id": "geometric_shadow",
                "net_pnl_pct": 99.0,
                "realism_profile": CANDIDATE_TIME_GEOMETRIC_PROFILE,
            },
            {
                "candidate_id": "public_shadow",
                "net_pnl_pct": 99.0,
                "realism_profile": CANDIDATE_TIME_PUBLIC_MARKET_PROFILE,
            },
            {
                "candidate_id": "blank_profile",
                "net_pnl_pct": 99.0,
                "realism_profile": "",
            },
            {
                "candidate_id": "unknown_profile",
                "net_pnl_pct": 99.0,
                "realism_profile": "unregistered_profile",
            },
        ]
    ).to_csv(backtest_dir / "backtest_results_03_shadow.csv", index=False)

    with caplog.at_level("INFO"):
        admitted = collect_authoritative_backtest_rows(backtest_dir)

    assert set(admitted["candidate_id"]) == {
        "historical_only",
        "legacy_preserved",
    }
    preserved = admitted.loc[admitted["candidate_id"] == "legacy_preserved"]
    assert float(preserved.iloc[0]["net_pnl_pct"]) == 2.0
    assert (
        "historical_unstamped=1, legacy=1, excluded_shadow=3, "
        "excluded_blank=1, excluded_unknown=1"
    ) in caplog.text

    optimized_pool = threshold_optimizer.load_backtest_results(backtest_dir)
    assert optimized_pool["candidate_id"].tolist() == admitted["candidate_id"].tolist()
