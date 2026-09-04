"""ERR-062 regression: the Phase-2 beta calibration upgrade must never deploy
a calibrator that is worse than the gate-accepted sigmoid/isotonic one.

The historical defect (artifact 20260701_224916): the reliability diagnostic
ran on the RAW model output, so any badly-calibrated raw model triggered
``needs_beta_upgrade`` even when the accepted sigmoid had already fixed the
miscalibration, and beta replaced the sigmoid on convergence alone — deploying
ECE 0.0652 over the sigmoid's 0.0181 on the same pooled OOS predictions.

The fix: the diagnostic runs on the ACCEPTED calibrator's output (residual
miscalibration), and beta is deployed only when its OOS ECE strictly beats the
accepted calibrator's. These tests train end-to-end on synthetic data with a
deliberately overconfident raw model and assert the invariant on whichever
branch executes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neutralgrid.models.meta_labeler import MetaLabeler, MetaLabelerConfig


def _synthetic_training_df(n: int = 600, seed: int = 7) -> pd.DataFrame:
    """Strong-signal features + label noise → overconfident raw model whose
    raw OOS probabilities are miscalibrated (the ERR-062 trigger condition).

    Symbols occupy CONTIGUOUS time blocks (15 rows each) so the symbol-blocked
    CPCV groups are time-localized and the purge does not erase every train
    fold (interleaved symbols would degenerate CPCV to 0 valid folds)."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    p_true = 1.0 / (1.0 + np.exp(-3.0 * x1))
    # 25% label noise keeps true frequencies away from the model's confident 0/1
    y = np.where(rng.uniform(size=n) < 0.75, (p_true > 0.5).astype(int),
                 rng.integers(0, 2, size=n))
    ts = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "f1": x1,
            "f2": x2,
            "f3": x3,
            "start_time_utc": ts,
            "t1": ts + pd.Timedelta(hours=4),
            # pnl encodes the label against the default 3.0 hurdle
            "pnl_pct": np.where(y == 1, 5.0, -2.0),
            "symbol": [f"SYM{i // 15}USDT" for i in range(n)],
        }
    )


@pytest.fixture(scope="module")
def trained_labeler() -> MetaLabeler:
    config = MetaLabelerConfig(features=["f1", "f2", "f3"])
    labeler = MetaLabeler(config)
    labeler.train(_synthetic_training_df())
    return labeler


class TestBetaCalibrationGate:
    def test_calibration_phase_ran(self, trained_labeler: MetaLabeler):
        report = trained_labeler._oos_calibration_report
        assert report, "OOS calibration phase did not run on the synthetic pool"
        assert "ece_calibrated" in report

    def test_deployed_beta_must_beat_displaced_calibrator(
        self, trained_labeler: MetaLabeler
    ):
        """The core ERR-062 invariant: if beta_oos is deployed, its ECE must be
        the one reported in ece_calibrated AND strictly better than the
        calibrator it displaced. Under the pre-fix code the deployed beta kept
        the sigmoid's ECE in the report and never recorded displacement, so
        this branch fails there."""
        report = trained_labeler._oos_calibration_report
        method = trained_labeler._oos_calibration_method
        if method == "beta_oos":
            assert report.get("beta_upgrade") is True
            assert "displaced_ece_calibrated" in report, (
                "beta deployed without recording the displaced calibrator"
            )
            assert report["ece_calibrated"] == pytest.approx(
                report["beta_ece_after"]
            ), "ece_calibrated must describe the DEPLOYED calibrator"
            assert (
                report["ece_calibrated"] < report["displaced_ece_calibrated"]
            ), "beta was deployed without beating the accepted calibrator's ECE"

    def test_rejected_beta_candidate_was_not_better(
        self, trained_labeler: MetaLabeler
    ):
        report = trained_labeler._oos_calibration_report
        if report.get("beta_upgrade_reason") == "ece_not_better":
            assert trained_labeler._oos_calibration_method != "beta_oos"
            assert (
                report["beta_ece_candidate"] >= report["ece_calibrated"]
            )

    def test_deployed_calibrator_ece_is_minimal_among_candidates(
        self, trained_labeler: MetaLabeler
    ):
        """Whichever branch executed, the deployed ECE must not exceed any
        recorded candidate/displaced ECE."""
        report = trained_labeler._oos_calibration_report
        deployed = float(report["ece_calibrated"])
        for key in ("displaced_ece_calibrated", "beta_ece_candidate"):
            if key in report:
                assert deployed <= float(report[key]) + 1e-12
