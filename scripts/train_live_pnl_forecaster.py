"""Train and bot-disjoint OOS-evaluate a shadow live-PnL forecaster."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neutralgrid.live.decision.pnl_forecast import (
    PnlForecastConfig,
    PnlForecastError,
    train_evaluate_shadow_forecaster,
)
from neutralgrid.live.decision.pnl_history import (
    PnlHistoryError,
    load_all_pnl_observations,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-root", type=Path, default=ROOT / "Live")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--horizon-minutes",
        type=float,
        required=True,
        help="Exact forward PnL horizon; intentionally has no implicit default.",
    )
    parser.add_argument(
        "--label-tolerance-minutes",
        type=float,
        required=True,
        help="Maximum delay beyond the exact horizon accepted for a forward label.",
    )
    parser.add_argument("--fit-fraction", type=float, default=0.60)
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    parser.add_argument("--prediction-interval-coverage", type=float, default=0.80)
    parser.add_argument("--min-fit-bots", type=int, default=4)
    parser.add_argument("--min-calibration-bots", type=int, default=2)
    parser.add_argument("--min-test-bots", type=int, default=2)
    parser.add_argument("--min-fit-samples", type=int, default=30)
    parser.add_argument("--min-calibration-samples", type=int, default=10)
    parser.add_argument("--min-test-samples", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = PnlForecastConfig(
        horizon_minutes=float(args.horizon_minutes),
        label_tolerance_minutes=float(args.label_tolerance_minutes),
        fit_fraction=float(args.fit_fraction),
        calibration_fraction=float(args.calibration_fraction),
        prediction_interval_coverage=float(args.prediction_interval_coverage),
        min_fit_bots=int(args.min_fit_bots),
        min_calibration_bots=int(args.min_calibration_bots),
        min_test_bots=int(args.min_test_bots),
        min_fit_samples=int(args.min_fit_samples),
        min_calibration_samples=int(args.min_calibration_samples),
        min_test_samples=int(args.min_test_samples),
    )
    try:
        observations = load_all_pnl_observations(live_root=args.live_root)
        report = train_evaluate_shadow_forecaster(
            observations,
            config=config,
            output_dir=args.output_dir,
        )
    except (PnlHistoryError, PnlForecastError, OSError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report.get("forecast_eligible") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
