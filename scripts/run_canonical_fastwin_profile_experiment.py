"""Freeze and development-evaluate the canonical FASTWIN profile experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from neutralgrid.scanner.canonical_fastwin_profile import (
    evaluate_development,
    evaluate_holdout,
    freeze_experiment,
    promote_from_holdout,
)


_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--output-dir", type=Path, required=True)
    freeze.add_argument("--fastwin-dir", type=Path, default=Path("data/fastwin_dataset"))
    freeze.add_argument("--results-dir", type=Path, default=Path("results"))
    freeze.add_argument(
        "--incumbent-model", type=Path, default=Path("data/profile/profile_model.json")
    )
    freeze.add_argument(
        "--incumbent-pattern", type=Path, default=Path("data/profile/pattern_profile.json")
    )
    freeze.add_argument(
        "--incumbent-workbook", type=Path, default=Path("data/new_expired_bots.xlsx")
    )
    evaluate = subparsers.add_parser("evaluate-development")
    evaluate.add_argument("--experiment-dir", type=Path, required=True)
    holdout = subparsers.add_parser("evaluate-holdout")
    holdout.add_argument("--experiment-dir", type=Path, required=True)
    promote = subparsers.add_parser("promote")
    promote.add_argument("--experiment-dir", type=Path, required=True)
    promote.add_argument("--profile-dir", type=Path, default=Path("data/profile"))
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "freeze":
        manifest = freeze_experiment(
            output_dir=args.output_dir,
            fastwin_dir=args.fastwin_dir,
            results_dir=args.results_dir,
            incumbent_model_path=args.incumbent_model,
            incumbent_pattern_path=args.incumbent_pattern,
            incumbent_workbook_path=args.incumbent_workbook,
            code_paths=(
                _ROOT / "src/neutralgrid/scanner/canonical_fastwin_profile.py",
                _ROOT / "src/neutralgrid/scanner/feature_extractor.py",
                _ROOT / "src/neutralgrid/indicators/technical.py",
                _ROOT / "src/neutralgrid/data/funding_rate.py",
            ),
        )
        print(args.output_dir / "manifest.json")
        print(manifest["split"])
        return 0
    if args.command == "evaluate-development":
        report = evaluate_development(args.experiment_dir)
        print(args.experiment_dir / "development_evaluation.json")
        print(report["decision"])
        return 0
    if args.command == "evaluate-holdout":
        report = evaluate_holdout(args.experiment_dir)
        print(args.experiment_dir / "holdout_evaluation.json")
        print(report["decision"])
        return 0
    result = promote_from_holdout(
        args.experiment_dir,
        profile_dir=args.profile_dir,
    )
    print(args.experiment_dir / "promotion_result.json")
    print(result["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
