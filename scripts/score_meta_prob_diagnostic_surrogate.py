"""Score feature snapshots with a quarantined meta-probability surrogate.

The companion training script creates a diagnostic-only regression artifact.
This scorer intentionally emits a small, clearly labelled proximity report; it
does not emit a runtime meta-labeler artifact or alter any production model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from neutralgrid.diagnostics.meta_prob_shadow import score_csv_to_shadow_report


def parse_args() -> argparse.Namespace:
    """Return command-line arguments for diagnostic-only scoring."""
    parser = argparse.ArgumentParser(description="Score snapshots with a diagnostic meta_prob surrogate.")
    parser.add_argument("--artifact", required=True, help="Path to diagnostic_meta_prob_surrogate.joblib.")
    parser.add_argument("--input", required=True, help="CSV snapshot input to score.")
    parser.add_argument("--output", required=True, help="CSV output under a diagnostic path.")
    parser.add_argument(
        "--allow-hmm-lineage-mismatch",
        action="store_true",
        help=(
            "Required to score rows inferenced with a different HMM than the "
            "teacher snapshots. Such values are explicitly extrapolative proximity estimates."
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Validate a diagnostic artifact, score finite rows, and write a proximity report."""
    args = parse_args()
    artifact_path = Path(args.artifact).resolve()
    input_path = Path(args.input).resolve()
    summary = score_csv_to_shadow_report(
        artifact_path=artifact_path,
        input_path=input_path,
        output_path=Path(args.output),
        allow_hmm_lineage_mismatch=bool(args.allow_hmm_lineage_mismatch),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
