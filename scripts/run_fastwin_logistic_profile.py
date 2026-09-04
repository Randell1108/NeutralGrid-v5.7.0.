"""Fit the pre-registered development-only FASTWIN logistic profile."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from neutralgrid.scanner.fastwin_logistic_profile import evaluate_development


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = evaluate_development(
        preregistration_path=args.preregistration,
        output_dir=args.output_dir,
    )
    print(json.dumps(report["decision"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
