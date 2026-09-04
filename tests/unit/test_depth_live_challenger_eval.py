from __future__ import annotations

import pandas as pd
import pytest

from scripts.evaluate_depth_live_challenger import _metrics_at_top_fraction


def test_metrics_at_top_fraction_scores_selected_candidate() -> None:
    frame = pd.DataFrame(
        [
            {
                "candidate_id": "winner",
                "depth_oracle_label": 1,
                "depth_adjusted_pnl_pct": 8.0,
                "tail_pnl_pct": -2.0,
                "score": 0.9,
            },
            {
                "candidate_id": "loser",
                "depth_oracle_label": 0,
                "depth_adjusted_pnl_pct": 2.0,
                "tail_pnl_pct": -3.0,
                "score": 0.1,
            },
        ]
    )

    metrics = _metrics_at_top_fraction(frame, "score", fraction=0.25)

    assert metrics["k"] == 1
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["median_pnl"] == pytest.approx(8.0)
    assert metrics["false_positive_rate"] == pytest.approx(0.0)
    assert metrics["selected_candidate_ids"] == ["winner"]
