from __future__ import annotations

import pandas as pd

from neutralgrid.models.meta_labeler import ACTIVE_SNAPSHOT_META_FEATURES
from neutralgrid.training import live_outcome_ingestor
from neutralgrid.training.unified_training_builder import UnifiedTrainingBuilder


def _live_row(candidate_id: str, match_method: str) -> dict[str, object]:
    row: dict[str, object] = {
        feature: float(index + 1)
        for index, feature in enumerate(ACTIVE_SNAPSHOT_META_FEATURES)
    }
    row.update(
        {
            "candidate_id": candidate_id,
            "strategy_id": f"strategy-{candidate_id}",
            "match_method": match_method,
            "pnl_pct": 1.0,
            "source": "live",
        }
    )
    return row


def test_hybrid_meta_pool_accepts_only_direct_linkage(monkeypatch) -> None:
    live = pd.DataFrame(
        [
            _live_row("direct-candidate", "linkage"),
            _live_row("forensic-candidate", "forensic"),
            _live_row("conflict-candidate", "linkage_conflict"),
        ]
    )

    class _FakeIngestor:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def ingest(self) -> pd.DataFrame:
            return live.copy()

    monkeypatch.setattr(live_outcome_ingestor, "LiveOutcomeIngestor", _FakeIngestor)
    builder = UnifiedTrainingBuilder()

    result = builder._load_live_outcome_rows(
        expired_bots_path="unused.xlsx",
        linkage_dir="unused-linkage",
        scanner_results_dir="unused-results",
    )

    assert result["candidate_id"].tolist() == ["direct-candidate"]


def test_hybrid_meta_pool_fails_closed_without_match_method(monkeypatch) -> None:
    live = pd.DataFrame([_live_row("unknown-candidate", "linkage")]).drop(
        columns=["match_method"]
    )

    class _FakeIngestor:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def ingest(self) -> pd.DataFrame:
            return live.copy()

    monkeypatch.setattr(live_outcome_ingestor, "LiveOutcomeIngestor", _FakeIngestor)
    builder = UnifiedTrainingBuilder()

    result = builder._load_live_outcome_rows(
        expired_bots_path="unused.xlsx",
        linkage_dir="unused-linkage",
        scanner_results_dir="unused-results",
    )

    assert result.empty


def test_explicit_live_union_fails_when_no_direct_rows(monkeypatch) -> None:
    builder = UnifiedTrainingBuilder()
    backtest = pd.DataFrame(
        [
            {
                "candidate_id": "BTCUSDT_20260731_120000_deadbeef",
                "symbol": "BTCUSDT",
                "source_class": "backtest",
                "version_gated": False,
                "is_authoritative": True,
            }
        ]
    )
    monkeypatch.setattr(builder, "_load_backtest_rows", lambda: backtest.copy())
    monkeypatch.setattr(
        builder,
        "_load_live_outcome_rows",
        lambda **_kwargs: pd.DataFrame(),
    )

    try:
        builder.build_meta_labeler_pool(include_live_outcomes=True)
    except ValueError as exc:
        assert "no governed direct-linked live outcomes" in str(exc)
    else:
        raise AssertionError("explicit live union silently continued without live rows")
