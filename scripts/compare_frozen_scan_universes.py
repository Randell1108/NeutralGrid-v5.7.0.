"""Compare 250/250 and 350/350 selection from one frozen market-data boundary.

The normal pipeline refreshes market data during enrichment, so sequential live
runs are not a valid universe-size comparison. This script captures one public
350-symbol surface, replays both scenarios from that exact surface, and records
the artifact fingerprints used by both runs. It does not create exchange orders.
"""

from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
from pathlib import Path
import sys
from typing import Any, cast

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neutralgrid.api.binance_client import BinanceClient
from neutralgrid.core.config import get_config
from neutralgrid.models.artifacts import resolve_hmm_artifact_dir
from neutralgrid.models.meta_labeler import MetaLabeler
from neutralgrid.scanner.enrich_grid_params import EnrichConfig, enrich_with_grid_params
from neutralgrid.scanner.pattern_profile import PatternProfile
from neutralgrid.scanner.profile_model import load_profile_model
from neutralgrid.scanner.profile_model_walkforward import (
    resolve_active_pattern_profile_path,
    resolve_active_profile_model_path,
)
from neutralgrid.scanner.scan import (
    get_top_usdtm_symbols_by_quote_volume,
    scan_top_symbols,
)
from neutralgrid.validation.hmm_regime import ensure_hmm_model

from run_full_pipeline import _apply_afml_post_scoring, _build_potential_candidates


logger = logging.getLogger(__name__)


class BoundaryScanClient:
    """Freeze the universe-ranking responses while scan captures symbol data once."""

    def __init__(self, live: BinanceClient, exchange: dict[str, Any], tickers: Any) -> None:
        self._live = live
        self._exchange = deepcopy(exchange)
        self._tickers = deepcopy(tickers)
        self._ticker_endpoint = get_config().binance.endpoints["ticker"]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._live, name)

    async def get_exchange_info(self, symbol: str | None = None) -> dict[str, Any]:
        if symbol is None:
            return deepcopy(self._exchange)
        return await self._live.get_exchange_info(symbol)

    async def _request(self, endpoint: str, params: dict[str, Any] | None = None, *args: Any, **kwargs: Any) -> Any:
        if endpoint == self._ticker_endpoint and not params:
            return deepcopy(self._tickers)
        return await self._live._request(endpoint, params, *args, **kwargs)


class FrozenEnrichmentClient:
    """Return only captured market-data payloads; no replay call reaches Binance."""

    def __init__(self, market_data: dict[str, Any]) -> None:
        self._market_data = market_data

    async def get_all_market_data(self, symbol: str) -> dict[str, Any]:
        payload = self._market_data.get(symbol)
        if not isinstance(payload, dict):
            raise RuntimeError(f"frozen_market_data_missing:{symbol}")
        if "__capture_error__" in payload:
            raise RuntimeError(str(payload["__capture_error__"]))
        return deepcopy(payload)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_fingerprints(profile_path: Path, model_path: Path) -> dict[str, str]:
    hmm_dir = resolve_hmm_artifact_dir()
    paths = {
        "hmm_metadata": hmm_dir / "metadata.json",
        "hmm_model": hmm_dir / "model.joblib",
        "meta_model": ROOT / "models" / "meta_labeler.pkl",
        "meta_metadata": ROOT / "models" / "meta_labeler" / "metadata.json",
        "pattern_profile": profile_path,
        "profile_model": model_path,
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required artifact(s) missing: {', '.join(missing)}")
    return {name: _sha256(path) for name, path in paths.items()}


def _load_models() -> tuple[PatternProfile, Any, MetaLabeler, Path, Path]:
    cfg = get_config()
    profile_dir = cfg.resolve_path(cfg.artifacts.profile_dir)
    profile_path = resolve_active_pattern_profile_path(profile_dir)
    model_path = resolve_active_profile_model_path(profile_dir)
    profile = PatternProfile.load_json(profile_path)
    if profile is None:
        raise ValueError(f"Pattern profile could not be parsed: {profile_path}")
    profile_model = load_profile_model(model_path)
    if profile_model.features != profile.features:
        raise ValueError("Pattern profile/model feature mismatch")
    meta = MetaLabeler.load(ROOT / "models" / "meta_labeler.pkl")
    if not meta.is_trained or meta.promotion_status != "pass":
        raise ValueError("Active meta-labeler is not trained and promotion-approved")
    return profile, profile_model, meta, profile_path, model_path


async def _capture_market_data(live: BinanceClient, symbols: list[str], concurrency: int) -> dict[str, Any]:
    semaphore = asyncio.Semaphore(concurrency)

    async def capture_one(symbol: str) -> tuple[str, dict[str, Any]]:
        async with semaphore:
            try:
                return symbol, await live.get_all_market_data(symbol)
            except Exception as exc:
                return symbol, {"__capture_error__": f"{type(exc).__name__}:{exc}"}

    captured = await asyncio.gather(*(capture_one(symbol) for symbol in symbols))
    return dict(captured)


async def _capture_bundle(output_dir: Path, concurrency: int) -> Path:
    profile, profile_model, meta, profile_path, model_path = _load_models()
    await ensure_hmm_model(force=False)
    before = _artifact_fingerprints(profile_path, model_path)

    live = BinanceClient()
    try:
        status = await live.check_connection()
        if not status.get("connected"):
            raise RuntimeError("Binance public API connection failed")

        exchange = await live.get_exchange_info()
        ticker_endpoint = get_config().binance.endpoints["ticker"]
        tickers = await live._request(ticker_endpoint, params={})
        boundary_client = BoundaryScanClient(live, exchange, tickers)
        logger.info("Capturing frozen 350-symbol ranked universe")
        ranked = await get_top_usdtm_symbols_by_quote_volume(
            cast(BinanceClient, boundary_client), top_n=350
        )
        if len(ranked) < 350:
            raise RuntimeError(f"Frozen universe has only {len(ranked)} eligible symbols; need 350")

        scan_cache: dict[str, dict[str, Any]] = {}
        logger.info("Capturing scanner inputs for %d symbols", len(ranked))
        scan_frame = await scan_top_symbols(
            client=cast(BinanceClient, boundary_client),
            profile=profile,
            profile_model=profile_model,
            meta_labeler=meta,
            top_n=350,
            max_concurrency=8,
            min_delay_s=0.05,
            scan_data_out=scan_cache,
        )
        if scan_frame.empty:
            raise RuntimeError("Frozen scan returned no rows")

        universe_symbols = [symbol for symbol, _ in ranked]
        logger.info("Capturing enrichment inputs for %d symbols", len(universe_symbols))
        market_data = await _capture_market_data(live, universe_symbols, concurrency)
        # The boundary closes only after every payload used by both replays exists.
        boundary_utc = datetime.now(timezone.utc)
        after = _artifact_fingerprints(profile_path, model_path)
        if before != after:
            raise RuntimeError("Artifact fingerprints changed during boundary capture")

        bundle = {
            "schema_version": 1,
            "boundary_utc": boundary_utc.isoformat(),
            "universe": [{"symbol": symbol, "quote_volume_24h": volume} for symbol, volume in ranked],
            "scan_frame": scan_frame,
            "scan_cache": scan_cache,
            "market_data": market_data,
            "artifact_fingerprints": before,
        }
        bundle_path = output_dir / "frozen_boundary.joblib"
        joblib.dump(bundle, bundle_path, compress=3)
        return bundle_path
    finally:
        await live.close()


def _series(frame: pd.DataFrame, column: str, default: Any = pd.NA) -> pd.Series:
    if column in frame.columns:
        return cast(pd.Series, frame[column])
    return pd.Series(default, index=frame.index)


def _bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return _series(frame, column, False).astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _counts(values: pd.Series, limit: int = 25) -> dict[str, int]:
    cleaned = values.fillna("missing").astype(str).replace("", "missing")
    return {str(key): int(value) for key, value in cleaned.value_counts().head(limit).items()}


def _scenario_summary(frame: pd.DataFrame, universe_size: int) -> dict[str, Any]:
    terminal = (
        _bool_series(frame, "grid_is_valid")
        & _bool_series(frame, "hard_gate_passed")
        & _bool_series(frame, "stage_b_approved")
    )
    ev = cast(pd.Series, pd.to_numeric(_series(frame, "ev_score"), errors="coerce"))
    meta = cast(pd.Series, pd.to_numeric(_series(frame, "meta_prob"), errors="coerce"))
    terminal_count = int(terminal.sum())
    enriched = _series(frame, "grid_reason").notna()
    hard_gate = _bool_series(frame, "hard_gate_passed")
    stage_b = _bool_series(frame, "stage_b_approved")
    positive_ev = int((terminal & ev.gt(0)).sum())
    return {
        "universe_size": universe_size,
        "scanned_rows": int(len(frame)),
        "enriched_rows": int(enriched.sum()),
        "terminal_approved_count": terminal_count,
        "terminal_positive_ev_count": positive_ev,
        "terminal_positive_ev_share": (positive_ev / terminal_count) if terminal_count else None,
        "meta_prob_coverage": {
            "all_rows": int(meta.notna().sum()),
            "terminal_rows": int(meta.loc[terminal].notna().sum()),
        },
        "failure_stage_distribution": _counts(_series(frame, "failure_stage", "missing")),
        "hard_gate_failure_distribution": _counts(_series(frame, "hard_gate_reason", "missing").loc[~hard_gate]),
        "stage_b_failure_distribution": _counts(_series(frame, "stage_b_reason", "missing").loc[~stage_b]),
    }


async def _replay(bundle_path: Path, output_dir: Path) -> Path:
    bundle = joblib.load(bundle_path)
    if bundle.get("schema_version") != 1:
        raise ValueError("Unsupported frozen-boundary schema")
    profile, _, meta, profile_path, model_path = _load_models()
    current = _artifact_fingerprints(profile_path, model_path)
    if current != bundle.get("artifact_fingerprints"):
        raise RuntimeError("Artifact fingerprints differ from frozen boundary")

    scan_frame = bundle["scan_frame"].copy()
    universe = bundle["universe"]
    ordered_symbols = [str(row["symbol"]) for row in universe]
    frozen_client = FrozenEnrichmentClient(bundle["market_data"])
    scan_cache = bundle["scan_cache"]
    scenario_frames: dict[int, pd.DataFrame] = {}

    for size in (250, 350):
        logger.info("Replaying frozen %d/%d scenario", size, size)
        allowed = set(ordered_symbols[:size])
        candidates = scan_frame.loc[scan_frame["symbol"].astype(str).isin(allowed)].copy()
        enriched = await enrich_with_grid_params(
            df_candidates=candidates,
            client=cast(BinanceClient, frozen_client),
            pattern_profile=profile,
            cfg=EnrichConfig(score_threshold=45.0, max_symbols=size, concurrency=5),
            scan_data_cache=scan_cache,
            meta_labeler=meta,
        )
        enriched = _apply_afml_post_scoring(enriched)
        scenario_frames[size] = enriched
        enriched.to_csv(output_dir / f"deployment_ready_{size}_{size}.csv", index=False)
        _build_potential_candidates(enriched).to_csv(
            output_dir / f"potential_candidates_{size}_{size}.csv", index=False
        )

    after = _artifact_fingerprints(profile_path, model_path)
    if current != after:
        raise RuntimeError("Artifact fingerprints changed during frozen replay")

    summaries = {str(size): _scenario_summary(frame, size) for size, frame in scenario_frames.items()}
    terminal_250 = set(
        scenario_frames[250].loc[
            _bool_series(scenario_frames[250], "grid_is_valid")
            & _bool_series(scenario_frames[250], "hard_gate_passed")
            & _bool_series(scenario_frames[250], "stage_b_approved"),
            "symbol",
        ].astype(str)
    )
    terminal_350 = set(
        scenario_frames[350].loc[
            _bool_series(scenario_frames[350], "grid_is_valid")
            & _bool_series(scenario_frames[350], "hard_gate_passed")
            & _bool_series(scenario_frames[350], "stage_b_approved"),
            "symbol",
        ].astype(str)
    )
    boundary = datetime.fromisoformat(str(bundle["boundary_utc"]))
    report = {
        "schema_version": 1,
        "boundary_utc": boundary.isoformat(),
        "artifact_fingerprints": current,
        "scenarios": summaries,
        "terminal_symbol_delta": {
            "shared": sorted(terminal_250 & terminal_350),
            "only_250": sorted(terminal_250 - terminal_350),
            "only_350": sorted(terminal_350 - terminal_250),
        },
        "temporal_oos": {
            "status": "pending",
            "eligible_after_utc": (boundary + timedelta(hours=7)).isoformat(),
            "reason": "The active target needs future net-MTM outcome observations; none are available at capture time.",
        },
    }
    report_path = output_dir / "comparison_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replay-bundle", type=Path)
    parser.add_argument("--capture-concurrency", type=int, default=5)
    return parser.parse_args()


async def _main() -> int:
    args = _parse_args()
    if args.capture_concurrency <= 0:
        raise ValueError("--capture-concurrency must be > 0")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.replay_bundle is not None:
        bundle_path = args.replay_bundle
    else:
        bundle_path = await _capture_bundle(args.output_dir, args.capture_concurrency)
    report_path = await _replay(bundle_path, args.output_dir)
    logger.info("Frozen-universe comparison complete: %s", report_path)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(asyncio.run(_main()))
