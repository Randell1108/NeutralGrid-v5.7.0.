"""Development-only tests of additional recorded scan-time feature families.

These tests never read held-out outcomes or scores and cannot authorize promotion.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import study as s
from neutralgrid.backtest.candidate_pipeline import _SCANNER_TO_FEATURE
from neutralgrid.models.meta_labeler import _is_label_column


def main():
    logging.basicConfig(level=logging.INFO)
    plan = json.loads((s.OUT / "plan.json").read_text())
    original = s.FEATURES.copy()
    families = {
        "context": ["long_short_ratio", "funding_rate_zscore", "open_interest_change_pct", "bb_width_ratio_1h_15m"],
        "flow": ["top_account_lsr_log", "top_position_lsr_log", "top_position_vs_account_delta", "global_lsr_log", "taker_imbalance", "taker_buy_sell_log", "basis_pct"],
        "liquidity": ["spread_pct", "top20_bid_depth_usdt", "top20_ask_depth_usdt", "top20_depth_min_usdt", "book_imbalance_top20", "oi_notional", "oi_to_volume"],
        "pattern_inputs": ["parkinson_vol_ratio_4h_24h_pre", "variance_ratio_1m_15m_pre_2h", "funding_carry_expected_next_7h", "liquidity_stability_z_1h"],
    }
    extras = sum(families.values(), [])
    assert all(f in _SCANNER_TO_FEATURE.values() and not _is_label_column(f) for f in extras)
    groups = {"plus_" + k: original + v for k, v in families.items()}
    groups.update({"all_recorded": original + extras,
                   "all_without_ou": [f for f in original if f != "ou_halflife"] + extras,
                   "all_without_ev_ou": [f for f in original if f not in {"ev_score", "ou_halflife"}] + extras,
                   "additional_only": extras})
    s.write("supplement_plan.json", {"families": families, "groups": groups, "configurations": 16,
        "evaluation": "Same three development blocks and calibration/purge rules as study.py; NO holdout read, NO promotion selection.",
        "reason": "Inventory verified 22 additional complete scan-time columns after original study plan froze.",
        "exclusions": "No hlabel/outcome, primary_pipeline_score, surrogate predictions, HMM probabilities, or composite micro_osc_score added."})
    # Restrict to development IDs immediately; held-out targets are not used.
    data = pd.read_csv(s.TEMP / "complete_training.csv", low_memory=False)
    data["scan_time"] = pd.to_datetime(data.scan_time, utc=True)
    data = data.loc[data.scan_time.isin(pd.to_datetime(plan["development_scan_groups"], utc=True))].copy()
    data["event_end"] = pd.to_datetime(data.event_end, utc=True)
    for f in original + extras:
        data[f] = pd.to_numeric(data[f], errors="raise")
        assert np.isfinite(data[f]).all(), f
    raw = pd.read_csv(s.SOURCE_DIR / "backtest/training_data_20260904.csv", low_memory=False).set_index("candidate_id")
    for f in extras:
        assert np.array_equal(data[f].to_numpy(float), raw.loc[data.candidate_id, f].to_numpy(float)), f
    scans = sorted(data.scan_time.unique())
    blocks = [scans[-6:-4], scans[-4:-2], scans[-2:]]
    results = []
    for name, features in groups.items():
        for kind in ["logistic", "vote_logit_hgb"]:
            s.FEATURES = features
            spec = {"id": f"{kind}_{name}", "set": "full", "k": len(features), "estimator": kind}
            probs, ys, folds = [], [], []
            for block in blocks:
                test = data.loc[data.scan_time.isin(block)]
                past = data.loc[data.scan_time < min(block)]
                p, used, support = s.fit_predict(spec, past, test)
                y = test.fast_winner_target.to_numpy(int)
                probs.append(p)
                ys.append(y)
                folds.append({**s.metrics(y, p), **support, "features": used})
            result = {**spec, "features": features, "pooled": s.metrics(np.concatenate(ys), np.concatenate(probs)),
                      "mean_fold_auc": float(np.mean([f["auc"] for f in folds])), "folds": folds}
            results.append(result)
            s.write("supplement_development.json", results)
            print(json.dumps({"trial": spec["id"], "features": len(features), "mean_auc": result["mean_fold_auc"], **result["pooled"]}), flush=True)
    print("SUPPLEMENT_COMPLETE", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
