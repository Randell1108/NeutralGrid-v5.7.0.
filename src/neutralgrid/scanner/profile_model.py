from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any, Iterable, Mapping, Optional, cast

import numpy as np
import pandas as pd

from neutralgrid.core.config import get_config
from neutralgrid.core.constants import PROFIT_FACTOR_CAP
from neutralgrid.scanner._xlsx_io import (
    detect_format,
    raise_on_duplicate_strategy_id,
    read_sheet,
    read_single_sheet,
    validate_dataframe,
)
from neutralgrid.scanner.pattern_profile import DEFAULT_FEATURES


@dataclass(frozen=True)
class ProfileModel:
    """Probabilistic classifier for 'winner-like' regimes.

    Implements a Gaussian discriminant (log-likelihood ratio) over selected features.
    This is intentionally lightweight (numpy-only) and is suitable for real-time scoring.

    Score interpretation:
      - proba(row) returns an estimated probability in [0,1] that the row is winner-like.
      - llr(row) returns log(P(row|winner) / P(row|loser)).
    """

    features: list[str]
    winner_mu: dict[str, float]
    loser_mu: dict[str, float]
    inv_cov: list[list[float]]  # pooled, shrinkage-regularized inverse covariance
    prior_winner: float
    duration_band: Optional[dict[str, float]] = None
    # Phase 2.3 — z-score standardization, fit on concatenated labeled set.
    # Pre-Phase-2 artifacts have these as None; _vector falls back to raw values.
    feature_mean: Optional[dict[str, float]] = None
    feature_std: Optional[dict[str, float]] = None
    # Shared-median imputation vector fit on the concatenated labeled set.
    # Pre-existing artifacts may lack this; inference falls back to pooled
    # mean (if present) or the class-midpoint in artifact space.
    feature_impute: Optional[dict[str, float]] = None
    # Label policy provenance. Mirrors PatternProfile.selection_summary with
    # the extra field `losers_count` (pattern_profile is winner-only). Shared
    # keys: top_quantile, min_profit_factor, min_avg_profit_per_grid,
    # pnl_threshold, winners_count, winners_symbols, bounded/labeled universe
    # sizes, duration_band, xlsx source. Pre-existing artifacts lacking this
    # block load with selection_summary=None.
    selection_summary: Optional[dict[str, Any]] = None
    # Explicit scoring family. Legacy artifacts omit this field and therefore
    # retain the original pooled-covariance Gaussian discriminant exactly.
    model_family: str = "gaussian_lda_v1"
    # ``robust_logistic_v1`` stores coefficients in standardized feature
    # space. These fields are ignored by the legacy Gaussian family.
    linear_coef: Optional[dict[str, float]] = None
    linear_intercept: Optional[float] = None

    def _vector(self, row: Mapping[str, Any]) -> Optional[np.ndarray]:
        xs: list[float] = []
        for f in self.features:
            v = row.get(f, None)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                raw: Optional[float] = None
                if self.feature_impute is not None and f in self.feature_impute:
                    raw = float(self.feature_impute[f])
                elif self.feature_mean is not None and f in self.feature_mean:
                    raw = float(self.feature_mean[f])
                elif f in self.winner_mu and f in self.loser_mu:
                    xs.append((float(self.winner_mu[f]) + float(self.loser_mu[f])) / 2.0)
                    continue
                if raw is None:
                    return None
            else:
                try:
                    raw = float(v)
                except (TypeError, ValueError):
                    return None
            if (
                self.feature_mean is not None
                and self.feature_std is not None
                and f in self.feature_mean
                and f in self.feature_std
            ):
                std = float(self.feature_std[f])
                if std <= 0:
                    std = 1.0
                xs.append((raw - float(self.feature_mean[f])) / std)
            else:
                xs.append(raw)
        x = np.asarray(xs, dtype=float)
        if not np.all(np.isfinite(x)):
            return None
        return x

    def llr(self, row: Mapping[str, Any]) -> Optional[float]:
        x = self._vector(row)
        if x is None:
            return None

        if self.model_family == "robust_logistic_v1":
            if self.linear_coef is None or self.linear_intercept is None:
                return None
            try:
                coef = np.asarray(
                    [self.linear_coef[f] for f in self.features], dtype=float
                )
            except (KeyError, TypeError, ValueError):
                return None
            llr = float(x @ coef + float(self.linear_intercept))
            return llr if np.isfinite(llr) else None
        if self.model_family != "gaussian_lda_v1":
            return None

        mu_w = np.asarray([self.winner_mu[f] for f in self.features], dtype=float)
        mu_l = np.asarray([self.loser_mu[f] for f in self.features], dtype=float)
        inv = np.asarray(self.inv_cov, dtype=float)

        # Quadratic forms for pooled covariance
        dw = x - mu_w
        dl = x - mu_l
        q_w = float(dw.T @ inv @ dw)
        q_l = float(dl.T @ inv @ dl)

        # LLR up to an additive constant (det terms cancel with pooled cov).
        # Include class prior.
        prior = min(0.999, max(0.001, float(self.prior_winner)))
        llr = -0.5 * (q_w - q_l) + np.log(prior / (1.0 - prior))
        if not np.isfinite(llr):
            return None
        return float(llr)

    def proba(self, row: Mapping[str, Any]) -> Optional[float]:
        llr = self.llr(row)
        if llr is None:
            return None
        # Branch-stable logistic transform.  The direct ``exp(-llr)`` form
        # overflows for large negative but otherwise valid likelihood ratios.
        if llr >= 0.0:
            z = float(np.exp(-llr))
            p = 1.0 / (1.0 + z)
        else:
            z = float(np.exp(llr))
            p = z / (1.0 + z)
        return float(max(0.0, min(1.0, p)))

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "features": list(self.features),
            "winner_mu": dict(self.winner_mu),
            "loser_mu": dict(self.loser_mu),
            "inv_cov": self.inv_cov,
            "prior_winner": float(self.prior_winner),
            "model_family": self.model_family,
        }
        if self.duration_band is not None:
            result["duration_band"] = self.duration_band
        if self.feature_mean is not None:
            result["feature_mean"] = dict(self.feature_mean)
        if self.feature_std is not None:
            result["feature_std"] = dict(self.feature_std)
        if self.feature_impute is not None:
            result["feature_impute"] = dict(self.feature_impute)
        if self.selection_summary is not None:
            result["selection_summary"] = dict(self.selection_summary)
        if self.linear_coef is not None:
            result["linear_coef"] = dict(self.linear_coef)
        if self.linear_intercept is not None:
            result["linear_intercept"] = float(self.linear_intercept)
        return result

    @classmethod
    def from_json(cls, obj: Mapping[str, Any]) -> "ProfileModel":
        raw_band = obj.get("duration_band")
        duration_band: Optional[dict[str, float]] = None
        if isinstance(raw_band, dict):
            duration_band = {str(k): float(v) for k, v in raw_band.items()}
        raw_mean = obj.get("feature_mean")
        feature_mean: Optional[dict[str, float]] = None
        if isinstance(raw_mean, dict):
            feature_mean = {str(k): float(v) for k, v in raw_mean.items()}
        raw_std = obj.get("feature_std")
        feature_std: Optional[dict[str, float]] = None
        if isinstance(raw_std, dict):
            feature_std = {str(k): float(v) for k, v in raw_std.items()}
        raw_impute = obj.get("feature_impute")
        feature_impute: Optional[dict[str, float]] = None
        if isinstance(raw_impute, dict):
            feature_impute = {str(k): float(v) for k, v in raw_impute.items()}
        raw_summary = obj.get("selection_summary")
        selection_summary: Optional[dict[str, Any]] = None
        if isinstance(raw_summary, dict):
            selection_summary = dict(raw_summary)
        raw_linear_coef = obj.get("linear_coef")
        linear_coef: Optional[dict[str, float]] = None
        if isinstance(raw_linear_coef, dict):
            linear_coef = {
                str(k): float(v) for k, v in raw_linear_coef.items()
            }
        raw_linear_intercept = obj.get("linear_intercept")
        linear_intercept = (
            float(raw_linear_intercept)
            if raw_linear_intercept is not None
            else None
        )
        return cls(
            features=list(obj.get("features") or []),
            winner_mu={str(k): float(v) for k, v in (obj.get("winner_mu") or {}).items()},
            loser_mu={str(k): float(v) for k, v in (obj.get("loser_mu") or {}).items()},
            inv_cov=[[float(x) for x in row] for row in (obj.get("inv_cov") or [])],
            prior_winner=float(obj.get("prior_winner") or 0.5),
            duration_band=duration_band,
            feature_mean=feature_mean,
            feature_std=feature_std,
            feature_impute=feature_impute,
            selection_summary=selection_summary,
            model_family=str(obj.get("model_family") or "gaussian_lda_v1"),
            linear_coef=linear_coef,
            linear_intercept=linear_intercept,
        )


def train_profile_model_from_enhanced_xlsx(
    xlsx_path: str | Path,
    *,
    max_duration_hours: float = 7.0,
    min_profit_factor: float = 1.5,
    min_avg_profit_per_grid: Optional[float] = None,
    top_quantile: float = 0.75,
    features: Iterable[str] = DEFAULT_FEATURES,
    shrinkage: float = 0.30,
) -> ProfileModel:
    """Train a pooled-covariance Gaussian classifier using the enhanced winners file.

    Uses a bounded training universe: only rows with 0 <= duration_hours < max_duration_hours.
    """
    if min_avg_profit_per_grid is None:
        min_avg_profit_per_grid = get_config().grid.profit_grid_min_pct_static_fallback

    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Reference xlsx not found: {xlsx_path}")

    import logging as _logging
    _logger = _logging.getLogger(__name__)

    fmt = detect_format(xlsx_path)
    if fmt == "multi":
        df_entry = read_sheet(xlsx_path, "Entry Validation Metrics")
        df_perf = read_sheet(xlsx_path, "Performance Risk-Adjusted")
        df_mkt = read_sheet(xlsx_path, "Market and Volatility")

        all_warnings: list[str] = []
        df_entry, w1 = validate_dataframe(df_entry, "Entry Validation Metrics")
        df_perf, w2 = validate_dataframe(df_perf, "Performance Risk-Adjusted")
        df_mkt, w3 = validate_dataframe(df_mkt, "Market and Volatility")
        all_warnings.extend(w1 + w2 + w3)
        raise_on_duplicate_strategy_id(df_entry, context="Entry Validation Metrics")
        raise_on_duplicate_strategy_id(df_perf, context="Performance Risk-Adjusted")
        raise_on_duplicate_strategy_id(df_mkt, context="Market and Volatility")

        for warning in all_warnings:
            _logger.warning(f"Data validation: {warning}")

        # Allow-list merge from perf sheet (avoid duplicate columns)
        perf_cols_needed = [
            "strategy_id", "pnl_pct", "profit_factor", "duration_hours",
            "total_profit_usdt", "avg_profit_per_grid", "status",
        ]
        perf_cols_to_merge = ["strategy_id"] + [c for c in perf_cols_needed[1:] if c not in df_entry.columns]

        # Allow-list merge from mkt sheet. Post-Phase 5.1 the scanner no longer
        # declares `funding_rate_avg` / `open_interest_start` / `atr_avg` /
        # `volume_avg` as features, so nothing from the mkt sheet is currently
        # consumed. The merge is retained for multi-sheet format compatibility
        # but auto-skips via the `len(mkt_cols_to_merge) > 1` guard below.
        mkt_cols_needed = ["strategy_id"]
        mkt_cols_to_merge = ["strategy_id"] + [c for c in mkt_cols_needed[1:] if c not in df_entry.columns]

        df = df_entry.copy()
        if len(perf_cols_to_merge) > 1:
            df = df.merge(df_perf[perf_cols_to_merge], on="strategy_id", how="left")
        if len(mkt_cols_to_merge) > 1:
            df = df.merge(df_mkt[mkt_cols_to_merge], on="strategy_id", how="left")
    else:
        _logger.info("Single-sheet format detected; reading canonical sheet")
        df, sheet_name = read_single_sheet(xlsx_path)
        df, warnings_list = validate_dataframe(df, sheet_name)
        raise_on_duplicate_strategy_id(df, context=sheet_name)
        for warning in warnings_list:
            _logger.warning(f"Data validation: {warning}")

    # Coerce numeric
    numeric_cols = set(features) | {
        "pnl_pct", "profit_factor", "duration_hours", "avg_profit_per_grid"
    }
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # cap pathological profit_factor
    if "profit_factor" in df.columns:
        df["profit_factor"] = df["profit_factor"].replace([np.inf, -np.inf], np.nan)
        df["profit_factor"] = df["profit_factor"].clip(lower=0.0, upper=PROFIT_FACTOR_CAP)

    # ── Bounded training universe: 0 <= duration_hours < max_duration_hours ──
    duration_series = (
        cast(pd.Series, df["duration_hours"])
        if "duration_hours" in df.columns
        else pd.Series(np.nan, index=df.index, dtype=float)
    )
    df_train = cast(pd.DataFrame, df.loc[
        (duration_series >= 0.0) & (duration_series < max_duration_hours)
    ].copy())

    _logger.info(
        "Bounded universe: %d rows with 0 <= duration_hours < %.1f (of %d total)",
        len(df_train), max_duration_hours, len(df),
    )

    # ── df_labeled: exclude rows with missing label-defining fields ──
    pf_train = (
        cast(pd.Series, df_train["profit_factor"])
        if "profit_factor" in df_train.columns
        else pd.Series(np.nan, index=df_train.index, dtype=float)
    )
    pnl_train = (
        cast(pd.Series, df_train["pnl_pct"])
        if "pnl_pct" in df_train.columns
        else pd.Series(np.nan, index=df_train.index, dtype=float)
    )
    labelable_mask = pf_train.notna() & pnl_train.notna()
    df_labeled = cast(pd.DataFrame, df_train.loc[labelable_mask].copy())

    _logger.info(
        "Labeled subset: %d rows (excluded %d with missing profit_factor or pnl_pct)",
        len(df_labeled), len(df_train) - len(df_labeled),
    )

    # ── Compute pnl_thr on bounded universe only ──
    pnl_thr = float(cast(pd.Series, df_labeled["pnl_pct"]).quantile(top_quantile)) if len(df_labeled) > 0 else 0.0

    # ── Winner/loser selection within bounded labeled universe ──
    pf_labeled = cast(pd.Series, df_labeled["profit_factor"])
    pnl_labeled = cast(pd.Series, df_labeled["pnl_pct"])

    winners_mask = (
        (pf_labeled >= min_profit_factor)
        & (pnl_labeled >= pnl_thr)
    )

    # Enforce avg_profit_per_grid only when column is explicitly present in workbook
    if "avg_profit_per_grid" in df_labeled.columns:
        apg = cast(pd.Series, df_labeled["avg_profit_per_grid"])
        winners_mask = winners_mask & ((apg.isna()) | (apg >= min_avg_profit_per_grid))

    # Phase 2.1 — availability filter evaluated on labeled universe, not full df.
    # Using df (unbounded) was a leakage surface: features dense outside the
    # training universe but sparse inside it would still be admitted.
    feats = list(features)
    available: list[str] = []
    for f in feats:
        if f in df_labeled.columns and int(cast(pd.Series, df_labeled[f]).notna().sum()) >= 10:
            available.append(f)
    feats = available
    if not feats:
        raise ValueError(
            "No features available in the labeled universe (all dropped by availability filter)."
        )

    # Phase 2.4 — per-class sample floor: max(30, 3 * len(feats)). Floors below
    # this cannot support a len(feats)-dim Gaussian.
    min_samples = max(30, 3 * len(feats))

    df_w = cast(pd.DataFrame, df_labeled.loc[winners_mask].copy())
    if len(df_w) < min_samples:
        raise ValueError(
            f"Bounded universe produced only {len(df_w)} winners "
            f"(need >= max(30, 3*{len(feats)}) = {min_samples}). "
            f"Cannot train profile model on {len(df_labeled)} labeled rows "
            f"with pnl_thr={pnl_thr:.2f}, min_pf={min_profit_factor}."
        )

    # Losers from same bounded labeled universe (not workbook-wide complement)
    df_l = cast(pd.DataFrame, df_labeled.loc[~df_labeled.index.isin(df_w.index)].copy())
    if len(df_l) < min_samples:
        raise ValueError(
            f"Bounded universe produced only {len(df_l)} losers "
            f"(need >= max(30, 3*{len(feats)}) = {min_samples}). "
            f"Cannot train profile model."
        )

    Xw = cast(pd.DataFrame, df_w[feats]).astype(float).to_numpy()
    Xl = cast(pd.DataFrame, df_l[feats]).astype(float).to_numpy()

    # Phase 2.2 — shared-median imputation. Class-conditional imputation inflates
    # mu_w - mu_l when missingness correlates with the label.
    X_all = np.vstack([Xw, Xl])
    col_medians = np.nanmedian(X_all, axis=0)
    col_medians = np.where(np.isfinite(col_medians), col_medians, 0.0)

    def _fill_with(X: np.ndarray, fill: np.ndarray) -> np.ndarray:
        X = X.copy()
        inds = np.where(~np.isfinite(X))
        X[inds] = np.take(fill, inds[1])
        return X

    Xw = _fill_with(Xw, col_medians)
    Xl = _fill_with(Xl, col_medians)

    # Phase 2.3 — z-score standardization. Fit on concatenated labeled set so
    # that the 0.30 shrinkage target (identity) matches scale. Persist the
    # vectors in the artifact; ProfileModel._vector applies the same transform
    # at inference.
    X_all = np.vstack([Xw, Xl])
    feat_mean = np.mean(X_all, axis=0)
    feat_std = np.std(X_all, axis=0, ddof=0)
    feat_std = np.where(feat_std > 1e-12, feat_std, 1.0)
    Xw = (Xw - feat_mean) / feat_std
    Xl = (Xl - feat_mean) / feat_std

    mu_w = np.mean(Xw, axis=0)
    mu_l = np.mean(Xl, axis=0)

    # Pooled covariance on standardized features
    def _cov(X: np.ndarray) -> np.ndarray:
        if X.shape[0] <= 1:
            return np.eye(X.shape[1], dtype=float)
        return np.cov(X, rowvar=False)

    Sw = _cov(Xw)
    Sl = _cov(Xl)
    n_w = max(1, Xw.shape[0])
    n_l = max(1, Xl.shape[0])
    S = ((n_w - 1) * Sw + (n_l - 1) * Sl) / max(1, (n_w + n_l - 2))

    # Shrinkage towards diagonal for stability
    shrink = float(shrinkage)
    S = (1.0 - shrink) * S + shrink * np.diag(np.diag(S))

    # Add jitter
    S = S + 1e-6 * np.eye(S.shape[0], dtype=float)

    inv = np.linalg.inv(S)

    prior = float(n_w / (n_w + n_l))
    winner_mu = {f: float(mu_w[i]) for i, f in enumerate(feats)}
    loser_mu = {f: float(mu_l[i]) for i, f in enumerate(feats)}
    feature_mean = {f: float(feat_mean[i]) for i, f in enumerate(feats)}
    feature_std = {f: float(feat_std[i]) for i, f in enumerate(feats)}
    feature_impute = {f: float(col_medians[i]) for i, f in enumerate(feats)}

    if "symbol" in df_w.columns:
        winner_symbols = sorted(set(cast(pd.Series, df_w["symbol"]).astype(str).tolist()))
    else:
        winner_symbols = []

    selection_summary: dict[str, Any] = {
        "xlsx": str(xlsx_path),
        "top_quantile": float(top_quantile),
        "min_profit_factor": float(min_profit_factor),
        "min_avg_profit_per_grid": float(min_avg_profit_per_grid),
        "max_duration_hours": float(max_duration_hours),
        "duration_band": {"min_hours": 0.0, "max_hours": float(max_duration_hours)},
        "pnl_threshold": float(pnl_thr),
        "winners_count": int(len(df_w)),
        "losers_count": int(len(df_l)),
        "bounded_universe_size": int(len(df_train)),
        "labeled_universe_size": int(len(df_labeled)),
        "winners_symbols": winner_symbols,
        "shrinkage": float(shrinkage),
        "label_name": "fast_completed_winner_under_7h",
        "label_definition": {
            "duration_rule": (
                f"0 <= duration_hours < {float(max_duration_hours)}"
            ),
            "performance_rule": (
                "profit_factor >= min_profit_factor and "
                "pnl_pct >= train_only_top_quantile_threshold"
            ),
            "time_to_target_claimed": False,
        },
    }

    return ProfileModel(
        features=list(feats),
        winner_mu=winner_mu,
        loser_mu=loser_mu,
        inv_cov=inv.tolist(),
        prior_winner=prior,
        duration_band={"min_hours": 0.0, "max_hours": float(max_duration_hours)},
        feature_mean=feature_mean,
        feature_std=feature_std,
        feature_impute=feature_impute,
        selection_summary=selection_summary,
    )


def save_profile_model(model: ProfileModel, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.to_json(), indent=2, sort_keys=True), encoding="utf-8")


def load_profile_model(path: str | Path) -> ProfileModel:
    path = Path(path)
    obj = json.loads(path.read_text(encoding="utf-8"))
    model = ProfileModel.from_json(obj)
    unknown = sorted(set(model.features) - set(DEFAULT_FEATURES))
    if unknown:
        raise ValueError(
            f"Profile model contains unknown features outside the active contract: {unknown}"
        )
    if not model.features:
        raise ValueError("Profile model has no admitted features.")
    feature_count = len(model.features)
    if model.model_family not in {"gaussian_lda_v1", "robust_logistic_v1"}:
        raise ValueError(
            f"Profile model has unsupported model_family={model.model_family!r}"
        )
    if model.model_family == "robust_logistic_v1":
        if model.linear_coef is None or model.linear_intercept is None:
            raise ValueError(
                "Logistic profile model is missing linear_coef/linear_intercept"
            )
        missing_coef = sorted(set(model.features) - set(model.linear_coef))
        extra_coef = sorted(set(model.linear_coef) - set(model.features))
        if missing_coef or extra_coef:
            raise ValueError(
                "Logistic profile coefficient schema differs from active features: "
                f"missing={missing_coef}, extra={extra_coef}"
            )
        if not all(
            np.isfinite(float(model.linear_coef[feature]))
            for feature in model.features
        ) or not np.isfinite(float(model.linear_intercept)):
            raise ValueError("Logistic profile model contains non-finite parameters")
        if model.feature_mean is None or model.feature_std is None:
            raise ValueError(
                "Logistic profile model requires persisted feature_mean/feature_std"
            )
    inv_cov = np.asarray(model.inv_cov, dtype=float)
    if inv_cov.shape != (feature_count, feature_count):
        raise ValueError(
            "Profile model inv_cov shape "
            f"{inv_cov.shape} does not match {feature_count} features"
        )
    if not np.all(np.isfinite(inv_cov)):
        raise ValueError("Profile model contains non-finite inv_cov parameters")
    if not np.isfinite(float(model.prior_winner)):
        raise ValueError("Profile model contains non-finite prior_winner")
    for field_name, values in (
        ("winner_mu", model.winner_mu),
        ("loser_mu", model.loser_mu),
    ):
        missing = sorted(set(model.features) - set(values))
        if missing:
            raise ValueError(
                f"Profile model {field_name} is missing active features: {missing}"
            )
        if not all(np.isfinite(float(values[feature])) for feature in model.features):
            raise ValueError(
                f"Profile model contains non-finite {field_name} parameters"
            )
    for field_name, values in (
        ("feature_mean", model.feature_mean),
        ("feature_std", model.feature_std),
        ("feature_impute", model.feature_impute),
    ):
        if values is None:
            continue
        missing = sorted(set(model.features) - set(values))
        if missing:
            raise ValueError(
                f"Profile model {field_name} is missing active features: {missing}"
            )
        if not all(np.isfinite(float(values[feature])) for feature in model.features):
            raise ValueError(
                f"Profile model contains non-finite {field_name} parameters"
            )
    return model
