"""
PnL-aware candidate ranking with empirical EV alignment.

The analytical EV model is kept explicit and transparent, then aligned to
simulated outcomes using an empirical profile built from backtest results.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Optional, Dict, Any

from neutralgrid.core.constants import BOT_HORIZON_HOURS
from neutralgrid.scanner.empirical_profile_v20260302 import (
    EmpiricalProfile,
    align_ev_with_profile_context,
    empirical_profile_fingerprint,
    load_empirical_profile_cached,
    resolve_fill_rate_scale,
)


EV_CONTRACT_SCHEMA = "pnl_ranker_ev_v1"


@dataclass(frozen=True)
class RankingConfig:
    """Configuration for PnL-aware ranking."""

    # Horizon for EV calculation
    horizon_hours: float = BOT_HORIZON_HOURS

    # Stop loss threshold (decimal)
    sl_pct: float = -0.12

    # PnL hurdle for "success" definition (decimal)
    pnl_hurdle_pct: float = 0.03

    # Funding rate thresholds
    funding_extreme_threshold: float = 0.003

    # Expected fills per hour baseline
    baseline_fills_per_hour: float = 2.0

    # Empirical alignment profile
    use_empirical_alignment: bool = True
    empirical_profile_dir: str = "data/backtest_candidates"
    empirical_alignment_min_samples: int = 20

    # Penalty weights
    trend_penalty_weight: float = 0.5
    funding_penalty_weight: float = 0.0  # Funding handled in EV directly
    range_penalty_weight: float = 0.2

    # Range size bounds (percent)
    optimal_range_min_pct: float = 1.5
    optimal_range_max_pct: float = 5.0


@dataclass
class RankingComponents:
    """Decomposed ranking score components."""

    ev: float
    fill_revenue_pct: float
    funding_cost_pct: float
    boundary_loss_pct: float
    trend_penalty: float
    funding_penalty: float
    range_penalty: float
    total_penalty: float
    rank_score: float

    # EV consistency diagnostics
    ev_raw: float = 0.0
    ev_aligned: float = 0.0
    expected_fills: float = 0.0
    fill_rate_scale: float = 1.0
    fill_rate_scope: str = "none"
    fill_rate_scope_samples: float = 0.0
    ev_alignment_scope: str = "none"
    ev_alignment_scope_samples: float = 0.0
    ev_alignment_r2: float = 0.0

    inputs: Dict[str, Any] = field(default_factory=dict)

    @property
    def ev_24h(self) -> float:
        """Backward compatibility alias."""
        return self.ev


class PnLRanker:
    """
    Risk-adjusted candidate ranker.

    Score = EV_aligned * (1 - total_penalty)
    """

    def __init__(self, config: Optional[RankingConfig] = None):
        self.config = config or RankingConfig()
        self._empirical_profile: EmpiricalProfile | None = None
        if self.config.use_empirical_alignment:
            self._empirical_profile = load_empirical_profile_cached(
                str(self.config.empirical_profile_dir)
            )

    @property
    def empirical_profile(self) -> EmpiricalProfile | None:
        return self._empirical_profile

    @property
    def ev_contract_payload(self) -> Dict[str, Any]:
        """Canonical, output-relevant EV configuration for lineage auditing."""
        return {
            "schema": EV_CONTRACT_SCHEMA,
            "ranking_config": asdict(self.config),
            "empirical_profile_fingerprint": (
                empirical_profile_fingerprint(self._empirical_profile)
                if self._empirical_profile is not None
                else None
            ),
        }

    @property
    def ev_contract_fingerprint(self) -> str:
        canonical = json.dumps(
            self.ev_contract_payload,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _expected_fills(
        self,
        *,
        num_grids: int,
        fills_per_hour: float,
        symbol: str | None = None,
        trend_prob: float | None = None,
    ) -> tuple[float, float, str, float]:
        # Keep this formula unchanged for AFML bugfix compatibility tests.
        active_fraction = min(0.75, (num_grids / 50.0) ** 0.5 * 0.5)
        expected_fills = fills_per_hour * self.config.horizon_hours * active_fraction
        fill_rate_scale = 1.0
        fill_scope = "none"
        fill_scope_samples = 0.0
        if self._empirical_profile is not None:
            fill_ctx = resolve_fill_rate_scale(
                self._empirical_profile,
                symbol=symbol,
                trend_prob=trend_prob,
                min_samples=self.config.empirical_alignment_min_samples,
            )
            fill_rate_scale = float(fill_ctx["fill_rate_scale"])
            fill_scope = str(fill_ctx["scope"])
            fill_scope_samples = float(fill_ctx["samples"])
            expected_fills *= fill_rate_scale
        return expected_fills, fill_rate_scale, fill_scope, fill_scope_samples

    def compute_ev(
        self,
        profit_per_grid_pct: float,
        num_grids: int,
        survival_prob: float,
        funding_rate: Optional[float] = None,
        fills_per_hour: Optional[float] = None,
        symbol: Optional[str] = None,
        trend_prob: Optional[float] = None,
        leverage: int = 10,
    ) -> float:
        """
        Expected net return over horizon (percent), empirically aligned.
        """
        fph = fills_per_hour if fills_per_hour is not None else self.config.baseline_fills_per_hour
        expected_fills, _, _, _ = self._expected_fills(
            num_grids=num_grids,
            fills_per_hour=fph,
            symbol=symbol,
            trend_prob=trend_prob,
        )

        fill_revenue = survival_prob * expected_fills * (profit_per_grid_pct / 100.0)
        # Gate 9: funding_rate is per-NOTIONAL, but fill_revenue and boundary_loss
        # are per-MARGIN.  Multiply by leverage to convert funding to margin basis.
        funding_cost = 0.0
        if funding_rate is not None:
            funding_cost = abs(funding_rate) * leverage * (self.config.horizon_hours / 8.0)
        boundary_loss = (1.0 - survival_prob) * abs(self.config.sl_pct)
        ev_raw = (fill_revenue - funding_cost - boundary_loss) * 100.0

        if self._empirical_profile is None:
            return float(ev_raw)
        alignment = align_ev_with_profile_context(
            ev_raw_pct=ev_raw,
            profile=self._empirical_profile,
            symbol=symbol,
            trend_prob=trend_prob,
            min_samples=self.config.empirical_alignment_min_samples,
        )
        return float(alignment["ev_aligned"])

    # Backward compatibility alias
    compute_ev_24h = compute_ev

    def compute_penalties(
        self,
        trend_prob: float,
        funding_rate: Optional[float],
        range_size_pct: float,
    ) -> tuple[float, float, float, float]:
        """Compute trend/funding/range penalties in [0, 1]."""
        trend_penalty = trend_prob * self.config.trend_penalty_weight

        if funding_rate is not None:
            abs_funding = abs(funding_rate)
            if abs_funding > self.config.funding_extreme_threshold:
                excess = (
                    abs_funding - self.config.funding_extreme_threshold
                ) / self.config.funding_extreme_threshold
                funding_penalty = min(1.0, excess) * self.config.funding_penalty_weight
            else:
                funding_penalty = 0.0
        else:
            funding_penalty = 0.0

        if range_size_pct < self.config.optimal_range_min_pct:
            gap = self.config.optimal_range_min_pct - range_size_pct
            range_penalty = (
                min(1.0, gap / self.config.optimal_range_min_pct)
                * self.config.range_penalty_weight
            )
        elif range_size_pct > self.config.optimal_range_max_pct:
            gap = range_size_pct - self.config.optimal_range_max_pct
            range_penalty = (
                min(1.0, gap / self.config.optimal_range_max_pct)
                * self.config.range_penalty_weight
            )
        else:
            range_penalty = 0.0

        total_penalty = min(1.0, trend_penalty + funding_penalty + range_penalty)
        return trend_penalty, funding_penalty, range_penalty, total_penalty

    def rank_score(
        self,
        profit_per_grid_pct: float,
        num_grids: int,
        survival_prob: float,
        trend_prob: float,
        funding_rate: Optional[float] = None,
        range_size_pct: float = 3.0,
        fills_per_hour: Optional[float] = None,
        symbol: Optional[str] = None,
        leverage: int = 10,
    ) -> float:
        """Compute final risk-adjusted rank score."""
        result = self.compute_score(
            profit_per_grid_pct=profit_per_grid_pct,
            num_grids=num_grids,
            survival_prob=survival_prob,
            trend_prob=trend_prob,
            funding_rate=funding_rate,
            range_size_pct=range_size_pct,
            fills_per_hour=fills_per_hour,
            symbol=symbol,
            leverage=leverage,
        )
        return float(result.rank_score)

    def compute_score(
        self,
        profit_per_grid_pct: float,
        num_grids: int,
        survival_prob: float,
        trend_prob: float,
        funding_rate: Optional[float] = None,
        range_size_pct: float = 3.0,
        fills_per_hour: Optional[float] = None,
        symbol: Optional[str] = None,
        leverage: int = 10,
    ) -> RankingComponents:
        """Compute full ranking decomposition."""
        fph = fills_per_hour if fills_per_hour is not None else self.config.baseline_fills_per_hour
        expected_fills, fill_rate_scale, fill_rate_scope, fill_rate_scope_samples = self._expected_fills(
            num_grids=num_grids,
            fills_per_hour=fph,
            symbol=symbol,
            trend_prob=trend_prob,
        )
        fill_revenue_pct = survival_prob * expected_fills * (profit_per_grid_pct / 100.0) * 100.0
        # Gate 9: funding_rate is per-NOTIONAL, but fill_revenue_pct and
        # boundary_loss_pct are per-MARGIN.  Multiply by leverage to convert
        # funding to the same margin basis as revenue and loss.
        funding_cost_pct = 0.0
        if funding_rate is not None:
            funding_cost_pct = abs(funding_rate) * leverage * (self.config.horizon_hours / 8.0) * 100.0
        boundary_loss_pct = (1.0 - survival_prob) * abs(self.config.sl_pct) * 100.0

        ev_raw = fill_revenue_pct - funding_cost_pct - boundary_loss_pct
        ev_aligned = ev_raw
        ev_align_scope = "none"
        ev_align_samples = 0.0
        ev_align_r2 = 0.0
        if self._empirical_profile is not None:
            alignment = align_ev_with_profile_context(
                ev_raw_pct=ev_raw,
                profile=self._empirical_profile,
                symbol=symbol,
                trend_prob=trend_prob,
                min_samples=self.config.empirical_alignment_min_samples,
            )
            ev_aligned = float(alignment["ev_aligned"])
            ev_align_scope = str(alignment["scope"])
            ev_align_samples = float(alignment["samples"])
            ev_align_r2 = float(alignment["r2"])

        trend_penalty, funding_penalty, range_penalty, total_penalty = self.compute_penalties(
            trend_prob=trend_prob,
            funding_rate=funding_rate,
            range_size_pct=range_size_pct,
        )
        rank_score = ev_aligned * (1.0 - total_penalty)

        return RankingComponents(
            ev=float(ev_aligned),
            fill_revenue_pct=float(fill_revenue_pct),
            funding_cost_pct=float(funding_cost_pct),
            boundary_loss_pct=float(boundary_loss_pct),
            trend_penalty=float(trend_penalty),
            funding_penalty=float(funding_penalty),
            range_penalty=float(range_penalty),
            total_penalty=float(total_penalty),
            rank_score=float(rank_score),
            ev_raw=float(ev_raw),
            ev_aligned=float(ev_aligned),
            expected_fills=float(expected_fills),
            fill_rate_scale=float(fill_rate_scale),
            fill_rate_scope=fill_rate_scope,
            fill_rate_scope_samples=float(fill_rate_scope_samples),
            ev_alignment_scope=ev_align_scope,
            ev_alignment_scope_samples=float(ev_align_samples),
            ev_alignment_r2=float(ev_align_r2),
            inputs={
                "profit_per_grid_pct": profit_per_grid_pct,
                "num_grids": num_grids,
                "survival_prob": survival_prob,
                "trend_prob": trend_prob,
                "funding_rate": funding_rate,
                "range_size_pct": range_size_pct,
                "symbol": symbol,
                "leverage": leverage,
            },
        )


def create_pnl_ranker(
    horizon_hours: float = BOT_HORIZON_HOURS,
    sl_pct: float = -0.12,
    pnl_hurdle_pct: float = 0.03,
    funding_extreme_threshold: float = 0.003,
) -> PnLRanker:
    """Factory function with standard settings."""
    config = RankingConfig(
        horizon_hours=horizon_hours,
        sl_pct=sl_pct,
        pnl_hurdle_pct=pnl_hurdle_pct,
        funding_extreme_threshold=funding_extreme_threshold,
    )
    return PnLRanker(config)
