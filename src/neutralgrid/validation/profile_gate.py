from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class ProfileGate:
    """Learned RSI/ADX bounds derived from historical winners."""

    adx_1h_max: float
    adx_15m_max: float
    adx_5m_max: float
    rsi_15m_low: float
    rsi_15m_high: float

    @staticmethod
    def _widen(lo: float, hi: float, widen: float) -> tuple[float, float]:
        w = (hi - lo) * float(widen)
        return lo - w, hi + w

    @classmethod
    def from_pattern_profile(cls, profile, *, widen: float = 0.15) -> "ProfileGate":
        """Build from scanner.pattern_profile.PatternProfile (q10/q90 bands)."""
        def band(name: str, fallback_lo: float, fallback_hi: float) -> tuple[float, float]:
            lo = profile.q10.get(name)
            hi = profile.q90.get(name)
            if lo is None or hi is None:
                lo, hi = fallback_lo, fallback_hi
            lo, hi = cls._widen(float(lo), float(hi), widen)
            return lo, hi

        _, adx1_hi = band("adx_1h", 0.0, 22.0)
        _, adx15_hi = band("adx_15m", 0.0, 23.0)
        _, adx5_hi = band("adx_5m", 0.0, 30.0)
        rsi_lo, rsi_hi = band("rsi_15m", 45.0, 55.0)

        # Defensive clamps to prevent pathological learned bounds.
        adx1_hi = min(adx1_hi, 45.0)
        adx15_hi = min(adx15_hi, 50.0)
        adx5_hi = min(adx5_hi, 60.0)

        rsi_lo = max(rsi_lo, 5.0)
        rsi_hi = min(rsi_hi, 95.0)

        return cls(
            adx_1h_max=float(adx1_hi),
            adx_15m_max=float(adx15_hi),
            adx_5m_max=float(adx5_hi),
            rsi_15m_low=float(rsi_lo),
            rsi_15m_high=float(rsi_hi),
        )
