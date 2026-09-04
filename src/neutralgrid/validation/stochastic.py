"""
Stochastic Regime Checker for AFML-Compliant Validation.

Implements statistical tests for regime quality:
- Survival probability: Monte Carlo simulation of range containment
- Hurst exponent: Ensemble of R/S, DFA, and Variance Ratio estimators
  with bias correction, bootstrap significance, geometric lag spacing,
  overlapping windows, and WLS regression
- OU half-life: Mean-reversion speed from Ornstein-Uhlenbeck fit

Design principles:
- Deterministic with explicit random_state for reproducibility
- Conservative estimates (favor false negatives over false positives)
- Clear thresholds from AFML research

Hurst quality improvements:
- Compute on log-returns instead of raw log prices (removes trend bias)
- DFA-1 estimator alongside R/S (more stable under trends)
- Variance Ratio estimator for independent cross-check
- Geometric lag spacing (equal octave representation in log-log fit)
- Overlapping sub-windows via sliding_window_view (variance reduction)
- Weighted Least Squares regression (weight by sqrt(n_windows))
- Finite-sample R/S bias correction via shuffled null
- Shuffled-bootstrap significance test (H != 0.5)
- Multi-estimator ensemble (median of R/S, DFA, VR)
- Stability and confidence diagnostics
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from neutralgrid.core.constants import BOT_HORIZON_BARS_15M


@dataclass(frozen=True)
class StochasticConfig:
    """Configuration for stochastic regime checks."""

    # Survival probability settings
    survival_horizon: int = BOT_HORIZON_BARS_15M   # H* = 6h = 24 bars @ 15m
    mc_paths: int = 10000                           # Monte Carlo simulation paths

    # Hurst exponent threshold
    hurst_max: float = 0.55          # Above = trending (reject)

    # OU half-life bounds (in bars)
    ou_halflife_min: int = 4         # Below = too fast, unstable
    ou_halflife_max: int = 48        # Above = too slow, not mean-reverting

    # Random state for reproducibility
    random_state: int = 42

    # --- Hurst quality settings ---
    hurst_num_lags: int = 30                # Geometric lag spacing points
    hurst_bootstrap_n: int = 200            # Shuffle bootstrap samples (0 = disabled)
    hurst_use_returns: bool = True          # Compute R/S on returns (not raw log prices)
    hurst_use_overlapping: bool = True      # Overlapping sub-windows for R/S
    hurst_bias_correct: bool = True         # Finite-sample R/S bias correction
    hurst_significance_alpha: float = 0.10  # Significance level for bootstrap test

    def __post_init__(self):
        if self.survival_horizon <= 0:
            raise ValueError("survival_horizon must be > 0")
        if self.mc_paths <= 0:
            raise ValueError("mc_paths must be > 0")
        if not 0 < self.hurst_max < 1:
            raise ValueError("hurst_max must be in (0, 1)")
        if self.hurst_num_lags < 3:
            raise ValueError("hurst_num_lags must be >= 3")
        if self.hurst_bootstrap_n < 0:
            raise ValueError("hurst_bootstrap_n must be >= 0")
        if not 0 < self.hurst_significance_alpha < 1:
            raise ValueError("hurst_significance_alpha must be in (0, 1)")


@dataclass
class StochasticResult:
    """Results from stochastic regime analysis."""

    # OU process parameters
    ou_theta: float              # Mean-reversion speed
    ou_mu: float                 # Long-term mean (log-price)
    ou_sigma: float              # Volatility
    ou_halflife: float           # Half-life in bars

    # Hurst exponent (ensemble estimate)
    hurst_exponent: float

    # Survival probability
    survival_prob: float

    # Pass/fail checks. survival_ok removed 2026-05-22 (E4): the survival gate
    # was inert -- survival_min was always 0.0 so survival_prob >= 0.0 always
    # held. MC containment is owned by Stage B micro_osc Gate 4. survival_prob
    # is still computed and reported as a metric above.
    hurst_ok: bool
    halflife_ok: bool

    # Hurst quality diagnostics (additive, backward-compatible)
    hurst_significant: Optional[bool] = None    # H significantly != 0.5
    hurst_stability: Optional[float] = None     # Inter-estimator spread (lower=better)
    hurst_stderr: Optional[float] = None        # Bootstrap standard error

    @property
    def all_passed(self) -> bool:
        return self.hurst_ok and self.halflife_ok


class StochasticRegimeChecker:
    """
    Stochastic analysis of price regime for grid bot suitability.

    Uses Ornstein-Uhlenbeck process modeling and Hurst exponent analysis
    to assess whether the current regime is suitable for grid trading.

    Key checks:
    1. Hurst exponent <= 0.55: Not in a trending regime
    2. OU half-life in [4, 48] bars: Reasonable mean-reversion speed
    3. Survival probability is computed for downstream containment analysis

    Hurst estimation uses an ensemble of three methods:
    - R/S (Rescaled Range) with overlapping windows, geometric lags, WLS,
      and finite-sample bias correction
    - DFA-1 (Detrended Fluctuation Analysis) for trend robustness
    - Variance Ratio for independent cross-validation
    The final estimate is the median of available estimators.

    Usage:
        checker = StochasticRegimeChecker(StochasticConfig())
        result = checker.analyze(
            log_prices=np.log(prices),
            range_high=np.log(high),
            range_low=np.log(low),
        )
        if result.all_passed:
            print("Regime suitable for grid trading")
    """

    def __init__(self, config: Optional[StochasticConfig] = None):
        """
        Initialize StochasticRegimeChecker.

        Args:
            config: StochasticConfig with thresholds and simulation params
        """
        self.config = config or StochasticConfig()
        self._rng = np.random.default_rng(self.config.random_state)

    def estimate_ou_params(
        self,
        log_prices: np.ndarray,
        dt: float = 1.0,
    ) -> Tuple[float, float, float]:
        """
        Estimate Ornstein-Uhlenbeck process parameters via OLS regression.

        The OU process: dX = theta(mu - X)dt + sigma*dW

        We estimate using the discrete approximation:
        X[t+1] - X[t] = theta*(mu - X[t])*dt + sigma*sqrt(dt)*epsilon

        Which rearranges to:
        dX = a + b*X[t] where a = theta*mu*dt, b = -theta*dt

        Args:
            log_prices: Log prices array
            dt: Time step (default 1 bar)

        Returns:
            Tuple of (theta, mu, sigma)
        """
        if len(log_prices) < 10:
            return (0.0, 0.0, 0.0)

        # Compute returns
        x = log_prices[:-1]
        dx = np.diff(log_prices)

        # OLS regression: dx = a + b*x + residual
        # Using normal equations
        n = len(x)
        x_mean = np.mean(x)
        dx_mean = np.mean(dx)

        # Compute covariance and variance
        cov_x_dx = np.sum((x - x_mean) * (dx - dx_mean)) / n
        var_x = np.sum((x - x_mean) ** 2) / n

        if var_x < 1e-12:
            return (0.0, float(np.mean(log_prices)), 0.0)

        # OLS coefficients
        b = cov_x_dx / var_x
        a = dx_mean - b * x_mean

        # Extract OU parameters
        # b = -theta*dt => theta = -b/dt
        # a = theta*mu*dt => mu = a / (theta*dt) = -a/b (when b != 0)
        theta = -b / dt

        if abs(b) < 1e-12:
            mu = np.mean(log_prices)
        else:
            mu = -a / b

        # Estimate sigma from residual variance
        residuals = dx - (a + b * x)
        sigma_squared = np.var(residuals, ddof=1) / dt if len(residuals) > 1 else 0.0
        sigma = np.sqrt(max(0, sigma_squared))

        # Ensure theta is positive (mean-reverting)
        theta = max(0, theta)

        return (float(theta), float(mu), float(sigma))

    def compute_ou_halflife(self, theta: float) -> float:
        """
        Compute OU process half-life.

        Half-life is the time for the process to revert halfway to the mean.
        tau = ln(2) / theta

        Args:
            theta: Mean-reversion speed parameter

        Returns:
            Half-life in same units as dt (bars)
        """
        if theta <= 1e-12:
            return float("inf")
        return float(np.log(2) / theta)

    def estimate_ou_params_detrended(
        self,
        log_prices: np.ndarray,
        dt: float = 1.0,
    ) -> Tuple[float, float, float]:
        """
        OU parameters fitted on linearly de-trended log prices (ERR-092).

        The raw OLS fit regresses dx on the price LEVEL; a deterministic drift
        c*t contributes ~0 to cov(x, dx) but inflates var(x) by c^2*T^2/12, so
        the slope is attenuated toward 0 and the half-life inflates toward inf
        for any drifting series — trend masquerades as "no mean-reversion".
        Modeling X_t = alpha + c*t + Z_t and fitting the OU on the residual Z_t
        measures the reversion speed of the oscillation component around the
        trend, which is the quantity the half-life gate intends to test.
        (The Hurst path already de-trends by working on returns/DFA; this is
        the equivalent fix for the OU path.)

        Returns:
            Tuple of (theta, mu, sigma) for the de-trended component; mu is
            expressed at the level of the trend line's endpoint so it remains
            comparable to the last log price.
        """
        n = len(log_prices)
        if n < 10:
            return (0.0, 0.0, 0.0)

        t = np.arange(n, dtype=float)
        slope, intercept = np.polyfit(t, log_prices, deg=1)
        residuals = log_prices - (intercept + slope * t)

        theta, mu_z, sigma = self.estimate_ou_params(residuals, dt=dt)
        # Anchor mu at the trend endpoint so it lives on the log-price scale.
        mu = float(intercept + slope * (n - 1) + mu_z)
        return (float(theta), mu, float(sigma))

    # =========================================================================
    # Hurst Estimators
    # =========================================================================

    def _rs_hurst(
        self,
        series: np.ndarray,
        min_lag: int,
        max_lag: int,
        num_lags: int,
        overlapping: bool,
        use_wls: bool,
    ) -> float:
        """
        Compute Hurst exponent via improved R/S (Rescaled Range) analysis.

        Improvements over naive R/S:
        - Geometric lag spacing: equal representation per octave in log-log fit
        - Overlapping sub-windows: reduces variance via sliding_window_view
        - WLS regression: weights by sqrt(n_windows) for reliability

        Args:
            series: Input time series (returns or log prices)
            min_lag: Minimum window size (>= 4)
            max_lag: Maximum window size
            num_lags: Number of geometrically-spaced lag points
            overlapping: Use overlapping windows (True) or non-overlapping (False)
            use_wls: Use weighted least squares (True) or OLS (False)

        Returns:
            Hurst exponent in [0, 1], or 0.5 if insufficient data
        """
        n = len(series)
        if n < 20:
            return 0.5

        min_lag = max(min_lag, 4)
        max_lag = min(max_lag, n // 2)
        if max_lag <= min_lag:
            return 0.5

        # Geometric lag spacing (#7)
        lags = np.unique(np.geomspace(min_lag, max_lag, num=num_lags).astype(int))

        lag_vals = []
        rs_means = []
        weights = []

        for lag in lags:
            lag = int(lag)
            if lag < 4 or lag > n:
                continue

            if overlapping and n >= lag:
                # Overlapping windows via sliding_window_view (#8)
                windows = np.lib.stride_tricks.sliding_window_view(series, lag)
            else:
                num_win = n // lag
                if num_win < 2:
                    continue
                windows = series[:num_win * lag].reshape(num_win, lag)

            if len(windows) < 1:
                continue

            # Vectorized R/S across all windows
            means = windows.mean(axis=1, keepdims=True)
            deviations = windows - means
            cumsums = np.cumsum(deviations, axis=1)
            R = cumsums.max(axis=1) - cumsums.min(axis=1)
            S = windows.std(axis=1, ddof=1)

            valid_mask = (S > 1e-12) & np.isfinite(S)
            if not valid_mask.any():
                continue

            rs_vals = R[valid_mask] / S[valid_mask]
            finite_mask = np.isfinite(rs_vals) & (rs_vals > 0)
            rs_vals = rs_vals[finite_mask]

            if len(rs_vals) > 0:
                lag_vals.append(lag)
                rs_means.append(float(np.mean(rs_vals)))
                weights.append(np.sqrt(len(rs_vals)))

        if len(lag_vals) < 3:
            return 0.5

        log_lags = np.log(np.array(lag_vals, dtype=float))
        log_rs = np.log(np.array(rs_means, dtype=float))
        w = np.array(weights, dtype=float)

        valid = np.isfinite(log_lags) & np.isfinite(log_rs)
        log_lags, log_rs, w = log_lags[valid], log_rs[valid], w[valid]

        if len(log_lags) < 3:
            return 0.5

        # WLS regression (#10)
        if use_wls and w.sum() > 0:
            W = w / w.sum()
            mean_x = np.sum(W * log_lags)
            mean_y = np.sum(W * log_rs)
            cov = np.sum(W * (log_lags - mean_x) * (log_rs - mean_y))
            var = np.sum(W * (log_lags - mean_x) ** 2)
        else:
            n_pts = len(log_lags)
            mean_x = np.mean(log_lags)
            mean_y = np.mean(log_rs)
            cov = np.sum((log_lags - mean_x) * (log_rs - mean_y)) / n_pts
            var = np.sum((log_lags - mean_x) ** 2) / n_pts

        if var < 1e-12:
            return 0.5

        H = cov / var
        return float(max(0.0, min(1.0, H)))

    def _dfa_hurst(
        self,
        series: np.ndarray,
        min_scale: int = 4,
        max_scale: Optional[int] = None,
        num_scales: int = 30,
    ) -> Optional[float]:
        """
        Compute Hurst exponent via Detrended Fluctuation Analysis (DFA-1).

        More robust under trends than R/S because local polynomial
        detrending removes trend contamination within each segment.

        Steps:
        1. Compute profile: cumulative sum of mean-centered series
        2. For each scale s: divide into segments, fit linear trend,
           compute RMS of detrended residuals
        3. H = slope of log(F(s)) vs log(s)

        Args:
            series: Input time series (typically returns)
            min_scale: Minimum segment size
            max_scale: Maximum segment size (default: len/4)
            num_scales: Number of geometrically-spaced scale points

        Returns:
            Hurst exponent in [0, 1], or None if insufficient data
        """
        n = len(series)
        if n < 20:
            return None

        # Build profile (cumulative sum of mean-centered series)
        profile = np.cumsum(series - np.mean(series))

        if max_scale is None:
            max_scale = n // 4
        max_scale = min(max_scale, n // 4)
        min_scale = max(min_scale, 4)

        if max_scale <= min_scale:
            return None

        scales = np.unique(np.geomspace(min_scale, max_scale, num=num_scales).astype(int))

        scale_vals = []
        fluct_vals = []

        for s in scales:
            s = int(s)
            n_seg = n // s
            if n_seg < 1:
                continue

            variances = []

            # Forward segments
            for v in range(n_seg):
                seg = profile[v * s:(v + 1) * s]
                if len(seg) < s:
                    continue
                x_ax = np.arange(s, dtype=float)
                # Linear detrend (DFA-1)
                coeffs = np.polyfit(x_ax, seg, 1)
                trend = coeffs[0] * x_ax + coeffs[1]
                resid = seg - trend
                variances.append(np.mean(resid ** 2))

            # Backward segments (use remaining data from the end)
            for v in range(n_seg):
                start = n - (v + 1) * s
                end = n - v * s
                if start < 0:
                    continue
                seg = profile[start:end]
                if len(seg) < s:
                    continue
                x_ax = np.arange(len(seg), dtype=float)
                coeffs = np.polyfit(x_ax, seg, 1)
                trend = coeffs[0] * x_ax + coeffs[1]
                resid = seg - trend
                variances.append(np.mean(resid ** 2))

            if variances:
                F = np.sqrt(np.mean(variances))
                if np.isfinite(F) and F > 0:
                    scale_vals.append(s)
                    fluct_vals.append(F)

        if len(scale_vals) < 3:
            return None

        log_s = np.log(np.array(scale_vals, dtype=float))
        log_f = np.log(np.array(fluct_vals, dtype=float))

        valid = np.isfinite(log_s) & np.isfinite(log_f)
        log_s, log_f = log_s[valid], log_f[valid]

        if len(log_s) < 3:
            return None

        n_pts = len(log_s)
        mean_x = np.mean(log_s)
        mean_y = np.mean(log_f)
        cov = np.sum((log_s - mean_x) * (log_f - mean_y)) / n_pts
        var = np.sum((log_s - mean_x) ** 2) / n_pts

        if var < 1e-12:
            return None

        H = cov / var
        return float(max(0.0, min(1.0, H)))

    def _variance_ratio_hurst(self, returns: np.ndarray) -> Optional[float]:
        """
        Compute Hurst exponent via Variance Ratio method.

        For fractional Brownian motion:
            VR(q) = Var(q-return) / (q * Var(1-return)) = q^(2H-1)
        So slope of log(VR) vs log(q) = 2H-1, giving H = (slope+1)/2.

        Uses overlapping q-period returns for efficiency.

        Args:
            returns: Log-returns array

        Returns:
            Hurst exponent in [0, 1], or None if insufficient data
        """
        n = len(returns)
        if n < 30:
            return None

        var_1 = np.var(returns, ddof=1)
        if var_1 < 1e-15:
            return None

        max_q = min(n // 4, 50)
        if max_q < 4:
            return None
        lags = np.unique(np.geomspace(2, max_q, num=15).astype(int))

        log_q_list = []
        log_vr_list = []

        # Precompute cumulative returns once (used for overlapping q-returns)
        cumret = np.concatenate(([0.0], np.cumsum(returns)))

        for q in lags:
            q = int(q)
            if q < 2 or q >= n:
                continue
            # Overlapping q-period returns via cumsum for efficiency
            q_rets = cumret[q:] - cumret[:-q]
            if len(q_rets) < 2:
                continue
            var_q = np.var(q_rets, ddof=1)
            vr = var_q / (q * var_1)
            if vr > 0 and np.isfinite(vr):
                log_q_list.append(np.log(q))
                log_vr_list.append(np.log(vr))

        if len(log_q_list) < 3:
            return None

        log_q = np.array(log_q_list)
        log_vr = np.array(log_vr_list)

        n_pts = len(log_q)
        mean_x = np.mean(log_q)
        mean_y = np.mean(log_vr)
        cov = np.sum((log_q - mean_x) * (log_vr - mean_y)) / n_pts
        var = np.sum((log_q - mean_x) ** 2) / n_pts

        if var < 1e-12:
            return None

        slope = cov / var  # = 2H - 1
        H = (slope + 1.0) / 2.0
        return float(max(0.0, min(1.0, H)))

    # =========================================================================
    # Full Hurst estimation pipeline
    # =========================================================================

    def _compute_hurst_full(
        self,
        log_prices: np.ndarray,
        min_lag: int = 4,
        max_lag: Optional[int] = None,
    ) -> Tuple[float, Optional[bool], Optional[float], Optional[float]]:
        """
        Full Hurst estimation with ensemble, bias correction, and bootstrap.

        Pipeline:
        1. Compute R/S on returns (with geometric lags, overlapping, WLS)
        2. Bootstrap R/S null distribution (for bias correction + significance)
        3. Apply finite-sample bias correction to R/S
        4. Compute DFA-1 on returns
        5. Compute Variance Ratio on returns
        6. Ensemble: median of available estimators
        7. Test bias-corrected R/S against R/S bootstrap null

        Returns:
            Tuple of (hurst, significant, stability, stderr)
            - hurst: Ensemble Hurst exponent estimate
            - significant: Whether H is significantly != 0.5 (or None)
            - stability: Inter-estimator spread (lower=better, or None)
            - stderr: Bootstrap standard error (or None)
        """
        cfg = self.config

        # Guard: strip NaN/Inf from input
        finite_mask = np.isfinite(log_prices)
        if not np.all(finite_mask):
            log_prices = log_prices[finite_mask]

        n = len(log_prices)
        if n < 20:
            return (0.5, None, None, None)

        returns = np.diff(log_prices)
        if len(returns) < 20:
            return (0.5, None, None, None)

        # Fix #1: Select base series for R/S
        base_series = returns if cfg.hurst_use_returns else log_prices
        base_n = len(base_series)

        eff_max_lag = max_lag if max_lag is not None else base_n // 4
        eff_max_lag = min(eff_max_lag, base_n // 2)
        eff_min_lag = max(min_lag, 4)

        if eff_max_lag <= eff_min_lag:
            return (0.5, None, None, None)

        _rs_min_lag = eff_min_lag
        _rs_max_lag = eff_max_lag
        _rs_num_lags = cfg.hurst_num_lags
        _rs_overlapping = bool(cfg.hurst_use_overlapping)
        _rs_use_wls = True

        # === R/S (raw) ===
        h_rs_raw = self._rs_hurst(
            series=base_series,
            min_lag=_rs_min_lag, max_lag=_rs_max_lag,
            num_lags=_rs_num_lags, overlapping=_rs_overlapping,
            use_wls=_rs_use_wls,
        )

        # === Bootstrap shuffles (for bias correction + significance) ===
        boot_arr = None
        n_boot = 0
        if cfg.hurst_bootstrap_n > 0:
            n_boot = cfg.hurst_bootstrap_n
        elif cfg.hurst_bias_correct:
            n_boot = 10  # Quick shuffles just for bias correction

        if n_boot > 0 and len(returns) >= 20:
            boot_h = np.empty(n_boot)
            for i in range(n_boot):
                shuf_ret = returns.copy()
                self._rng.shuffle(shuf_ret)
                if cfg.hurst_use_returns:
                    shuf_series = shuf_ret
                else:
                    # Reconstruct log prices from shuffled returns
                    shuf_series = np.concatenate(
                        [[log_prices[0]], log_prices[0] + np.cumsum(shuf_ret)]
                    )
                boot_h[i] = self._rs_hurst(
                    series=shuf_series,
                    min_lag=_rs_min_lag, max_lag=_rs_max_lag,
                    num_lags=_rs_num_lags, overlapping=_rs_overlapping,
                    use_wls=_rs_use_wls,
                )
            boot_arr = boot_h

        # === Bias correction (#5): subtract finite-sample R/S bias ===
        h_rs = h_rs_raw
        if cfg.hurst_bias_correct and boot_arr is not None:
            null_mean = float(np.mean(boot_arr))
            h_rs = float(max(0.0, min(1.0, h_rs_raw - (null_mean - 0.5))))

        # === DFA-1 (#2): always on returns ===
        h_dfa = self._dfa_hurst(
            series=returns,
            min_scale=eff_min_lag,
            max_scale=eff_max_lag,
            num_scales=cfg.hurst_num_lags,
        )

        # === Variance Ratio (#9): always on returns ===
        h_vr = self._variance_ratio_hurst(returns=returns)

        # === Ensemble (#9): median of available estimators ===
        estimates = [h_rs]
        if h_dfa is not None:
            estimates.append(h_dfa)
        if h_vr is not None:
            estimates.append(h_vr)

        hurst = float(np.median(estimates))
        if not np.isfinite(hurst):
            hurst = 0.5  # Default to random walk if estimation fails

        # === Stability: inter-estimator spread (#4) ===
        stability = None
        if len(estimates) >= 2:
            stability = round(float(np.max(estimates) - np.min(estimates)), 4)

        # === Bootstrap significance test (#6) ===
        # Test the RAW R/S estimate against the R/S-only bootstrap null.
        # Using h_rs_raw (not bias-corrected h_rs) so estimate and null
        # distribution are on the same scale — mixing bias-corrected
        # estimate with raw null produces an invalid test.
        significant = None
        stderr = None
        if boot_arr is not None and cfg.hurst_bootstrap_n > 0:
            stderr = round(float(np.std(boot_arr)), 4)
            alpha = cfg.hurst_significance_alpha
            lo = np.percentile(boot_arr, 100 * alpha / 2)
            hi = np.percentile(boot_arr, 100 * (1 - alpha / 2))
            significant = bool(h_rs_raw < lo or h_rs_raw > hi)

        return (hurst, significant, stability, stderr)

    def compute_hurst_exponent(
        self,
        log_prices: np.ndarray,
        min_lag: int = 4,
        max_lag: Optional[int] = None,
    ) -> float:
        """
        Compute Hurst exponent using ensemble estimation.

        Backward-compatible wrapper around _compute_hurst_full().
        Returns only the scalar Hurst exponent. Use analyze() for
        the full result with diagnostics.

        H < 0.5: Mean-reverting (anti-persistent)
        H = 0.5: Random walk (Brownian motion)
        H > 0.5: Trending (persistent)

        For grid bots, we want H <= 0.55 (not trending).

        Args:
            log_prices: Log prices array
            min_lag: Minimum lag for analysis (>=4)
            max_lag: Maximum lag (default: len/4)

        Returns:
            Hurst exponent in [0, 1]
        """
        hurst, _, _, _ = self._compute_hurst_full(log_prices, min_lag, max_lag)
        return hurst

    def compute_survival_probability(
        self,
        ou_theta: float,
        ou_mu: float,
        ou_sigma: float,
        current_log_price: float,
        range_high_log: float,
        range_low_log: float,
        horizon: Optional[int] = None,
        n_paths: Optional[int] = None,
    ) -> float:
        """
        Monte Carlo simulation of survival probability within range.

        Simulates OU process paths and computes fraction that stay
        within [range_low, range_high] over the horizon.

        Args:
            ou_theta: Mean-reversion speed
            ou_mu: Long-term mean
            ou_sigma: Volatility
            current_log_price: Starting log price
            range_high_log: Log of range upper bound
            range_low_log: Log of range lower bound
            horizon: Number of steps (default from config)
            n_paths: Number of MC paths (default from config)

        Returns:
            Survival probability in [0, 1]
        """
        if horizon is None:
            horizon = self.config.survival_horizon
        if n_paths is None:
            n_paths = self.config.mc_paths

        # Handle degenerate cases
        if ou_sigma < 1e-12 or ou_theta < 1e-12:
            # No randomness or no mean-reversion
            # Just check if current price is in range
            return 1.0 if range_low_log <= current_log_price <= range_high_log else 0.0

        # Simulate OU paths using Euler-Maruyama
        dt = 1.0  # One bar
        sqrt_dt = np.sqrt(dt)

        # Initialize paths at current price
        paths = np.full(n_paths, current_log_price)
        survived = np.ones(n_paths, dtype=bool)

        for _ in range(horizon):
            # OU SDE: dX = theta*(mu - X)*dt + sigma*dW
            dW = self._rng.standard_normal(n_paths) * sqrt_dt
            drift = ou_theta * (ou_mu - paths) * dt
            diffusion = ou_sigma * dW

            paths = paths + drift + diffusion

            # Check if paths breach boundaries
            breached = (paths < range_low_log) | (paths > range_high_log)
            survived = survived & ~breached

            # Early termination if all paths have breached
            if not np.any(survived):
                return 0.0

        return float(np.mean(survived))

    def analyze(
        self,
        log_prices: np.ndarray,
        range_high_log: float,
        range_low_log: float,
    ) -> StochasticResult:
        """
        Full stochastic regime analysis.

        Args:
            log_prices: Array of log prices
            range_high_log: Log of range upper bound
            range_low_log: Log of range lower bound

        Returns:
            StochasticResult with all metrics and pass/fail checks
        """
        # Estimate OU parameters. ERR-092: the REPORTED theta/mu/sigma and the
        # half-life come from the de-trended fit (drift no longer masquerades
        # as "no mean-reversion"; measured on the 2026-07-11 cohort the raw
        # estimator put 74% of symbols above the window purely from drift
        # attenuation). The survival MC keeps the RAW fit so its containment
        # semantics — including the theta~0 degenerate branch for genuinely
        # drifting series — are unchanged.
        theta_raw, mu_raw, sigma_raw = self.estimate_ou_params(log_prices)
        theta, mu, sigma = self.estimate_ou_params_detrended(log_prices)

        # Compute half-life (de-trended reversion speed)
        halflife = self.compute_ou_halflife(theta)

        # Compute Hurst exponent with full quality suite
        (hurst, hurst_significant, hurst_stability, hurst_stderr) = (
            self._compute_hurst_full(log_prices)
        )

        # Compute survival probability (raw fit, unchanged semantics)
        current = log_prices[-1] if len(log_prices) > 0 else mu_raw
        survival = self.compute_survival_probability(
            ou_theta=theta_raw,
            ou_mu=mu_raw,
            ou_sigma=sigma_raw,
            current_log_price=current,
            range_high_log=range_high_log,
            range_low_log=range_low_log,
        )

        # Check thresholds
        hurst_ok = hurst <= self.config.hurst_max
        halflife_ok = (
            self.config.ou_halflife_min <= halflife <= self.config.ou_halflife_max
            if np.isfinite(halflife)
            else False
        )

        return StochasticResult(
            ou_theta=theta,
            ou_mu=mu,
            ou_sigma=sigma,
            ou_halflife=halflife,
            hurst_exponent=hurst,
            survival_prob=survival,
            hurst_ok=hurst_ok,
            halflife_ok=halflife_ok,
            hurst_significant=hurst_significant,
            hurst_stability=hurst_stability,
            hurst_stderr=hurst_stderr,
        )
