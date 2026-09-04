"""
Bayesian optimization for joint threshold calibration via GP-UCB.

Finds optimal threshold vector theta that maximizes:
    objective = SR_deflated * sqrt(N_deployed)

Uses a 7D search space with sensitivity-driven dimension reduction:
parameters contributing < 5% variance are fixed at defaults.

Pipeline:
  1. Sensitivity analysis (5 samples per param)
  2. Latin Hypercube initialization (20 points)
  3. GP-UCB optimization (55 iterations)
  4. Walk-forward validation with retention gate

No Binance API calls -- works only on cached/pre-computed data.
"""
from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OptimizationConfig:
    """Configuration for Bayesian threshold optimization."""
    n_initial_samples: int = 20  # Latin Hypercube for initialization
    n_optimization_steps: int = 55  # GP-UCB iterations
    n_sensitivity_samples: int = 5  # Per-parameter values for sensitivity
    sensitivity_threshold: float = 0.05  # Fix params below 5% variance
    kappa: float = 2.0  # UCB exploration parameter
    staleness_days: int = 14
    checkpoint_path: str = "data/cache/bayesopt_checkpoint_v20260311.pkl"
    retention_min: float = 0.70  # Walk-forward retention gate


@dataclass(frozen=True)
class ThresholdSearchSpace:
    """7D parameter bounds for threshold optimization."""
    range_prob_min: Tuple[float, float] = (0.15, 0.50)
    min_tos: Tuple[float, float] = (20.0, 60.0)
    min_position_fraction: Tuple[float, float] = (0.02, 0.15)
    meta_hurdle_pct: Tuple[float, float] = (1.0, 6.0)
    gb_max_depth: Tuple[int, int] = (2, 4)  # Discrete
    conformal_alpha: Tuple[float, float] = (0.10, 0.35)
    pnl_threshold_quantile: Tuple[float, float] = (0.50, 0.90)


@dataclass(frozen=True)
class OptimizationResult:
    """Immutable result of Bayesian optimization."""
    best_theta: Dict[str, float]
    best_objective: float
    n_evaluations: int
    retention_ratio: float
    passed_retention_gate: bool
    sensitivity_report: Dict[str, float]
    convergence_history: List[float]
    computed_at_utc: str


class BayesianThresholdOptimizer:
    """GP-UCB Bayesian optimizer for joint threshold calibration.

    Usage::

        opt = BayesianThresholdOptimizer()
        result = opt.optimize(evaluate_fn, default_theta)
        if result.passed_retention_gate:
            # Deploy optimized thresholds
    """

    def __init__(
        self,
        config: Optional[OptimizationConfig] = None,
        search_space: Optional[ThresholdSearchSpace] = None,
    ) -> None:
        self._cfg = config or OptimizationConfig()
        self._space = search_space or ThresholdSearchSpace()
        self._rng_counter: int = 0

    @property
    def search_space(self) -> Dict[str, Tuple[float, float]]:
        """Public accessor for the parameter search bounds."""
        return self._get_bounds()

    def _get_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Extract parameter bounds from search space."""
        return {
            "range_prob_min": self._space.range_prob_min,
            "min_tos": self._space.min_tos,
            "min_position_fraction": self._space.min_position_fraction,
            "meta_hurdle_pct": self._space.meta_hurdle_pct,
            "gb_max_depth": (float(self._space.gb_max_depth[0]),
                             float(self._space.gb_max_depth[1])),
            "conformal_alpha": self._space.conformal_alpha,
            "pnl_threshold_quantile": self._space.pnl_threshold_quantile,
        }

    def _latin_hypercube_sample(self, n: int) -> List[Dict[str, float]]:
        """Generate n points via Latin Hypercube Sampling.

        Parameters
        ----------
        n : int
            Number of sample points.

        Returns
        -------
        List of dicts, each mapping parameter name to value.
        """
        bounds = self._get_bounds()
        param_names = list(bounds.keys())
        d = len(param_names)

        rng = np.random.RandomState(42)

        # LHS: stratified sampling
        samples: List[Dict[str, float]] = []
        for i in range(d):
            lo, hi = bounds[param_names[i]]
            # Create stratified intervals
            intervals = np.linspace(lo, hi, n + 1)
            # Random point within each interval
            points = np.array([
                rng.uniform(intervals[j], intervals[j + 1])
                for j in range(n)
            ])
            rng.shuffle(points)
            if i == 0:
                for p in points:
                    samples.append({param_names[i]: float(p)})
            else:
                for j, p in enumerate(points):
                    samples[j][param_names[i]] = float(p)

        # Handle discrete params
        for s in samples:
            s["gb_max_depth"] = float(round(s["gb_max_depth"]))

        return samples

    def _compute_objective(
        self,
        theta: Dict[str, float],
        backtest_results: List[Dict[str, Any]],
    ) -> float:
        """Compute optimization objective: SR_deflated * sqrt(N_deployed).

        Parameters
        ----------
        theta : dict
            Current threshold vector.
        backtest_results : list
            Pre-computed backtest results with pnl, labels, etc.

        Returns
        -------
        float
            Objective value. Returns -inf if no deployable results.
        """
        if not backtest_results:
            return float("-inf")

        # Filter results by theta thresholds
        deployed_pnls: List[float] = []
        for r in backtest_results:
            # Apply threshold gates
            range_prob = r.get("range_prob", 0.0)
            tos = r.get("tos", 0.0)
            pos_frac = r.get("position_size_fraction", 0.0)
            pnl = r.get("net_pnl_pct", 0.0)

            passes = (
                range_prob >= theta.get("range_prob_min", 0.25)
                and tos >= theta.get("min_tos", 40.0)
                and pos_frac >= theta.get("min_position_fraction", 0.05)
            )
            if passes:
                deployed_pnls.append(float(pnl))

        n_deployed = len(deployed_pnls)
        if n_deployed < 2:
            return float("-inf")

        pnl_arr = np.array(deployed_pnls)
        mean_pnl = float(np.mean(pnl_arr))
        std_pnl = float(np.std(pnl_arr, ddof=1))

        if std_pnl < 1e-10:
            sr = 0.0
        else:
            sr = mean_pnl / std_pnl

        # Simple deflation: SR_deflated = SR * (1 - 1/N)
        sr_deflated = sr * (1.0 - 1.0 / n_deployed)

        objective = sr_deflated * math.sqrt(n_deployed)
        return float(objective)

    def run_sensitivity_analysis(
        self,
        evaluate_fn: Callable[[Dict[str, float]], float],
        default_theta: Dict[str, float],
    ) -> Dict[str, float]:
        """Run one-at-a-time sensitivity analysis.

        Parameters
        ----------
        evaluate_fn : callable
            Takes theta dict -> returns objective value.
        default_theta : dict
            Default parameter values.

        Returns
        -------
        dict
            Parameter name -> variance fraction (sums to ~1.0).
        """
        bounds = self._get_bounds()
        n_samples = self._cfg.n_sensitivity_samples

        param_variances: Dict[str, float] = {}
        _ = evaluate_fn(default_theta)  # Warm-up / verify callable

        for param, (lo, hi) in bounds.items():
            values = np.linspace(lo, hi, n_samples)
            if param == "gb_max_depth":
                values = np.unique(np.round(values).astype(int)).astype(float)

            objectives: List[float] = []
            for v in values:
                theta_v = dict(default_theta)
                theta_v[param] = float(v)
                obj = evaluate_fn(theta_v)
                objectives.append(obj)

            # Filter out -inf
            finite_obj = [o for o in objectives if np.isfinite(o)]
            if len(finite_obj) >= 2:
                param_variances[param] = float(np.var(finite_obj))
            else:
                param_variances[param] = 0.0

        # Normalize to fractions
        total_var = sum(param_variances.values())
        if total_var > 0:
            sensitivity = {k: v / total_var for k, v in param_variances.items()}
        else:
            n = len(param_variances)
            sensitivity = {k: 1.0 / n for k in param_variances}

        logger.info("Sensitivity analysis: %s",
                    {k: f"{v:.4f}" for k, v in sensitivity.items()})

        return sensitivity

    def optimize(
        self,
        evaluate_fn: Callable[[Dict[str, float]], float],
        default_theta: Dict[str, float],
    ) -> OptimizationResult:
        """Run full Bayesian optimization pipeline.

        Parameters
        ----------
        evaluate_fn : callable
            Takes theta dict -> returns objective value.
        default_theta : dict
            Default parameter values.

        Returns
        -------
        OptimizationResult
        """
        cfg = self._cfg

        # Step 1: Sensitivity analysis
        logger.info("BayesOpt: running sensitivity analysis...")
        sensitivity = self.run_sensitivity_analysis(evaluate_fn, default_theta)

        # Fix low-sensitivity params
        active_params = {
            k for k, v in sensitivity.items()
            if v >= cfg.sensitivity_threshold
        }
        fixed_params = {
            k: default_theta.get(k, 0.0)
            for k in sensitivity
            if k not in active_params
        }

        if not active_params:
            active_params = set(sensitivity.keys())
            fixed_params = {}
            logger.warning("All params below sensitivity threshold — keeping all active")

        logger.info(
            "BayesOpt: %d active params: %s, %d fixed: %s",
            len(active_params), sorted(active_params),
            len(fixed_params), sorted(fixed_params),
        )

        # Step 2: LHS initialization on active params only
        logger.info("BayesOpt: LHS initialization (%d points)...", cfg.n_initial_samples)
        lhs_samples = self._latin_hypercube_sample(cfg.n_initial_samples)

        X_observed: List[List[float]] = []
        y_observed: List[float] = []
        convergence: List[float] = []
        best_obj = float("-inf")
        best_theta = dict(default_theta)

        active_list = sorted(active_params)

        for sample in lhs_samples:
            theta = dict(fixed_params)
            for k in active_list:
                theta[k] = sample.get(k, default_theta.get(k, 0.0))

            obj = evaluate_fn(theta)
            x_vec = [theta[k] for k in active_list]
            X_observed.append(x_vec)
            y_observed.append(obj)

            if np.isfinite(obj) and obj > best_obj:
                best_obj = obj
                best_theta = dict(theta)

            convergence.append(best_obj if np.isfinite(best_obj) else 0.0)

        # Step 3: GP-UCB loop
        _GaussianProcessRegressor: Any = None
        _Matern: Any = None
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor as _GPR
            from sklearn.gaussian_process.kernels import Matern as _M
            _GaussianProcessRegressor = _GPR
            _Matern = _M
            use_gp = True
        except ImportError:
            logger.warning("sklearn GP not available — falling back to random search")
            use_gp = False

        logger.info("BayesOpt: GP-UCB optimization (%d steps)...", cfg.n_optimization_steps)

        for step in range(cfg.n_optimization_steps):
            if use_gp and len(X_observed) >= 5:
                # Fit GP
                X_arr = np.array(X_observed)
                y_arr = np.array(y_observed)

                # Filter -inf values
                finite_mask = np.isfinite(y_arr)
                if finite_mask.sum() < 5:
                    # Not enough finite points for GP
                    next_theta = self._random_sample(active_list, fixed_params)
                else:
                    X_gp = X_arr[finite_mask]
                    y_gp = y_arr[finite_mask]

                    try:
                        gp = _GaussianProcessRegressor(
                            kernel=_Matern(nu=2.5),
                            alpha=1e-6,
                            normalize_y=True,
                            random_state=42 + step,
                        )
                        gp.fit(X_gp, y_gp)

                        # UCB acquisition: find maximum of mu + kappa * sigma
                        next_theta = self._maximize_ucb(
                            gp, active_list, fixed_params,
                            kappa=cfg.kappa, n_candidates=500, seed=step,
                        )
                    except Exception as e:
                        logger.warning("GP fit failed at step %d: %s", step, e)
                        next_theta = self._random_sample(
                            active_list, fixed_params
                        )
            else:
                next_theta = self._random_sample(active_list, fixed_params)

            obj = evaluate_fn(next_theta)
            x_vec = [next_theta[k] for k in active_list]
            X_observed.append(x_vec)
            y_observed.append(obj)

            if np.isfinite(obj) and obj > best_obj:
                best_obj = obj
                best_theta = dict(next_theta)

            convergence.append(best_obj if np.isfinite(best_obj) else 0.0)

            # Checkpoint periodically
            if (step + 1) % 10 == 0:
                self.save_checkpoint({
                    "X": X_observed, "y": y_observed,
                    "best_theta": best_theta, "best_obj": best_obj,
                    "step": step,
                })
                logger.info(
                    "BayesOpt step %d/%d: best_obj=%.4f",
                    step + 1, cfg.n_optimization_steps, best_obj,
                )

        now_utc = datetime.now(timezone.utc).isoformat()

        return OptimizationResult(
            best_theta=best_theta,
            best_objective=float(best_obj) if np.isfinite(best_obj) else 0.0,
            n_evaluations=len(y_observed),
            retention_ratio=0.0,  # Set by validate_walk_forward
            passed_retention_gate=False,
            sensitivity_report=sensitivity,
            convergence_history=convergence,
            computed_at_utc=now_utc,
        )

    def _random_sample(
        self,
        active_list: List[str],
        fixed_params: Dict[str, float],
    ) -> Dict[str, float]:
        """Generate a reproducible random sample for fallback."""
        bounds = self._get_bounds()
        rng = np.random.RandomState(42 + self._rng_counter)
        self._rng_counter += 1
        theta = dict(fixed_params)
        for k in active_list:
            lo, hi = bounds[k]
            val = float(rng.uniform(lo, hi))
            if k == "gb_max_depth":
                val = float(round(val))
            theta[k] = val
        return theta

    def _maximize_ucb(
        self,
        gp: Any,
        active_list: List[str],
        fixed_params: Dict[str, float],
        kappa: float = 2.0,
        n_candidates: int = 500,
        seed: int = 0,
    ) -> Dict[str, float]:
        """Find theta maximizing UCB = mu + kappa * sigma via random candidate sampling."""
        bounds = self._get_bounds()
        rng = np.random.RandomState(42 + seed)

        # Generate candidate points
        candidates = np.zeros((n_candidates, len(active_list)))
        for i, k in enumerate(active_list):
            lo, hi = bounds[k]
            candidates[:, i] = rng.uniform(lo, hi, n_candidates)
            if k == "gb_max_depth":
                candidates[:, i] = np.round(candidates[:, i])

        # Predict with GP
        mu, sigma = gp.predict(candidates, return_std=True)
        ucb = mu + kappa * sigma

        # Select best
        best_idx = int(np.argmax(ucb))
        theta = dict(fixed_params)
        for i, k in enumerate(active_list):
            theta[k] = float(candidates[best_idx, i])

        return theta

    def validate_walk_forward(
        self,
        theta: Dict[str, float],
        evaluate_fn_train: Callable[[Dict[str, float]], float],
        evaluate_fn_test: Callable[[Dict[str, float]], float],
    ) -> tuple[float, bool]:
        """Walk-forward validation of optimized thresholds.

        Parameters
        ----------
        theta : dict
            Optimized threshold vector.
        evaluate_fn_train : callable
            Objective evaluation on training period.
        evaluate_fn_test : callable
            Objective evaluation on test period.

        Returns
        -------
        (retention_ratio, passes_gate)
        """
        obj_train = evaluate_fn_train(theta)
        obj_test = evaluate_fn_test(theta)

        if not np.isfinite(obj_train) or abs(obj_train) < 1e-10:
            logger.warning("Walk-forward: train objective non-finite or zero")
            return 0.0, False

        retention = obj_test / obj_train if obj_train > 0 else 0.0
        passes = retention >= self._cfg.retention_min

        logger.info(
            "Walk-forward: train=%.4f, test=%.4f, retention=%.4f, passes=%s",
            obj_train, obj_test, retention, passes,
        )

        return float(retention), passes

    def save_checkpoint(self, state: Dict[str, Any]) -> None:
        """Save optimization state to checkpoint file."""
        path = Path(self._cfg.checkpoint_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(state, f)
        except Exception as e:
            logger.warning("Failed to save checkpoint: %s", e)

    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load optimization state from checkpoint file."""
        path = Path(self._cfg.checkpoint_path)
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning("Failed to load checkpoint: %s", e)
            return None

    def save_result(
        self,
        result: OptimizationResult,
        path: Optional[Path] = None,
    ) -> None:
        """Save optimization result as JSON."""
        p = path or Path("data/cache/bayesopt_result_v20260311.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(result)
        p.write_text(json.dumps(data, indent=2, default=str))
        logger.info("Optimization result saved to %s", p)
