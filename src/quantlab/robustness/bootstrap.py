"""
Bootstrap analysis for statistical significance
"""
from datetime import datetime
from typing import Optional, List, Tuple

import numpy as np
import polars as pl

from ..common.logging import get_logger
from ..common.perf import measure_time

logger = get_logger(__name__)


class BootstrapAnalysis:
    """
    Bootstrap analysis for statistical significance testing.
    """

    def __init__(self, spec: dict):
        """
        Initialize bootstrap analysis.

        Args:
            spec: Strategy specification
        """
        self.spec = spec
        self.n_bootstrap = spec.get("validation", {}).get("bootstrap_samples", 1000)
        self.block_size = spec.get("validation", {}).get("bootstrap_block_size", 5)  # 5-day blocks
        self.confidence_level = spec.get("validation", {}).get("bootstrap_confidence", 0.95)

    def run_analysis(
        self,
        df: pl.DataFrame,
        strategy_func,
        metric_name: str = "sharpe_ratio",
        n_bootstrap: Optional[int] = None,
    ) -> dict:
        """
        Run bootstrap analysis.

        Args:
            df: OHLCV dataframe
            strategy_func: Function that takes (df, params) and returns (signals, metrics)
            metric_name: Metric to bootstrap
            n_bootstrap: Number of bootstrap samples (overrides spec if provided)

        Returns:
            Dictionary with bootstrap results
        """
        if n_bootstrap is None:
            n_bootstrap = self.n_bootstrap

        logger.info("bootstrap_start", samples=n_bootstrap)

        # Calculate original metric
        original_params = self.spec.get("backtest", {}).get("params", {})
        _, original_metrics = strategy_func(df, original_params)
        original_metric = original_metrics.get(metric_name, 0)

        # Generate bootstrap samples
        bootstrap_metrics = []

        with measure_time("bootstrap_analysis"):
            for i in range(n_bootstrap):
                # Generate block bootstrap sample
                try:
                    bootstrap_df = self._block_bootstrap_sample(df)
                    
                    # Calculate metric on bootstrap sample
                    _, boot_metrics = strategy_func(bootstrap_df, original_params)
                    boot_metric = boot_metrics.get(metric_name, 0)
                    
                    bootstrap_metrics.append(boot_metric)
                except Exception as e:
                    logger.warning("bootstrap_sample_error", iteration=i, error=str(e))
                    bootstrap_metrics.append(float('nan'))

        # Remove NaN values for analysis
        bootstrap_metrics = [m for m in bootstrap_metrics if not np.isnan(m)]

        # Calculate bootstrap statistics
        results = self._calculate_bootstrap_stats(
            original_metric, bootstrap_metrics
        )

        logger.info("bootstrap_complete", samples=len(bootstrap_metrics))

        return {
            "original_metric": original_metric,
            "bootstrap_metrics": bootstrap_metrics,
            "statistics": results,
            "confidence_intervals": self._calculate_confidence_intervals(bootstrap_metrics),
            "p_value": self._calculate_p_value(original_metric, bootstrap_metrics),
            "completed_at": datetime.utcnow().isoformat(),
        }

    def _block_bootstrap_sample(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Generate a block bootstrap sample.

        Args:
            df: Original dataframe

        Returns:
            Bootstrapped dataframe
        """
        n = len(df)
        block_size = min(self.block_size, n)

        # Calculate number of blocks needed
        n_blocks = int(np.ceil(n / block_size))

        # Generate random starting indices for blocks
        start_indices = np.random.randint(0, n - block_size + 1, size=n_blocks)

        # Collect blocks
        blocks = []
        for start_idx in start_indices:
            end_idx = min(start_idx + block_size, n)
            block = df.slice(start_idx, end_idx - start_idx)
            blocks.append(block)

        # Concatenate blocks
        boot_df = pl.concat(blocks)

        # Trim to original size
        if len(boot_df) > n:
            boot_df = boot_df.head(n)
        elif len(boot_df) < n:
            # Repeat last block to fill
            last_block = boot_df.tail(n - len(boot_df))
            boot_df = pl.concat([boot_df, last_block])

        return boot_df

    def _calculate_bootstrap_stats(self, original_metric: float, bootstrap_metrics: List[float]) -> dict:
        """
        Calculate bootstrap statistics.

        Args:
            original_metric: Original metric value
            bootstrap_metrics: Bootstrap metric values

        Returns:
            Statistics dictionary
        """
        if not bootstrap_metrics:
            return {"error": "No bootstrap samples"}

        boot_array = np.array(bootstrap_metrics)

        stats = {
            "mean": float(np.mean(boot_array)),
            "std": float(np.std(boot_array)),
            "median": float(np.median(boot_array)),
            "min": float(np.min(boot_array)),
            "max": float(np.max(boot_array)),
            "bias": float(original_metric - np.mean(boot_array)),  # Bias correction
            "bias_corrected_estimate": float(2 * original_metric - np.mean(boot_array)),  # Bias-corrected
            "acceleration": self._calculate_acceleration(boot_array, original_metric),
        }

        return stats

    def _calculate_confidence_intervals(self, bootstrap_metrics: List[float]) -> dict:
        """
        Calculate confidence intervals.

        Args:
            bootstrap_metrics: Bootstrap metric values

        Returns:
            Confidence intervals dictionary
        """
        if not bootstrap_metrics:
            return {"error": "No bootstrap samples"}

        boot_array = np.array(bootstrap_metrics)

        # Standard confidence interval (normal approximation)
        mean = np.mean(boot_array)
        std = np.std(boot_array)
        z_score = 1.96 if self.confidence_level == 0.95 else 1.645  # For 90% confidence

        normal_ci = {
            "lower": float(mean - z_score * std),
            "upper": float(mean + z_score * std),
        }

        # Percentile confidence interval
        alpha = 1 - self.confidence_level
        lower_percentile = alpha / 2
        upper_percentile = 1 - alpha / 2

        sorted_metrics = np.sort(boot_array)
        n_samples = len(sorted_metrics)

        perc_lower_idx = int(lower_percentile * n_samples)
        perc_upper_idx = int(upper_percentile * n_samples)

        percentile_ci = {
            "lower": float(sorted_metrics[perc_lower_idx]),
            "upper": float(sorted_metrics[perc_upper_idx]),
        }

        return {
            "normal_approximation": normal_ci,
            "percentile": percentile_ci,
            "confidence_level": self.confidence_level,
        }

    def _calculate_p_value(self, original_metric: float, bootstrap_metrics: List[float]) -> float:
        """
        Calculate p-value for the hypothesis that the true metric is zero.

        Args:
            original_metric: Original metric value
            bootstrap_metrics: Bootstrap metric values

        Returns:
            P-value
        """
        if not bootstrap_metrics:
            return 1.0

        boot_array = np.array(bootstrap_metrics)

        # Two-sided test: probability that |metric| >= |original_metric|
        observed_abs = abs(original_metric)
        extreme_count = np.sum(np.abs(boot_array) >= observed_abs)

        p_value = extreme_count / len(boot_array)
        return float(p_value)

    def _calculate_acceleration(self, bootstrap_metrics: np.ndarray, original_metric: float) -> float:
        """
        Calculate acceleration for BCa confidence intervals.

        Args:
            bootstrap_metrics: Bootstrap metric values
            original_metric: Original metric value

        Returns:
            Acceleration value
        """
        # Jackknife estimates
        n = len(bootstrap_metrics)
        if n < 3:
            return 0.0

        # Leave-one-out estimates (simplified)
        # In practice, this would require re-running the strategy n times
        # For now, we'll return 0 as a placeholder
        return 0.0

    def calculate_sharpe_ratio_significance(
        self,
        df: pl.DataFrame,
        strategy_func,
    ) -> dict:
        """
        Specialized method for Sharpe ratio significance testing.

        Args:
            df: OHLCV dataframe
            strategy_func: Strategy function

        Returns:
            Sharpe ratio significance results
        """
        # Run bootstrap on Sharpe ratio
        results = self.run_analysis(df, strategy_func, "sharpe_ratio")

        # Additional Sharpe-specific analysis
        original_sharpe = results["original_metric"]
        bootstrap_sharpes = results["bootstrap_metrics"]

        # Calculate information ratio (Sharpe ratio adjusted for estimation error)
        if results["statistics"] and "std" in results["statistics"]:
            est_error = results["statistics"]["std"]
            if est_error > 0:
                info_ratio = original_sharpe / est_error
            else:
                info_ratio = float('inf') if original_sharpe != 0 else 0.0
        else:
            info_ratio = 0.0

        # Effective number of independent bets
        n_independent = len(df) / (4.0 * (self.block_size ** 2)) if self.block_size > 0 else len(df)

        return {
            "original_sharpe": original_sharpe,
            "sharpe_significance": results,
            "information_ratio": info_ratio,
            "effective_independence": n_independent,
            "defensive_sharpe": self._calculate_defensive_sharpe(
                original_sharpe, est_error, n_independent
            ),
        }

    def _calculate_defensive_sharpe(self, original_sharpe: float, est_error: float, n_independent: float) -> float:
        """
        Calculate defensive Sharpe ratio that accounts for multiple testing.

        Args:
            original_sharpe: Original Sharpe ratio
            est_error: Estimation error
            n_independent: Effective number of independent tests

        Returns:
            Defensive Sharpe ratio
        """
        # Apply Bonferroni correction for multiple testing
        corrected_alpha = 0.05 / max(1, n_independent)  # At least 1 test
        
        # Calculate critical value for significance
        import scipy.stats as stats
        try:
            critical_value = stats.norm.ppf(1 - corrected_alpha/2)
            # Adjust Sharpe by critical value
            defensive_sharpe = original_sharpe - critical_value * est_error
        except:
            defensive_sharpe = original_sharpe  # Fallback
            
        return float(defensive_sharpe)
