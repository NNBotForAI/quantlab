"""
Walk-forward analysis for strategy robustness
"""
from datetime import datetime
from typing import Optional, List, Tuple

import numpy as np
import polars as pl

from ..common.logging import get_logger
from ..common.perf import measure_time

logger = get_logger(__name__)


class WalkForwardAnalysis:
    """
    Walk-forward analysis to test strategy robustness over time.
    """

    def __init__(self, spec: dict):
        """
        Initialize walk-forward analysis.

        Args:
            spec: Strategy specification
        """
        self.spec = spec
        self.lookback_window = spec.get("validation", {}).get("walk_forward_lookback", 252)  # 1 year
        self.holdout_window = spec.get("validation", {}).get("walk_forward_holdout", 63)  # 3 months
        self.min_training_window = spec.get("validation", {}).get("walk_forward_min_train", 126)  # 6 months

    def run_analysis(
        self,
        df: pl.DataFrame,
        strategy_func,
        initial_capital: float,
    ) -> dict:
        """
        Run walk-forward analysis.

        Args:
            df: OHLCV dataframe with ts_utc column
            strategy_func: Function that takes (df, params) and returns (signals, metrics)
            initial_capital: Starting capital

        Returns:
            Dictionary with walk-forward results
        """
        logger.info("walk_forward_start")

        # Sort dataframe by timestamp
        df = df.sort("ts_utc")

        # Get unique dates
        dates = df["ts_utc"].unique().sort()

        if len(dates) < self.min_training_window + self.holdout_window:
            logger.warning("insufficient_data", required=self.min_training_window + self.holdout_window, available=len(dates))
            return {"error": "Insufficient data for walk-forward analysis"}

        results = []
        current_idx = self.min_training_window

        with measure_time("walk_forward_analysis"):
            while current_idx + self.holdout_window <= len(dates):
                # Define training and holdout periods
                train_end_idx = current_idx
                holdout_start_idx = current_idx
                holdout_end_idx = current_idx + self.holdout_window

                # Get date ranges
                train_dates = dates[:train_end_idx]
                holdout_dates = dates[holdout_start_idx:holdout_end_idx]

                # Filter dataframes
                train_df = df.filter(
                    pl.col("ts_utc").is_between(train_dates[0], train_dates[-1])
                )
                holdout_df = df.filter(
                    pl.col("ts_utc").is_between(holdout_dates[0], holdout_dates[-1])
                )

                if len(train_df) < self.min_training_window or len(holdout_df) == 0:
                    current_idx += 1
                    continue

                try:
                    # Optimize strategy on training data
                    best_params = self._optimize_on_window(train_df, strategy_func)

                    # Test on holdout data
                    holdout_signals, holdout_metrics = strategy_func(holdout_df, best_params)

                    # Store results
                    window_result = {
                        "training_period": {
                            "start": train_dates[0].isoformat(),
                            "end": train_dates[-1].isoformat(),
                        },
                        "holdout_period": {
                            "start": holdout_dates[0].isoformat(),
                            "end": holdout_dates[-1].isoformat(),
                        },
                        "best_params": best_params,
                        "holdout_metrics": holdout_metrics,
                    }
                    results.append(window_result)

                except Exception as e:
                    logger.warning("window_error", start=holdout_dates[0], end=holdout_dates[-1], error=str(e))

                current_idx += self.holdout_window

        # Aggregate results
        aggregated_results = self._aggregate_results(results)

        logger.info("walk_forward_complete", windows=len(results))

        return {
            "windows": results,
            "aggregated": aggregated_results,
            "completed_at": datetime.utcnow().isoformat(),
        }

    def _optimize_on_window(
        self,
        df: pl.DataFrame,
        strategy_func,
    ) -> dict:
        """
        Optimize strategy on a single window.

        Args:
            df: Training dataframe
            strategy_func: Strategy function

        Returns:
            Best parameters
        """
        # For this implementation, we'll use a simple grid search
        # In production, this would use the optimization framework
        best_score = float('-inf')
        best_params = {}

        # Example parameter grid (this would come from spec in production)
        for momentum_period in [10, 20, 30, 40]:
            for long_thresh in [0.01, 0.02, 0.03]:
                for short_thresh in [-0.01, -0.02, -0.03]:
                    params = {
                        "momentum_period": momentum_period,
                        "long_threshold": long_thresh,
                        "short_threshold": short_thresh,
                    }

                    try:
                        signals, metrics = strategy_func(df, params)
                        score = metrics.get("sharpe_ratio", 0)  # Use Sharpe as score

                        if score > best_score:
                            best_score = score
                            best_params = params

                    except Exception:
                        continue

        return best_params

    def _aggregate_results(self, results: List[dict]) -> dict:
        """
        Aggregate walk-forward results.

        Args:
            results: List of window results

        Returns:
            Aggregated metrics
        """
        if not results:
            return {"error": "No results to aggregate"}

        # Extract metrics from each window
        sharpe_ratios = []
        calmar_ratios = []
        max_drawdowns = []
        total_returns = []

        for result in results:
            metrics = result["holdout_metrics"]
            sharpe_ratios.append(metrics.get("sharpe_ratio", 0))
            calmar_ratios.append(metrics.get("calmar_ratio", 0))
            max_drawdowns.append(metrics.get("max_drawdown", 0))
            total_returns.append(metrics.get("total_return", 0))

        # Calculate statistics
        return {
            "sharpe_ratio": {
                "mean": np.mean(sharpe_ratios),
                "std": np.std(sharpe_ratios),
                "min": np.min(sharpe_ratios),
                "max": np.max(sharpe_ratios),
                "median": np.median(sharpe_ratios),
            },
            "calmar_ratio": {
                "mean": np.mean(calmar_ratios),
                "std": np.std(calmar_ratios),
                "min": np.min(calmar_ratios),
                "max": np.max(calmar_ratios),
                "median": np.median(calmar_ratios),
            },
            "max_drawdown": {
                "mean": np.mean(max_drawdowns),
                "std": np.std(max_drawdowns),
                "min": np.min(max_drawdowns),
                "max": np.max(max_drawdowns),
                "median": np.median(max_drawdowns),
            },
            "total_return": {
                "mean": np.mean(total_returns),
                "std": np.std(total_returns),
                "min": np.min(total_returns),
                "max": np.max(total_returns),
                "median": np.median(total_returns),
            },
            "consistency_score": self._calculate_consistency_score(sharpe_ratios),
            "total_windows": len(results),
        }

    def _calculate_consistency_score(self, sharpe_series: List[float]) -> float:
        """
        Calculate consistency score based on stability of Sharpe ratios.

        Args:
            sharpe_series: Series of Sharpe ratios over time

        Returns:
            Consistency score (0-1, higher is more consistent)
        """
        if len(sharpe_series) < 2:
            return 0.0

        # Calculate coefficient of variation (lower is more consistent)
        mean_sharpe = np.mean(sharpe_series)
        std_sharpe = np.std(sharpe_series)

        if mean_sharpe == 0:
            return 0.0

        # Coefficient of variation
        cv = abs(std_sharpe / mean_sharpe)

        # Convert to score (1 - normalized CV)
        # Cap CV at 1.0 to prevent negative scores
        cv_normalized = min(cv, 1.0)
        consistency_score = max(0.0, 1.0 - cv_normalized)

        return consistency_score
