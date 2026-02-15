"""
Regime-based analysis for different market conditions
"""
from datetime import datetime
from typing import Optional, List, Tuple

import numpy as np
import polars as pl

from ..common.logging import get_logger
from ..common.perf import measure_time

logger = get_logger(__name__)


class RegimeAnalysis:
    """
    Regime-based analysis to test strategy performance across different market conditions.
    """

    def __init__(self, spec: dict):
        """
        Initialize regime analysis.

        Args:
            spec: Strategy specification
        """
        self.spec = spec
        self.regime_detection_method = spec.get("validation", {}).get("regime_method", "volatility")
        self.lookback_window = spec.get("validation", {}).get("regime_lookback", 252)  # 1 year
        self.vol_threshold = spec.get("validation", {}).get("regime_vol_threshold", 0.15)  # 15% annualized vol
        self.ma_threshold = spec.get("validation", {}).get("regime_ma_threshold", 0.05)  # 5% above/below MA

    def run_analysis(
        self,
        df: pl.DataFrame,
        strategy_func,
    ) -> dict:
        """
        Run regime analysis.

        Args:
            df: OHLCV dataframe
            strategy_func: Function that takes (df, params) and returns (signals, metrics)

        Returns:
            Dictionary with regime analysis results
        """
        logger.info("regime_analysis_start")

        # Calculate regime indicators
        df_with_regimes = self._detect_regimes(df)

        # Get unique regimes
        regimes = df_with_regimes["regime"].unique().to_list()

        results = {}
        all_regime_results = []

        with measure_time("regime_analysis"):
            for regime in regimes:
                # Filter data for regime
                regime_df = df_with_regimes.filter(pl.col("regime") == regime)

                if len(regime_df) == 0:
                    continue

                # Run strategy on regime data
                try:
                    params = self.spec.get("backtest", {}).get("params", {})
                    signals, metrics = strategy_func(regime_df, params)

                    regime_result = {
                        "regime": regime,
                        "start_date": regime_df["ts_utc"].min().isoformat(),
                        "end_date": regime_df["ts_utc"].max().isoformat(),
                        "n_observations": len(regime_df),
                        "metrics": metrics,
                    }
                    results[regime] = regime_result
                    all_regime_results.append(regime_result)

                except Exception as e:
                    logger.warning("regime_analysis_error", regime=regime, error=str(e))

        # Calculate cross-regime statistics
        cross_regime_stats = self._calculate_cross_regime_stats(all_regime_results)

        logger.info("regime_analysis_complete", regimes=len(regimes))

        return {
            "regime_results": results,
            "cross_regime_stats": cross_regime_stats,
            "completed_at": datetime.utcnow().isoformat(),
        }

    def _detect_regimes(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Detect market regimes based on volatility and trend.

        Args:
            df: OHLCV dataframe

        Returns:
            Dataframe with regime column added
        """
        # Calculate returns
        df = df.with_columns(
            pl.col("close").pct_change().alias("returns")
        )

        # Calculate rolling volatility (annualized)
        vol_window = min(self.lookback_window, len(df) - 1)
        df = df.with_columns(
            (pl.col("returns").rolling_std(vol_window) * np.sqrt(252)).alias("ann_vol")
        )

        # Calculate trend (relative to moving average)
        ma_window = min(self.lookback_window // 2, len(df) - 1)
        df = df.with_columns(
            pl.col("close").rolling_mean(ma_window).alias("ma")
        )
        df = df.with_columns(
            ((pl.col("close") - pl.col("ma")) / pl.col("ma")).alias("relative_pos")
        )

        # Assign regimes based on volatility and trend
        df = df.with_columns(
            pl.when(pl.col("ann_vol") > self.vol_threshold)
            .then(
                pl.when(pl.col("relative_pos") > self.ma_threshold)
                .then(pl.lit("high_vol_up"))
                .when(pl.col("relative_pos") < -self.ma_threshold)
                .then(pl.lit("high_vol_down"))
                .otherwise(pl.lit("high_vol_sideways"))
            )
            .otherwise(
                pl.when(pl.col("relative_pos") > self.ma_threshold)
                .then(pl.lit("low_vol_up"))
                .when(pl.col("relative_pos") < -self.ma_threshold)
                .then(pl.lit("low_vol_down"))
                .otherwise(pl.lit("low_vol_sideways"))
            )
            .alias("regime")
        )

        return df

    def _calculate_cross_regime_stats(self, regime_results: List[dict]) -> dict:
        """
        Calculate statistics across regimes.

        Args:
            regime_results: List of regime analysis results

        Returns:
            Cross-regime statistics
        """
        if not regime_results:
            return {"error": "No regime results"}

        # Extract metrics for each regime
        all_metrics = {}
        regime_metrics = {}

        for result in regime_results:
            regime = result["regime"]
            metrics = result["metrics"]

            regime_metrics[regime] = metrics

            # Collect all metrics across regimes
            for metric_name, metric_value in metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(metric_value)

        # Calculate stability measures
        stability_measures = {}
        for metric_name, values in all_metrics.items():
            if len(values) > 1:
                # Calculate coefficient of variation (lower is more stable)
                mean_val = np.mean(values)
                std_val = np.std(values)

                if mean_val != 0:
                    cv = abs(std_val / mean_val)
                    stability_measures[f"{metric_name}_stability"] = 1.0 / (1.0 + cv)  # Higher is more stable
                else:
                    stability_measures[f"{metric_name}_stability"] = 0.0

        # Calculate regime-specific performance ranking
        performance_rankings = self._calculate_performance_rankings(regime_metrics)

        return {
            "n_regimes": len(regime_results),
            "regime_performance": regime_metrics,
            "stability_measures": stability_measures,
            "performance_rankings": performance_rankings,
            "regime_distribution": self._calculate_regime_distribution(regime_results),
        }

    def _calculate_performance_rankings(self, regime_metrics: dict) -> dict:
        """
        Calculate performance rankings across regimes.

        Args:
            regime_metrics: Metrics for each regime

        Returns:
            Performance rankings
        """
        if not regime_metrics:
            return {}

        # Rank regimes by key metrics
        rankings = {}

        # By Sharpe ratio
        sharpe_scores = {
            regime: metrics.get("sharpe_ratio", 0)
            for regime, metrics in regime_metrics.items()
        }
        sorted_by_sharpe = sorted(sharpe_scores.items(), key=lambda x: x[1], reverse=True)
        rankings["by_sharpe"] = {
            regime: rank + 1
            for rank, (regime, _) in enumerate(sorted_by_sharpe)
        }

        # By Calmar ratio
        calmar_scores = {
            regime: metrics.get("calmar_ratio", 0)
            for regime, metrics in regime_metrics.items()
        }
        sorted_by_calmar = sorted(calmar_scores.items(), key=lambda x: x[1], reverse=True)
        rankings["by_calmar"] = {
            regime: rank + 1
            for rank, (regime, _) in enumerate(sorted_by_calmar)
        }

        # By total return
        return_scores = {
            regime: metrics.get("total_return", 0)
            for regime, metrics in regime_metrics.items()
        }
        sorted_by_return = sorted(return_scores.items(), key=lambda x: x[1], reverse=True)
        rankings["by_return"] = {
            regime: rank + 1
            for rank, (regime, _) in enumerate(sorted_by_return)
        }

        return rankings

    def _calculate_regime_distribution(self, regime_results: List[dict]) -> dict:
        """
        Calculate distribution of observations across regimes.

        Args:
            regime_results: List of regime results

        Returns:
            Regime distribution
        """
        distribution = {}
        total_obs = sum(result["n_observations"] for result in regime_results)

        for result in regime_results:
            regime = result["regime"]
            n_obs = result["n_observations"]
            distribution[regime] = {
                "count": n_obs,
                "percentage": (n_obs / total_obs) * 100 if total_obs > 0 else 0,
            }

        return distribution

    def detect_market_regimes_simple(
        self,
        prices: np.ndarray,
        method: str = "volatility_trend",
    ) -> np.ndarray:
        """
        Simple regime detection for given price series.

        Args:
            prices: Price series
            method: Regime detection method

        Returns:
            Array of regime labels
        """
        returns = np.diff(np.log(prices))
        
        if method == "volatility_trend":
            # Calculate rolling volatility and trend
            vol_window = min(63, len(returns))  # 3-month window
            rolling_vol = np.zeros(len(returns))
            
            for i in range(vol_window, len(returns)):
                rolling_vol[i] = np.std(returns[i-vol_window:i]) * np.sqrt(252)
            
            # Simple classification
            high_vol = rolling_vol > self.vol_threshold
            uptrend = prices[1:] > np.convolve(prices[1:], np.ones(20)/20, mode='same')  # 20-day MA
            
            regimes = np.where(high_vol, 
                              np.where(uptrend[vol_window:], "high_vol_up", "high_vol_down"),
                              np.where(uptrend[vol_window:], "low_vol_up", "low_vol_down"))
            
            # Prepend NaNs to match original length
            regimes_full = np.full(len(prices), "unknown", dtype=object)
            regimes_full[vol_window+1:] = regimes
            
            return regimes_full
        else:
            # Default: return all as 'normal'
            return np.full(len(prices), "normal", dtype=object)
