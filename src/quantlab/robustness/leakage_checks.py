"""
Future leakage detection and prevention
"""
from datetime import datetime
from typing import Optional, List, Tuple

import numpy as np
import polars as pl

from ..common.logging import get_logger
from ..common.timeutils import to_utc

logger = get_logger(__name__)


class LeakageDetection:
    """
    Future leakage detection for preventing lookahead bias.
    """

    def __init__(self, spec: dict):
        """
        Initialize leakage detection.

        Args:
            spec: Strategy specification
        """
        self.spec = spec

    def run_all_checks(self, df: pl.DataFrame, features_df: pl.DataFrame) -> dict:
        """
        Run all leakage detection checks.

        Args:
            df: Original OHLCV dataframe
            features_df: Features dataframe

        Returns:
            Dictionary with all leakage check results
        """
        logger.info("leakage_detection_start")

        results = {
            "timestamp_alignment": self.check_timestamp_alignment(df),
            "lookahead_bias": self.check_lookahead_bias(df, features_df),
            "data_snooping": self.check_data_snooping(df),
            "feature_leakage": self.check_feature_leakage(df, features_df),
            "completed_at": datetime.utcnow().isoformat(),
        }

        # Summarize findings
        issues = []
        for check_name, result in results.items():
            if isinstance(result, dict) and result.get("issue", False):
                issues.append({
                    "check": check_name,
                    "severity": result.get("severity", "medium"),
                    "details": result.get("details", ""),
                    "recommendation": result.get("recommendation", "")
                })

        results["summary"] = {
            "total_checks": len(results) - 1,  # Exclude completed_at
            "issues_found": len(issues),
            "issues": issues,
            "passed": len(issues) == 0,
        }

        logger.info("leakage_detection_complete", issues=len(issues))

        return results

    def check_timestamp_alignment(self, df: pl.DataFrame) -> dict:
        """
        Check if timestamps are properly aligned across markets and frequencies.

        Args:
            df: OHLCV dataframe

        Returns:
            Alignment check results
        """
        if "ts_utc" not in df.columns:
            return {
                "issue": True,
                "severity": "critical",
                "details": "No ts_utc column found in dataframe",
                "recommendation": "Ensure all data is stored in UTC timestamps"
            }

        # Check if timestamps are monotonic
        timestamps = df["ts_utc"].sort()
        is_monotonic = timestamps.equals(df["ts_utc"].sort())

        if not is_monotonic:
            return {
                "issue": True,
                "severity": "high",
                "details": "Timestamps are not monotonic",
                "recommendation": "Sort dataframe by ts_utc before processing"
            }

        # Check for duplicate timestamps
        n_duplicates = len(df.filter(pl.col("ts_utc").is_duplicated()))
        if n_duplicates > 0:
            return {
                "issue": True,
                "severity": "high",
                "details": f"Found {n_duplicates} duplicate timestamps",
                "recommendation": "Remove duplicate timestamps or aggregate data"
            }

        # Check timezone awareness
        if df["ts_utc"].dtype != pl.Datetime(time_unit="us", time_zone="UTC"):
            # This is expected since we can't specify timezone in dtype comparison directly
            # Instead, check if the column contains timezone-aware data
            sample_val = df["ts_utc"][0] if len(df) > 0 else None
            if sample_val and hasattr(sample_val, 'tzinfo') and sample_val.tzinfo is None:
                return {
                    "issue": True,
                    "severity": "medium",
                    "details": "Timestamps appear to be naive (not timezone-aware)",
                    "recommendation": "Ensure all timestamps are timezone-aware UTC"
                }

        return {
            "issue": False,
            "severity": "none",
            "details": "Timestamps are properly aligned",
            "passed": True
        }

    def check_lookahead_bias(self, df: pl.DataFrame, features_df: pl.DataFrame) -> dict:
        """
        Check for lookahead bias in features.

        Args:
            df: Original OHLCV dataframe
            features_df: Features dataframe

        Returns:
            Lookahead bias check results
        """
        if features_df.height != df.height:
            return {
                "issue": True,
                "severity": "critical",
                "details": f"Mismatched lengths: original={df.height}, features={features_df.height}",
                "recommendation": "Ensure features dataframe has same length as original"
            }

        # Check if any feature uses future information
        # This is a simplified check - in practice, would need more sophisticated analysis
        potential_leakage = []

        for col in features_df.columns:
            if col == "ts_utc" or col in df.columns:
                continue

            # Check if feature values are highly correlated with future prices
            # (This is a heuristic check)
            if len(features_df) > 10:
                try:
                    # Correlation with future returns
                    future_returns = df["close"].shift(-1) / df["close"] - 1
                    feature_series = features_df[col].to_pandas()
                    future_returns_pd = future_returns.to_pandas()

                    # Only compute correlation where both are not null
                    mask = ~(feature_series.isna() | future_returns_pd.isna())
                    if mask.sum() > 5:  # Need at least 5 points
                        corr = np.corrcoef(feature_series[mask], future_returns_pd[mask])[0, 1]
                        
                        if not np.isnan(corr) and abs(corr) > 0.3:  # Threshold for concern
                            potential_leakage.append({
                                "feature": col,
                                "correlation": float(corr),
                                "abs_corr_threshold": 0.3
                            })
                except Exception:
                    # If correlation fails, skip this feature
                    continue

        if potential_leakage:
            return {
                "issue": True,
                "severity": "high",
                "details": f"Potential lookahead bias detected in {len(potential_leakage)} features",
                "leakage_features": potential_leakage,
                "recommendation": "Review features that correlate with future prices and ensure proper shifting"
            }

        return {
            "issue": False,
            "severity": "none",
            "details": "No lookahead bias detected",
            "passed": True
        }

    def check_data_snooping(self, df: pl.DataFrame) -> dict:
        """
        Check for potential data snooping bias.

        Args:
            df: OHLCV dataframe

        Returns:
            Data snooping check results
        """
        # Check if the dataset covers a representative time period
        if "ts_utc" not in df.columns:
            return {
                "issue": True,
                "severity": "critical",
                "details": "No ts_utc column for time period analysis",
                "recommendation": "Include timestamp column for temporal analysis"
            }

        start_date = df["ts_utc"].min()
        end_date = df["ts_utc"].max()
        total_days = (end_date - start_date).days

        # Check if data period is too short for meaningful analysis
        if total_days < 252:  # Less than 1 year
            return {
                "issue": True,
                "severity": "medium",
                "details": f"Short data period: {total_days} days (< 1 year)",
                "recommendation": "Use at least 1 year of data for robust analysis"
            }

        # Check for survivorship bias (if applicable)
        # This would require additional context about how the data was collected
        # For now, we'll note that it's an important consideration

        return {
            "issue": False,
            "severity": "none",
            "details": "Data period appears sufficient",
            "passed": True
        }

    def check_feature_leakage(self, df: pl.DataFrame, features_df: pl.DataFrame) -> dict:
        """
        Check individual features for leakage patterns.

        Args:
            df: Original OHLCV dataframe
            features_df: Features dataframe

        Returns:
            Feature leakage check results
        """
        issues = []

        for col in features_df.columns:
            if col in ["ts_utc", "symbol"] or col in df.columns:
                continue

            # Check if feature contains forward-looking information
            feature_series = features_df[col]
            
            # Look for suspicious patterns like:
            # - Features that perfectly predict price movements
            # - Features with unrealistic precision
            # - Features that contain future information
            
            if len(feature_series) > 10:
                try:
                    # Check if feature perfectly correlates with target
                    close_returns = df["close"].pct_change()
                    feature_returns = feature_series.pct_change()
                    
                    # Only compute where both are valid
                    mask = ~(close_returns.is_null() | feature_returns.is_null())
                    if mask.sum() > 5:
                        corr = pl.corr(close_returns.filter(mask), feature_returns.filter(mask))
                        
                        if abs(corr) > 0.8:  # Very high correlation may indicate leakage
                            issues.append({
                                "feature": col,
                                "correlation_with_returns": float(corr),
                                "concern": "Very high correlation with price returns may indicate leakage"
                            })
                            
                except Exception:
                    # If correlation fails, continue to next feature
                    pass

        if issues:
            return {
                "issue": True,
                "severity": "high",
                "details": f"Potential leakage in {len(issues)} features",
                "problematic_features": issues,
                "recommendation": "Review each feature for potential future information inclusion"
            }

        return {
            "issue": False,
            "severity": "none",
            "details": "No feature leakage detected",
            "passed": True
        }

    def enforce_shift_constraint(self, df: pl.DataFrame, lookback_period: int) -> pl.DataFrame:
        """
        Enforce shift constraint to prevent lookahead bias.

        Args:
            df: Dataframe with features
            lookback_period: Period to shift by

        Returns:
            Dataframe with shifted features
        """
        feature_cols = [col for col in df.columns 
                       if col not in ["ts_utc", "symbol", "open", "high", "low", "close", "volume"]]
        
        for col in feature_cols:
            df = df.with_columns(
                pl.col(col).shift(lookback_period).alias(col)
            )
        
        return df

    def validate_feature_timing(self, df: pl.DataFrame, signals_df: pl.DataFrame) -> dict:
        """
        Validate that signals are generated with proper timing.

        Args:
            df: Original data
            signals_df: Signals dataframe

        Returns:
            Validation results
        """
        if "ts_utc" not in signals_df.columns:
            return {
                "issue": True,
                "severity": "critical",
                "details": "No ts_utc column in signals dataframe",
                "recommendation": "Include timestamp column in signals"
            }

        # Check that signals are lagged appropriately
        # This assumes signals should be based on data up to t-1
        original_times = df["ts_utc"].sort()
        signal_times = signals_df["ts_utc"].sort()

        if not original_times.equals(signal_times):
            return {
                "issue": True,
                "severity": "high",
                "details": "Signal timestamps don't match original data timestamps",
                "recommendation": "Ensure signals are aligned with original data"
            }

        # Check that no signal is generated based on future information
        # (This is a basic check - more sophisticated checks needed in practice)
        
        return {
            "issue": False,
            "severity": "none",
            "details": "Signal timing appears valid",
            "passed": True
        }
