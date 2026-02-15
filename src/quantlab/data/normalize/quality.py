"""
Data quality checks
"""
from typing import Optional

import polars as pl

from ...common.logging import get_logger

logger = get_logger(__name__)


def check_ohlc_consistency(df: pl.DataFrame) -> pl.DataFrame:
    """
    Check OHLC data consistency.

    Args:
        df: OHLCV dataframe

    Returns:
        DataFrame with quality flags
    """
    df = df.with_columns([
        # High should be >= Open, Low, Close
        (pl.col("high") >= pl.col("open")).alias("high_ge_open"),
        (pl.col("high") >= pl.col("low")).alias("high_ge_low"),
        (pl.col("high") >= pl.col("close")).alias("high_ge_close"),

        # Low should be <= Open, High, Close
        (pl.col("low") <= pl.col("open")).alias("low_le_open"),
        (pl.col("low") <= pl.col("high")).alias("low_le_high"),
        (pl.col("low") <= pl.col("close")).alias("low_le_close"),

        # Volume should be >= 0
        (pl.col("volume") >= 0).alias("volume_non_negative"),

        # Prices should be > 0
        (pl.col("open") > 0).alias("open_positive"),
        (pl.col("high") > 0).alias("high_positive"),
        (pl.col("low") > 0).alias("low_positive"),
        (pl.col("close") > 0).alias("close_positive"),
    ])

    # Check if any flags are False
    quality_issues = df.select([
        pl.col("high_ge_open").all().alias("all_high_ge_open"),
        pl.col("volume_non_negative").all().alias("all_volume_positive"),
        pl.col("open_positive").all().alias("all_open_positive"),
    ])

    logger.info("quality_check", **quality_issues.row(0, named=True))

    return df


def detect_duplicates(df: pl.DataFrame) -> Optional[pl.DataFrame]:
    """
    Detect duplicate timestamps.

    Args:
        df: OHLCV dataframe

    Returns:
        DataFrame with duplicates, or None if no duplicates
    """
    dupes = df.filter(
        df.select([
            pl.col("ts_utc").is_duplicated().alias("is_dupe")
        ]).select("is_dupe")
    )

    if not dupes.is_empty():
        logger.warning("duplicates_found", count=dupes.height)
        return dupes
    return None


def detect_gaps(
    df: pl.DataFrame,
    expected_freq: str,
) -> Optional[pl.DataFrame]:
    """
    Detect gaps in time series.

    Args:
        df: OHLCV dataframe (should be sorted by ts_utc)
        expected_freq: Expected frequency (e.g., "1h", "1d")

    Returns:
        DataFrame with gap information, or None if no gaps
    """
    if df.height < 2:
        return None

    # Calculate expected time delta
    freq_map = {
        "1m": timedelta(minutes=1),
        "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15),
        "1H": timedelta(hours=1),
        "1D": timedelta(days=1),
    }

    expected_delta = freq_map.get(expected_freq)
    if expected_delta is None:
        logger.warning("unknown_freq", freq=expected_freq)
        return None

    # Calculate actual deltas
    df = df.sort("ts_utc")
    df = df.with_columns(
        (pl.col("ts_utc").diff() / 1_000_000_000).alias("delta_seconds")
    )

    expected_seconds = expected_delta.total_seconds()

    # Find gaps
    gaps = df.filter(
        pl.col("delta_seconds") > expected_seconds
    )

    if not gaps.is_empty():
        logger.warning("gaps_found", count=gaps.height)
        return gaps
    return None


def clean_data(
    df: pl.DataFrame,
    drop_duplicates: bool = True,
    fill_zeros: bool = False,
) -> pl.DataFrame:
    """
    Clean OHLCV data.

    Args:
        df: OHLCV dataframe
        drop_duplicates: Remove duplicate timestamps
        fill_zeros: Fill zero prices with previous close

    Returns:
        Cleaned dataframe
    """
    # Remove duplicates
    if drop_duplicates:
        before = df.height
        df = df.unique(subset=["ts_utc", "symbol"], keep="first")
        if df.height < before:
            logger.info("duplicates_removed", count=before - df.height)

    # Fill zero prices with previous close
    if fill_zeros:
        df = df.sort("ts_utc")
        for col in ["open", "high", "low", "close"]:
            df = df.with_columns(
                pl.when(pl.col(col) == 0)
                .then(pl.col("close").shift(1))
                .otherwise(pl.col(col))
                .alias(col)
            )

    return df
