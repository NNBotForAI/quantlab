"""
Price adjustments and corporate actions
"""
from typing import Optional

import polars as pl

from ...common.logging import get_logger

logger = get_logger(__name__)


def apply_adjustments(
    df: pl.DataFrame,
    price_mode: str = "raw",
) -> pl.DataFrame:
    """
    Apply price adjustments for splits, dividends, etc.

    Args:
        df: OHLCV dataframe
        price_mode: "raw" or "adjusted"

    Returns:
        Adjusted dataframe
    """
    if price_mode == "raw":
        return df

    # For "adjusted" mode, we would need adjustment data
    # This is a placeholder - in production, load adjustment ratios
    # from a separate source and multiply OHLCV accordingly

    logger.warning(
        "adjustments_placeholder",
        message="Price adjustments not implemented - returning raw prices"
    )
    return df


def forward_fill_prices(
    df: pl.DataFrame,
    max_gap: Optional[int] = None,
) -> pl.DataFrame:
    """
    Forward fill missing price data.

    Args:
        df: OHLCV dataframe
        max_gap: Maximum gap to fill (None = fill all)

    Returns:
        Forward-filled dataframe
    """
    if max_gap is None:
        # Forward fill all gaps
        df = df.select([
            pl.col("ts_utc"),
            pl.col("symbol"),
            pl.col("open").fill_null(strategy="forward"),
            pl.col("high").fill_null(strategy="forward"),
            pl.col("low").fill_null(strategy="forward"),
            pl.col("close").fill_null(strategy="forward"),
            pl.col("volume").fill_null(0),  # Zero volume for missing bars
        ])
    else:
        # Fill gaps up to max_gap bars
        for col in ["open", "high", "low", "close"]:
            df = df.with_columns(
                pl.col(col).fill_null(strategy="forward", limit=max_gap)
            )
        df = df.with_columns(
            pl.col("volume").fill_null(0)
        )

    return df
