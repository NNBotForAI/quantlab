"""
Technical indicators using Polars and Numba
"""
from typing import Optional

import numba
import numpy as np
import polars as pl

from ..common.logging import get_logger

logger = get_logger(__name__)


# Numba-optimized rolling functions for cases where Polars is slow
@numba.njit
def _rolling_mean_numba(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean using Numba."""
    result = np.empty_like(arr)
    result[:window-1] = np.nan

    for i in range(window-1, len(arr)):
        result[i] = np.nanmean(arr[i-window+1:i+1])

    return result


@numba.njit
def _rolling_std_numba(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling std using Numba."""
    result = np.empty_like(arr)
    result[:window-1] = np.nan

    for i in range(window-1, len(arr)):
        window_data = arr[i-window+1:i+1]
        result[i] = np.nanstd(window_data)

    return result


@numba.njit
def _rolling_rsi_numba(arr: np.ndarray, period: int) -> np.ndarray:
    """Rolling RSI using Numba."""
    deltas = np.diff(arr)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum()/period
    down = -seed[seed < 0].sum()/period
    rs = up/down if down != 0 else np.inf

    result = np.empty(len(arr))
    result[:period] = np.nan

    for i in range(period, len(arr)):
        delta = deltas[i-1]

        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period

        rs = up / down if down != 0 else np.inf
        result[i] = 100. - 100. / (1. + rs)

    return result


def momentum(
    df: pl.DataFrame,
    period: int = 20,
    price_col: str = "close",
) -> pl.DataFrame:
    """
    Compute momentum indicator.

    Args:
        df: OHLCV dataframe
        period: Lookback period
        price_col: Price column to use

    Returns:
        DataFrame with momentum column
    """
    df = df.with_columns([
        (pl.col(price_col) / pl.col(price_col).shift(period) - 1).alias("momentum")
    ])

    return df


def volatility(
    df: pl.DataFrame,
    period: int = 20,
    price_col: str = "close",
) -> pl.DataFrame:
    """
    Compute rolling volatility.

    Args:
        df: OHLCV dataframe
        period: Lookback period
        price_col: Price column to use

    Returns:
        DataFrame with volatility column
    """
    # Compute returns first
    df = df.with_columns(
        pl.col(price_col).pct_change().alias("returns")
    )

    # Rolling std of returns
    df = df.with_columns(
        pl.col("returns").rolling_std(period).alias("volatility")
    )

    return df


def rsi(
    df: pl.DataFrame,
    period: int = 14,
    price_col: str = "close",
    use_numba: bool = True,
) -> pl.DataFrame:
    """
    Compute RSI indicator.

    Args:
        df: OHLCV dataframe
        period: RSI period
        price_col: Price column to use
        use_numba: Use Numba for faster computation

    Returns:
        DataFrame with RSI column
    """
    if use_numba:
        # Use Numba for performance
        arr = df[price_col].to_numpy()
        rsi_values = _rolling_rsi_numba(arr, period)
        df = df.with_columns(
            pl.Series("rsi", rsi_values)
        )
    else:
        # Use Polars (slower but simpler)
        df = df.with_columns(
            pl.col(price_col)
            .diff()
            .alias("delta")
        )
        df = df.with_columns([
            pl.when(pl.col("delta") >= 0)
            .then(pl.col("delta"))
            .otherwise(0)
            .alias("gain"),
            pl.when(pl.col("delta") < 0)
            .then(-pl.col("delta"))
            .otherwise(0)
            .alias("loss"),
        ])
        df = df.with_columns([
            pl.col("gain").rolling_mean(period).alias("avg_gain"),
            pl.col("loss").rolling_mean(period).alias("avg_loss"),
        ])
        df = df.with_columns(
            (100 - (100 / (1 + pl.col("avg_gain") / pl.col("avg_loss")))).alias("rsi")
        )
        df = df.drop(["delta", "gain", "loss", "avg_gain", "avg_loss"])

    return df


def sma(
    df: pl.DataFrame,
    period: int,
    price_col: str = "close",
) -> pl.DataFrame:
    """
    Simple moving average.

    Args:
        df: OHLCV dataframe
        period: SMA period
        price_col: Price column to use

    Returns:
        DataFrame with SMA column
    """
    df = df.with_columns(
        pl.col(price_col).rolling_mean(period).alias(f"sma_{period}")
    )
    return df


def ema(
    df: pl.DataFrame,
    period: int,
    price_col: str = "close",
) -> pl.DataFrame:
    """
    Exponential moving average.

    Args:
        df: OHLCV dataframe
        period: EMA period
        price_col: Price column to use

    Returns:
        DataFrame with EMA column
    """
    span = period
    alpha = 2 / (span + 1)

    df = df.with_columns(
        pl.col(price_col)
        .ewm_mean(alpha=alpha, adjust=False)
        .alias(f"ema_{period}")
    )
    return df


def bollinger_bands(
    df: pl.DataFrame,
    period: int = 20,
    std_dev: float = 2.0,
    price_col: str = "close",
) -> pl.DataFrame:
    """
    Bollinger bands.

    Args:
        df: OHLCV dataframe
        period: Period for bands
        std_dev: Standard deviations
        price_col: Price column to use

    Returns:
        DataFrame with BB upper, middle, lower
    """
    df = df.with_columns(
        pl.col(price_col).rolling_mean(period).alias("bb_middle"),
    )
    df = df.with_columns(
        pl.col(price_col).rolling_std(period).alias("bb_std"),
    )
    df = df.with_columns([
        (pl.col("bb_middle") + std_dev * pl.col("bb_std")).alias("bb_upper"),
        (pl.col("bb_middle") - std_dev * pl.col("bb_std")).alias("bb_lower"),
    ])

    return df


def atr(
    df: pl.DataFrame,
    period: int = 14,
) -> pl.DataFrame:
    """
    Average True Range.

    Args:
        df: OHLCV dataframe
        period: ATR period

    Returns:
        DataFrame with ATR column
    """
    df = df.with_columns(
        (pl.col("high") - pl.col("low")).alias("tr1"),
        (pl.col("high") - pl.col("close").shift(1).abs()).alias("tr2"),
        (pl.col("low") - pl.col("close").shift(1).abs()).alias("tr3"),
    )
    df = df.with_columns(
        pl.max_horizontal(["tr1", "tr2", "tr3"]).alias("true_range")
    )
    df = df.with_columns(
        pl.col("true_range").rolling_mean(period).alias("atr")
    )
    df = df.drop(["tr1", "tr2", "tr3", "true_range"])

    return df
