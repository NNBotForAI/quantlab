"""
Signal generation with enforced shift(1) to prevent lookahead bias
"""
from typing import Optional

import polars as pl

from ..common.logging import get_logger

logger = get_logger(__name__)


def shift_enforce(
    df: pl.DataFrame,
    columns: list[str],
    periods: int = 1,
) -> pl.DataFrame:
    """
    Enforce shift on columns to prevent lookahead bias.

    Args:
        df: Input dataframe
        columns: Columns to shift
        periods: Number of periods to shift (default 1)

    Returns:
        Dataframe with shifted columns
    """
    for col in columns:
        if col in df.columns:
            df = df.with_columns(
                pl.col(col).shift(periods).alias(col)
            )
    return df


def momentum_signal(
    df: pl.DataFrame,
    period: int = 20,
    long_threshold: float = 0.02,
    short_threshold: float = -0.02,
) -> pl.DataFrame:
    """
    Generate momentum-based signals.

    Args:
        df: DataFrame with price data
        period: Lookback period
        long_threshold: Momentum threshold for long
        short_threshold: Momentum threshold for short

    Returns:
        DataFrame with signal column (-1, 0, 1)
    """
    # Compute momentum
    df = df.with_columns(
        (pl.col("close") / pl.col("close").shift(period) - 1).alias("momentum_raw")
    )

    # Enforce shift to prevent lookahead
    df = shift_enforce(df, ["momentum_raw"])

    # Generate signals
    df = df.with_columns(
        pl.when(pl.col("momentum_raw") >= long_threshold)
        .then(1)
        .when(pl.col("momentum_raw") <= short_threshold)
        .then(-1)
        .otherwise(0)
        .alias("signal")
    )

    return df


def rsi_signal(
    df: pl.DataFrame,
    period: int = 14,
    overbought: float = 70,
    oversold: float = 30,
) -> pl.DataFrame:
    """
    Generate RSI-based signals.

    Args:
        df: DataFrame with price data
        period: RSI period
        overbought: Overbought threshold
        oversold: Oversold threshold

    Returns:
        DataFrame with signal column
    """
    # Compute RSI
    from .indicators import rsi as compute_rsi
    df = compute_rsi(df, period=period)

    # Enforce shift
    df = shift_enforce(df, ["rsi"])

    # Generate signals
    df = df.with_columns(
        pl.when(pl.col("rsi") <= oversold)
        .then(1)
        .when(pl.col("rsi") >= overbought)
        .then(-1)
        .otherwise(0)
        .alias("signal")
    )

    return df


def ma_crossover_signal(
    df: pl.DataFrame,
    fast_period: int = 10,
    slow_period: int = 20,
) -> pl.DataFrame:
    """
    Generate moving average crossover signals.

    Args:
        df: DataFrame with price data
        fast_period: Fast MA period
        slow_period: Slow MA period

    Returns:
        DataFrame with signal column
    """
    from .indicators import sma

    # Compute MAs
    df = sma(df, fast_period)
    df = sma(df, slow_period)

    # Enforce shift
    df = shift_enforce(df, [f"sma_{fast_period}", f"sma_{slow_period}"])

    # Generate crossover signals
    df = df.with_columns([
        (pl.col(f"sma_{fast_period}") > pl.col(f"sma_{slow_period}")).alias("bullish"),
    ])
    df = df.with_columns(
        pl.col("bullish").diff().alias("bullish_diff")
    )
    df = df.with_columns(
        pl.when(pl.col("bullish_diff") == 1)
        .then(1)  # Golden cross
        .when(pl.col("bullish_diff") == -1)
        .then(-1)  # Death cross
        .otherwise(0)
        .alias("signal")
    )
    df = df.drop(["bullish", "bullish_diff"])

    return df


def bollinger_band_signal(
    df: pl.DataFrame,
    period: int = 20,
    std_dev: float = 2.0,
) -> pl.DataFrame:
    """
    Generate Bollinger band signals.

    Args:
        df: DataFrame with price data
        period: BB period
        std_dev: Standard deviations

    Returns:
        DataFrame with signal column
    """
    from .indicators import bollinger_bands

    # Compute Bollinger bands
    df = bollinger_bands(df, period=period, std_dev=std_dev)

    # Enforce shift
    df = shift_enforce(df, ["bb_upper", "bb_lower"])

    # Generate signals
    df = df.with_columns(
        pl.when(pl.col("close") <= pl.col("bb_lower"))
        .then(1)  # Price at lower band - potential buy
        .when(pl.col("close") >= pl.col("bb_upper"))
        .then(-1)  # Price at upper band - potential sell
        .otherwise(0)
        .alias("signal")
    )

    return df


def combine_signals(
    df: pl.DataFrame,
    signal_cols: list[str],
    method: str = "majority",
) -> pl.DataFrame:
    """
    Combine multiple signals.

    Args:
        df: DataFrame with multiple signal columns
        signal_cols: List of signal column names
        method: Combination method ("majority", "sum", "all")

    Returns:
        DataFrame with combined_signal column
    """
    if method == "majority":
        # Majority vote
        df = df.with_columns(
            pl.sum_horizontal(signal_cols).alias("signal_sum")
        )
        df = df.with_columns(
            pl.when(pl.col("signal_sum") > 0)
            .then(1)
            .when(pl.col("signal_sum") < 0)
            .then(-1)
            .otherwise(0)
            .alias("combined_signal")
        )
        df = df.drop("signal_sum")

    elif method == "sum":
        # Sum of signals (can be >1 or <-1)
        df = df.with_columns(
            pl.sum_horizontal(signal_cols).alias("combined_signal")
        )

    elif method == "all":
        # All signals must agree
        df = df.with_columns(
            pl.when(pl.all_horizontal(pl.col(col) == 1 for col in signal_cols))
            .then(1)
            .when(pl.all_horizontal(pl.col(col) == -1 for col in signal_cols))
            .then(-1)
            .otherwise(0)
            .alias("combined_signal")
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    return df
