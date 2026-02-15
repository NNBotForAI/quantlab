"""
Optuna objective functions for different strategies
"""
from typing import Dict, Any, Callable

import numpy as np
import optuna
import polars as pl

from ..common.logging import get_logger
from ..features.indicators import momentum, rsi, sma
from ..backtest.metrics import calculate_all_metrics
from ..backtest.vectorbt_engine import VectorBTBacktestEngine

logger = get_logger(__name__)


def create_objective_function(
    spec: dict,
    data_df: pl.DataFrame,
    score_weights: Dict[str, float] = None,
) -> Callable:
    """
    Create an objective function for Optuna optimization.

    Args:
        spec: Strategy specification
        data_df: Input data for backtesting
        score_weights: Weights for different metrics in scoring function

    Returns:
        Objective function that takes parameters and returns a score
    """
    if score_weights is None:
        # Default weights: prioritize Sharpe and Calmar, penalize turnover and overfitting
        score_weights = {
            "sharpe_ratio": 0.6,
            "calmar_ratio": 0.6,
            "turnover_penalty": -0.2,
            "overfit_penalty": -0.2,
        }

    def objective(trial: optuna.Trial) -> float:
        """
        Objective function for Optuna.

        Args:
            trial: Optuna trial object

        Returns:
            Score to maximize
        """
        # Get parameters from trial
        params = _extract_trial_params(trial, spec)

        try:
            # Generate signals based on parameters
            signals_df = _generate_signals(data_df, spec, params)

            # Run backtest
            engine = VectorBTBacktestEngine(spec)
            signals = signals_df["signal"].to_numpy()

            results = engine.run_backtest(
                df=signals_df,
                signals=signals,
                initial_capital=spec["backtest"]["initial_capital"]
            )

            # Calculate metrics
            metrics = calculate_all_metrics(
                results["equity_curve"],
                results["trades"],
                spec["backtest"]["initial_capital"]
            )

            # Calculate composite score
            score = _calculate_composite_score(metrics, score_weights)

            return score

        except Exception as e:
            logger.warning("objective_error", params=params, error=str(e))
            return float('-inf')

    return objective


def _extract_trial_params(trial: optuna.Trial, spec: dict) -> Dict[str, Any]:
    """
    Extract parameters from Optuna trial based on spec.

    Args:
        trial: Optuna trial
        spec: Strategy specification

    Returns:
        Dictionary of parameters
    """
    params = {}

    if "optimization" in spec:
        for param_name, param_def in spec["optimization"].items():
            param_type = param_def.get("type", "float")
            lower = param_def.get("lower")
            upper = param_def.get("upper")
            choices = param_def.get("choices")

            if param_type == "float":
                params[param_name] = trial.suggest_float(param_name, lower, upper)
            elif param_type == "int":
                step = param_def.get("step", 1)
                params[param_name] = trial.suggest_int(param_name, lower, upper, step=step)
            elif param_type == "categorical":
                params[param_name] = trial.suggest_categorical(param_name, choices)

    return params


def _generate_signals(
    df: pl.DataFrame,
    spec: dict,
    params: Dict[str, Any],
) -> pl.DataFrame:
    """
    Generate signals based on strategy spec and parameters.

    Args:
        df: Input dataframe
        spec: Strategy specification
        params: Optimized parameters

    Returns:
        Dataframe with signal column
    """
    strategy_type = spec.get("strategy_type", "momentum")

    if strategy_type == "momentum":
        # Use momentum with optimized period
        period = params.get("momentum_period", 20)
        long_thresh = params.get("long_threshold", 0.02)
        short_thresh = params.get("short_threshold", -0.02)

        df = df.with_columns(
            (pl.col("close") / pl.col("close").shift(period) - 1).alias("momentum_raw")
        )

        # Shift to prevent lookahead bias
        df = df.with_columns(
            pl.col("momentum_raw").shift(1).alias("momentum")
        )

        # Generate signals
        df = df.with_columns(
            pl.when(pl.col("momentum") >= long_thresh)
            .then(1)
            .when(pl.col("momentum") <= short_thresh)
            .then(-1)
            .otherwise(0)
            .alias("signal")
        )

    elif strategy_type == "rsi":
        # Use RSI with optimized parameters
        period = params.get("rsi_period", 14)
        overbought = params.get("rsi_overbought", 70)
        oversold = params.get("rsi_oversold", 30)

        # Compute RSI
        from ..features.indicators import rsi as compute_rsi
        df = compute_rsi(df, period=period)

        # Shift to prevent lookahead bias
        df = df.with_columns(
            pl.col("rsi").shift(1).alias("rsi_lagged")
        )

        # Generate signals
        df = df.with_columns(
            pl.when(pl.col("rsi_lagged") <= oversold)
            .then(1)
            .when(pl.col("rsi_lagged") >= overbought)
            .then(-1)
            .otherwise(0)
            .alias("signal")
        )

    elif strategy_type == "ma_crossover":
        # Moving average crossover with optimized periods
        fast_period = params.get("fast_ma_period", 10)
        slow_period = params.get("slow_ma_period", 20)

        # Compute MAs
        from ..features.indicators import sma
        df = sma(df, fast_period)
        df = sma(df, slow_period)

        # Shift to prevent lookahead bias
        df = df.with_columns([
            pl.col(f"sma_{fast_period}").shift(1).alias(f"sma_{fast_period}_lagged"),
            pl.col(f"sma_{slow_period}").shift(1).alias(f"sma_{slow_period}_lagged"),
        ])

        # Generate crossover signals
        df = df.with_columns([
            (pl.col(f"sma_{fast_period}_lagged") > pl.col(f"sma_{slow_period}_lagged")).alias("bullish"),
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
        df = df.drop(["bullish", "bullish_diff", f"sma_{fast_period}_lagged", f"sma_{slow_period}_lagged"])

    else:
        # Default: momentum strategy
        period = params.get("momentum_period", 20)
        long_thresh = params.get("long_threshold", 0.02)
        short_thresh = params.get("short_threshold", -0.02)

        df = df.with_columns(
            (pl.col("close") / pl.col("close").shift(period) - 1).alias("momentum_raw")
        )
        df = df.with_columns(
            pl.col("momentum_raw").shift(1).alias("momentum")
        )
        df = df.with_columns(
            pl.when(pl.col("momentum") >= long_thresh)
            .then(1)
            .when(pl.col("momentum") <= short_thresh)
            .then(-1)
            .otherwise(0)
            .alias("signal")
        )

    return df


def _calculate_composite_score(
    metrics: Dict[str, float],
    weights: Dict[str, float],
) -> float:
    """
    Calculate composite score from metrics.

    Args:
        metrics: Dictionary of calculated metrics
        weights: Weights for each metric component

    Returns:
        Composite score
    """
    score = 0.0

    # Sharpe ratio component
    if "sharpe_ratio" in metrics:
        sharpe = metrics["sharpe_ratio"]
        # Normalize extreme values
        sharpe = min(max(sharpe, -5.0), 5.0)
        score += weights.get("sharpe_ratio", 0) * sharpe

    # Calmar ratio component
    if "calmar_ratio" in metrics:
        calmar = metrics["calmar_ratio"]
        # Normalize extreme values
        calmar = min(max(calmar, -5.0), 5.0)
        score += weights.get("calmar_ratio", 0) * calmar

    # Turnover penalty
    if "turnover" in metrics:
        turnover = metrics["turnover"]
        # Higher turnover gets penalized
        turnover_penalty = min(turnover, 5.0)  # Cap penalty
        score += weights.get("turnover_penalty", 0) * turnover_penalty

    # Overfit penalty (could be based on in-sample vs out-sample difference)
    # For now, using a simple complexity penalty
    complexity_penalty = 0.0  # Placeholder
    score += weights.get("overfit_penalty", 0) * complexity_penalty

    return score


def momentum_objective(
    params: Dict[str, Any],
    data_df: pl.DataFrame,
) -> float:
    """
    Specific objective function for momentum strategies.

    Args:
        params: Momentum strategy parameters
        data_df: Input data

    Returns:
        Score
    """
    # Extract parameters
    period = params.get("period", 20)
    long_thresh = params.get("long_threshold", 0.02)
    short_thresh = params.get("short_threshold", -0.02)

    # Generate signals
    df = data_df.clone()
    df = df.with_columns(
        (pl.col("close") / pl.col("close").shift(period) - 1).alias("momentum_raw")
    )
    df = df.with_columns(
        pl.col("momentum_raw").shift(1).alias("momentum")
    )
    df = df.with_columns(
        pl.when(pl.col("momentum") >= long_thresh)
        .then(1)
        .when(pl.col("momentum") <= short_thresh)
        .then(-1)
        .otherwise(0)
        .alias("signal")
    )

    # Run simplified backtest
    signals = df["signal"].to_numpy()
    prices = df["close"].to_numpy()

    # Calculate simple returns
    position = np.zeros(len(signals))
    position[1:] = signals[:-1]  # Shift signals to avoid lookahead bias
    returns = np.diff(np.log(prices)) * position[1:]

    if len(returns) == 0:
        return float('-inf')

    # Calculate metrics
    total_return = np.sum(returns)
    volatility = np.std(returns) if len(returns) > 1 else 0
    sharpe = total_return / volatility if volatility > 0 else 0

    # Calculate max drawdown
    equity = np.concatenate([[1], np.exp(np.cumsum(returns))])
    cummax = np.maximum.accumulate(equity)
    drawdown = (equity - cummax) / cummax
    max_dd = np.min(drawdown)

    # Composite score
    calmar = total_return / abs(max_dd) if max_dd != 0 else 0
    score = 0.6 * sharpe + 0.6 * calmar

    return score
