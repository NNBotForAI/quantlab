"""
Backtest metrics calculation
"""
import numpy as np
import polars as pl

from ..common.logging import get_logger

logger = get_logger(__name__)


def calculate_all_metrics(
    equity_curve: pl.DataFrame,
    trades: pl.DataFrame,
    initial_capital: float,
    risk_free_rate: float = 0.02,
) -> dict:
    """
    Calculate comprehensive backtest metrics.

    Args:
        equity_curve: Equity curve dataframe
        trades: Trades dataframe
        initial_capital: Starting capital
        risk_free_rate: Risk-free rate for Sharpe

    Returns:
        Dictionary of metrics
    """
    equity = equity_curve["equity"].to_numpy()

    metrics = {
        # Return metrics
        "total_return": calculate_total_return(equity, initial_capital),
        "cagr": calculate_cagr(equity, initial_capital),
        "monthly_return": calculate_monthly_return(equity),

        # Risk metrics
        "sharpe_ratio": calculate_sharpe_ratio(equity, risk_free_rate),
        "sortino_ratio": calculate_sortino_ratio(equity, risk_free_rate),
        "max_drawdown": calculate_max_drawdown(equity),
        "calmar_ratio": calculate_calmar_ratio(equity, initial_capital),
        "avg_drawdown": calculate_avg_drawdown(equity),

        # Trade metrics
        "win_rate": calculate_win_rate(trades),
        "profit_factor": calculate_profit_factor(trades),
        "avg_win": calculate_avg_win(trades),
        "avg_loss": calculate_avg_loss(trades),
        "max_win": calculate_max_win(trades),
        "max_loss": calculate_max_loss(trades),

        # Activity metrics
        "total_trades": len(trades),
        "turnover": calculate_turnover(trades, initial_capital),
        "exposure": calculate_exposure(equity),
        "dd_duration": calculate_max_dd_duration(equity),
    }

    return metrics


def calculate_total_return(equity: np.ndarray, initial_capital: float) -> float:
    """Calculate total return."""
    return (equity[-1] / initial_capital) - 1


def calculate_cagr(equity: np.ndarray, initial_capital: float) -> float:
    """Calculate compound annual growth rate."""
    n_years = len(equity) / 252  # Assume 252 trading days/year
    if n_years == 0:
        return 0.0
    return (equity[-1] / initial_capital) ** (1 / n_years) - 1


def calculate_monthly_return(equity: np.ndarray) -> float:
    """Calculate average monthly return."""
    returns = np.diff(equity) / equity[:-1]
    monthly_returns = []
    for i in range(0, len(returns), 22):  # ~22 trading days/month
        if i + 22 < len(returns):
            monthly_returns.append(np.prod(1 + returns[i:i+22]) - 1)
    return np.mean(monthly_returns) if monthly_returns else 0.0


def calculate_sharpe_ratio(equity: np.ndarray, risk_free_rate: float) -> float:
    """Calculate Sharpe ratio."""
    returns = np.diff(equity) / equity[:-1]
    excess_returns = returns - (risk_free_rate / 252)
    if np.std(excess_returns) == 0:
        return 0.0
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)


def calculate_sortino_ratio(equity: np.ndarray, risk_free_rate: float) -> float:
    """Calculate Sortino ratio."""
    returns = np.diff(equity) / equity[:-1]
    excess_returns = returns - (risk_free_rate / 252)
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0 or np.std(downside_returns) == 0:
        return 0.0

    return np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)


def calculate_max_drawdown(equity: np.ndarray) -> float:
    """Calculate maximum drawdown."""
    cummax = np.maximum.accumulate(equity)
    drawdown = (equity - cummax) / cummax
    return drawdown.min()


def calculate_calmar_ratio(equity: np.ndarray, initial_capital: float) -> float:
    """Calculate Calmar ratio (CAGR / Max DD)."""
    cagr = calculate_cagr(equity, initial_capital)
    max_dd = calculate_max_drawdown(equity)

    if max_dd == 0:
        return 0.0

    return cagr / abs(max_dd)


def calculate_avg_drawdown(equity: np.ndarray) -> float:
    """Calculate average drawdown."""
    cummax = np.maximum.accumulate(equity)
    drawdown = (equity - cummax) / cummax
    return drawdown[drawdown < 0].mean()


def calculate_win_rate(trades: pl.DataFrame) -> float:
    """Calculate win rate."""
    if trades.is_empty() or "pnl" not in trades.columns:
        return 0.0

    winning_trades = trades.filter(pl.col("pnl") > 0)
    return len(winning_trades) / len(trades)


def calculate_profit_factor(trades: pl.DataFrame) -> float:
    """Calculate profit factor."""
    if trades.is_empty() or "pnl" not in trades.columns:
        return 0.0

    gross_profit = trades.filter(pl.col("pnl") > 0)["pnl"].sum()
    gross_loss = abs(trades.filter(pl.col("pnl") < 0)["pnl"].sum())

    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0

    return gross_profit / gross_loss


def calculate_avg_win(trades: pl.DataFrame) -> float:
    """Calculate average winning trade."""
    if trades.is_empty() or "pnl" not in trades.columns:
        return 0.0

    winning_trades = trades.filter(pl.col("pnl") > 0)
    if winning_trades.is_empty():
        return 0.0

    return winning_trades["pnl"].mean()


def calculate_avg_loss(trades: pl.DataFrame) -> float:
    """Calculate average losing trade."""
    if trades.is_empty() or "pnl" not in trades.columns:
        return 0.0

    losing_trades = trades.filter(pl.col("pnl") < 0)
    if losing_trades.is_empty():
        return 0.0

    return losing_trades["pnl"].mean()


def calculate_max_win(trades: pl.DataFrame) -> float:
    """Calculate maximum winning trade."""
    if trades.is_empty() or "pnl" not in trades.columns:
        return 0.0

    winning_trades = trades.filter(pl.col("pnl") > 0)
    if winning_trades.is_empty():
        return 0.0

    return winning_trades["pnl"].max()


def calculate_max_loss(trades: pl.DataFrame) -> float:
    """Calculate maximum losing trade."""
    if trades.is_empty() or "pnl" not in trades.columns:
        return 0.0

    losing_trades = trades.filter(pl.col("pnl") < 0)
    if losing_trades.is_empty():
        return 0.0

    return losing_trades["pnl"].min()


def calculate_turnover(trades: pl.DataFrame, initial_capital: float) -> float:
    """Calculate turnover rate."""
    if trades.is_empty():
        return 0.0

    if "value" in trades.columns:
        total_value = trades["value"].sum()
    else:
        # Estimate from price and quantity
        if "price" in trades.columns and "quantity" in trades.columns:
            total_value = (trades["price"] * trades["quantity"].abs()).sum()
        else:
            return 0.0

    return total_value / initial_capital


def calculate_exposure(equity: np.ndarray) -> float:
    """Calculate average exposure."""
    # This is a simplified calculation
    # In production, calculate from position data
    return 0.5  # Placeholder


def calculate_max_dd_duration(equity: np.ndarray) -> int:
    """Calculate maximum drawdown duration in bars."""
    cummax = np.maximum.accumulate(equity)
    drawdown = equity - cummax

    max_dd_duration = 0
    current_duration = 0

    for dd in drawdown:
        if dd < 0:
            current_duration += 1
            max_dd_duration = max(max_dd_duration, current_duration)
        else:
            current_duration = 0

    return max_dd_duration
