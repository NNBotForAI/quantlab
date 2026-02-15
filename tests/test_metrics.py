"""
Backtest metrics calculation tests
"""
import numpy as np
import polars as pl
import pytest

from src.quantlab.backtest.metrics import (
    calculate_all_metrics,
    calculate_total_return,
    calculate_cagr,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_calmar_ratio,
    calculate_win_rate,
    calculate_profit_factor,
)


@pytest.fixture
def sample_equity_curve():
    """Sample equity curve for testing."""
    return pl.DataFrame({
        "equity": np.array([
            100000, 101000, 102000, 101500, 102500,
            103000, 102000, 103500, 104000, 105000
        ])
    })


@pytest.fixture
def sample_trades():
    """Sample trades for testing."""
    return pl.DataFrame({
        "pnl": np.array([1000, -500, 1500, -200, 300, -100, 800, -400])
    })


def test_calculate_total_return(sample_equity_curve):
    """Test total return calculation."""
    equity = sample_equity_curve["equity"].to_numpy()
    initial = 100000

    total_return = calculate_total_return(equity, initial)

    # Final: 105000, Initial: 100000
    # Return: 0.05 (5%)
    np.testing.assert_almost_equal(total_return, 0.05, decimal=4)


def test_calculate_cagr(sample_equity_curve):
    """Test CAGR calculation."""
    equity = sample_equity_curve["equity"].to_numpy()
    initial = 100000

    cagr = calculate_cagr(equity, initial)

    # Assume 10 days = 10/252 years
    # (105000/100000)^(252/10) - 1
    assert cagr > 0, "CAGR should be positive"
    assert cagr < 10, "CAGR should be reasonable"


def test_calculate_sharpe_ratio(sample_equity_curve):
    """Test Sharpe ratio calculation."""
    equity = sample_equity_curve["equity"].to_numpy()

    sharpe = calculate_sharpe_ratio(equity, risk_free_rate=0.02)

    # Should be a positive number
    assert sharpe > 0, "Sharpe ratio should be positive"
    assert not np.isnan(sharpe), "Sharpe ratio should not be NaN"


def test_calculate_max_drawdown(sample_equity_curve):
    """Test maximum drawdown calculation."""
    equity = sample_equity_curve["equity"].to_numpy()

    max_dd = calculate_max_drawdown(equity)

    # Max drawdown should be negative
    assert max_dd < 0, "Max drawdown should be negative"
    assert max_dd >= -1, "Max drawdown should not be -100%"


def test_calculate_calmar_ratio(sample_equity_curve):
    """Test Calmar ratio calculation."""
    equity = sample_equity_curve["equity"].to_numpy()
    initial = 100000

    cagr = calculate_cagr(equity, initial)
    max_dd = calculate_max_drawdown(equity)

    calmar = calculate_calmar_ratio(equity, initial)

    # Calmar = CAGR / |MaxDD|
    expected = cagr / abs(max_dd) if max_dd != 0 else 0
    np.testing.assert_almost_equal(calmar, expected, decimal=4)


def test_calculate_win_rate(sample_trades):
    """Test win rate calculation."""
    win_rate = calculate_win_rate(sample_trades)

    # Winning trades: 5 (1000, 1500, 300, 800)
    # Total trades: 8
    # Win rate: 5/8 = 0.625
    np.testing.assert_almost_equal(win_rate, 0.625, decimal=3)


def test_calculate_profit_factor(sample_trades):
    """Test profit factor calculation."""
    profit_factor = calculate_profit_factor(sample_trades)

    # Gross profit: 1000 + 1500 + 300 + 800 = 3600
    # Gross loss: 500 + 200 + 100 + 400 = 1200
    # Profit factor: 3600 / 1200 = 3.0
    np.testing.assert_almost_equal(profit_factor, 3.0, decimal=1)


def test_calculate_all_metrics(sample_equity_curve, sample_trades):
    """Test comprehensive metrics calculation."""
    initial = 100000

    metrics = calculate_all_metrics(sample_equity_curve, sample_trades, initial)

    # Check that all expected metrics are present
    expected_metrics = [
        "total_return",
        "cagr",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "calmar_ratio",
        "avg_drawdown",
        "win_rate",
        "profit_factor",
        "avg_win",
        "avg_loss",
        "max_win",
        "max_loss",
        "total_trades",
        "turnover",
        "exposure",
        "dd_duration",
    ]

    for metric in expected_metrics:
        assert metric in metrics, f"Missing metric: {metric}"

    # Check reasonable values
    assert metrics["total_return"] > 0, "Total return should be positive"
    assert 0 <= metrics["win_rate"] <= 1, "Win rate should be between 0 and 1"
    assert metrics["profit_factor"] > 0, "Profit factor should be positive"
    assert metrics["max_drawdown"] <= 0, "Max drawdown should be <= 0"


def test_empty_trades():
    """Test metrics with empty trades."""
    empty_trades = pl.DataFrame({"pnl": []})
    equity = pl.DataFrame({"equity": [100000, 101000]})

    metrics = calculate_all_metrics(equity, empty_trades, 100000)

    # Should handle empty trades gracefully
    assert metrics["total_trades"] == 0
    assert metrics["win_rate"] == 0.0
    assert metrics["profit_factor"] == 0.0
