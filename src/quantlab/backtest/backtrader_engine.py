"""
Backtrader accurate backtest engine
"""
from datetime import datetime
from typing import Optional

import backtrader as bt
import numpy as np
import pandas as pd
import polars as pl

from ..common.logging import get_logger

logger = get_logger(__name__)


class BacktraderStrategy(bt.Strategy):
    """Backtrader strategy wrapper."""

    params = (
        ('signals', []),
        ('initial_capital', 100000),
    )

    def __init__(self):
        self.signals = self.params.signals
        self.i = 0
        self.order = None
        self.buy_price = None

    def next(self):
        if self.i >= len(self.signals):
            return

        # Get current signal
        signal = self.signals[self.i]

        # Check if there's a pending order
        if self.order:
            return

        # Current position
        position = self.getposition()

        # Execute based on signal
        if signal == 1 and not position:  # Buy signal
            self.order = self.buy()
        elif signal == -1 and position:  # Sell signal
            self.order = self.sell()

        self.i += 1


class BacktraderBacktestEngine:
    """
    Accurate backtest engine using Backtrader.

    Provides more detailed and accurate simulation than VectorBT.
    """

    def __init__(self, spec: dict):
        """
        Initialize Backtrader backtest engine.

        Args:
            spec: Strategy specification
        """
        self.spec = spec

    def run_backtest(
        self,
        df: pl.DataFrame,
        signals: np.ndarray,
        initial_capital: float,
    ) -> dict:
        """
        Run backtest using Backtrader.

        Args:
            df: OHLCV dataframe
            signals: Signal array (-1, 0, 1) for each timestamp
            initial_capital: Starting capital

        Returns:
            Dictionary with results (metrics, equity_curve, trades)
        """
        logger.info("backtrader_backtest_start", symbols=df["symbol"].unique().len())

        # Convert Polars to Pandas for Backtrader
        data_df = df.to_pandas()

        # Create cerebro engine
        cerebro = bt.Cerebro()

        # Set initial capital
        cerebro.broker.setcash(initial_capital)

        # Add data feed
        data_feed = self._create_data_feed(data_df)
        cerebro.adddata(data_feed)

        # Add strategy
        cerebro.addstrategy(
            BacktraderStrategy,
            signals=signals,
            initial_capital=initial_capital
        )

        # Add observers
        cerebro.addobserver(bt.observers.Value)
        cerebro.addobserver(bt.observers.BuySell)

        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='tradeanalyzer')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

        # Run backtest
        results = cerebro.run()

        # Extract results
        strat = results[0]

        # Get portfolio value over time
        equity_curve = self._extract_equity_curve(cerebro)

        # Get trades
        trades = self._extract_trades(strat)

        # Calculate metrics
        metrics = self._calculate_metrics(strat)

        return {
            "metrics": metrics,
            "equity_curve": equity_curve,
            "trades": trades,
        }

    def _create_data_feed(self, df: pd.DataFrame) -> bt.feeds.PandasData:
        """Create Backtrader data feed from dataframe."""
        return bt.feeds.PandasData(
            dataname=df,
            datetime='ts_utc',
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume',
            openinterest=-1  # Not used
        )

    def _extract_equity_curve(self, cerebro: bt.Cerebro) -> pl.DataFrame:
        """Extract equity curve from Backtrader."""
        # Get observer values
        value_observer = cerebro.observers._value
        values = value_observer.lines[0].array

        # Get datetime values
        # This is simplified - in practice, you'd need to extract the datetime index
        # from the data feed to align with the equity values

        # For now, return simplified version
        return pl.DataFrame({
            "equity": values,
            "ts_utc": range(len(values))  # Placeholder
        })

    def _extract_trades(self, strategy) -> pl.DataFrame:
        """Extract trades from Backtrader strategy."""
        # This would involve traversing the trade objects
        # For now, return empty dataframe
        return pl.DataFrame({
            "entry_time": [],
            "exit_time": [],
            "size": [],
            "pnl": [],
            "pnl_pct": []
        })

    def _calculate_metrics(self, strategy) -> dict:
        """Calculate metrics from Backtrader analyzers."""
        # Extract from analyzers
        trade_analyzer = strategy.analyzers.tradeanalyzer.get_analysis()
        sharpe = strategy.analyzers.sharpe.get_analysis()
        drawdown = strategy.analyzers.drawdown.get_analysis()

        metrics = {}

        # Win rate
        if 'won' in trade_analyzer and 'total' in trade_analyzer:
            won = trade_analyzer['won']['total'] if 'won' in trade_analyzer else 0
            total = trade_analyzer['total']['total'] if 'total' in trade_analyzer else 1
            metrics['win_rate'] = won / total if total > 0 else 0.0

        # Sharpe ratio
        metrics['sharpe_ratio'] = sharpe['sharperatio'] if 'sharperatio' in sharpe else 0.0

        # Max drawdown
        metrics['max_drawdown'] = drawdown['max']['drawdown'] if 'max' in drawdown else 0.0

        # Other metrics as placeholders
        metrics.update({
            "total_return": 0.0,
            "cagr": 0.0,
            "calmar_ratio": 0.0,
            "profit_factor": 0.0,
            "turnover": 0.0,
            "exposure": 0.0,
            "dd_duration": 0,
        })

        return metrics
