"""
VectorBT fast backtest engine with chunking
"""
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import vectorbt as vbt

from ..common.logging import get_logger
from ..common.perf import measure_time
from ..execution.base import ExecutionModel
from ..execution.cn_equity import CNEquityExecutionModel
from ..execution.us_equity import USEquityExecutionModel
from ..execution.crypto_spot import CryptoSpotExecutionModel
from ..execution.crypto_perp import CryptoPerpExecutionModel

logger = get_logger(__name__)


class VectorBTBacktestEngine:
    """
    Fast backtest engine using VectorBT.

    Supports chunking for memory control and parallel execution.
    """

    def __init__(self, spec: dict, execution_model: Optional[ExecutionModel] = None):
        """
        Initialize VectorBT backtest engine.

        Args:
            spec: Strategy specification
            execution_model: Execution model (auto-created if None)
        """
        self.spec = spec
        self.asset_type = spec["instrument"]["asset_type"]

        # Create execution model
        if execution_model is None:
            execution_model = self._create_execution_model(spec)
        self.execution = execution_model

        # Load chunking settings
        self.chunk_size = spec["performance"].get("vectorbt_chunking", 200)

    def _create_execution_model(self, spec: dict) -> ExecutionModel:
        """Create execution model based on asset type."""
        asset_type = spec["instrument"]["asset_type"]

        if asset_type == "CN_STOCK":
            return CNEquityExecutionModel(spec)
        elif asset_type == "US_STOCK":
            return USEquityExecutionModel(spec)
        elif asset_type == "CRYPTO_SPOT":
            return CryptoSpotExecutionModel(spec)
        elif asset_type == "CRYPTO_PERP":
            return CryptoPerpExecutionModel(spec)
        else:
            raise ValueError(f"Unknown asset type: {asset_type}")

    def run_backtest(
        self,
        df: pl.DataFrame,
        signals: np.ndarray,
        initial_capital: float,
    ) -> dict:
        """
        Run backtest using VectorBT.

        Args:
            df: OHLCV dataframe (must include close and volume)
            signals: Signal array (-1, 0, 1) for each timestamp
            initial_capital: Starting capital

        Returns:
            Dictionary with results (metrics, equity_curve, trades)
        """
        logger.info("backtest_start", symbols=df["symbol"].unique().len())

        with measure_time("vectorbt_backtest"):
            # Convert to numpy arrays for VectorBT
            close = df["close"].to_numpy()
            volume = df["volume"].to_numpy()

            # Apply execution constraints
            signals = self.execution.apply_shortability(signals)

            # Create portfolio using VectorBT
            portfolio = vbt.Portfolio.from_signals(
                close=close,
                entries=signals == 1,
                exits=signals == -1,
                init_cash=initial_capital,
                freq=self._get_vectorbt_freq(),
                fees=self._get_fees(),
                slippage=self._get_slippage(),
            )

            # Compute metrics
            metrics = self._compute_metrics(portfolio, initial_capital)

            # Get equity curve
            equity_curve = self._get_equity_curve(portfolio)

            # Get trades
            trades = self._get_trades(portfolio, df)

            return {
                "metrics": metrics,
                "equity_curve": equity_curve,
                "trades": trades,
            }

    def run_backtest_chunked(
        self,
        df: pl.DataFrame,
        signals: np.ndarray,
        initial_capital: float,
        chunk_size: Optional[int] = None,
    ) -> dict:
        """
        Run backtest with chunking for memory control.

        Args:
            df: OHLCV dataframe
            signals: Signal array
            initial_capital: Starting capital
            chunk_size: Symbols per chunk (defaults to self.chunk_size)

        Returns:
            Combined results from all chunks
        """
        if chunk_size is None:
            chunk_size = self.chunk_size

        symbols = df["symbol"].unique().to_list()

        if len(symbols) <= chunk_size:
            # No chunking needed
            return self.run_backtest(df, signals, initial_capital)

        logger.info("chunked_backtest", total_symbols=len(symbols), chunk_size=chunk_size)

        # Split into chunks
        all_equity_curves = []
        all_trades = []

        for i in range(0, len(symbols), chunk_size):
            chunk_symbols = symbols[i:i + chunk_size]
            logger.info("processing_chunk", chunk=i//chunk_size + 1, symbols=chunk_symbols)

            # Filter dataframe to chunk
            chunk_df = df.filter(pl.col("symbol").is_in(chunk_symbols))
            chunk_signals = signals[df["symbol"].is_in(chunk_symbols)]

            # Run backtest on chunk
            chunk_results = self.run_backtest(chunk_df, chunk_signals, initial_capital)

            all_equity_curves.append(chunk_results["equity_curve"])
            all_trades.append(chunk_results["trades"])

        # Combine results
        combined_equity = self._combine_equity_curves(all_equity_curves)
        combined_trades = self._combine_trades(all_trades)

        # Compute combined metrics
        combined_metrics = self._compute_metrics_from_equity(combined_equity, initial_capital)

        return {
            "metrics": combined_metrics,
            "equity_curve": combined_equity,
            "trades": combined_trades,
        }

    def _get_vectorbt_freq(self) -> str:
        """Get VectorBT frequency string."""
        freq = self.spec["data"]["frequency"]
        freq_map = {
            "1m": "1T",
            "5m": "5T",
            "15m": "15T",
            "1H": "1H",
            "1D": "1D",
        }
        return freq_map.get(freq, "1D")

    def _get_fees(self) -> float:
        """Get fee rate."""
        return self.execution.commission_bps / 10000

    def _get_slippage(self) -> float:
        """Get slippage rate."""
        return self.execution.slippage_bps / 10000

    def _compute_metrics(self, portfolio, initial_capital: float) -> dict:
        """Compute portfolio metrics."""
        return {
            "total_return": portfolio.total_return(),
            "cagr": portfolio.cagr(),
            "sharpe_ratio": portfolio.sharpe_ratio(),
            "max_drawdown": portfolio.max_drawdown(),
            "calmar_ratio": portfolio.calmar_ratio(),
            "win_rate": portfolio.trades.win_rate(),
            "profit_factor": portfolio.trades.profit_factor(),
            "turnover": portfolio.trades.total_value() / initial_capital,
            "exposure": portfolio.trades.exposure.mean(),
            "dd_duration": portfolio.max_drawdown_duration(),
        }

    def _compute_metrics_from_equity(self, equity_curve: pl.DataFrame, initial_capital: float) -> dict:
        """Compute metrics from equity curve (for chunked results)."""
        equity = equity_curve["equity"].to_numpy()

        # Total return
        total_return = (equity[-1] / equity[0]) - 1

        # CAGR (assuming daily data)
        n_years = len(equity) / 252
        cagr = (equity[-1] / equity[0]) ** (1 / n_years) - 1 if n_years > 0 else 0

        # Max drawdown
        cummax = np.maximum.accumulate(equity)
        drawdown = (equity - cummax) / cummax
        max_drawdown = drawdown.min()

        # Sharpe ratio (simplified)
        returns = np.diff(equity) / equity[:-1]
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

        # Calmar ratio
        calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0

        return {
            "total_return": float(total_return),
            "cagr": float(cagr),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "calmar_ratio": float(calmar),
            "turnover": 0.0,  # Placeholder
            "exposure": 0.0,  # Placeholder
            "dd_duration": 0,  # Placeholder
        }

    def _get_equity_curve(self, portfolio) -> pl.DataFrame:
        """Get equity curve as Polars dataframe."""
        equity = portfolio.value()
        return pl.DataFrame({
            "equity": equity,
            "ts_utc": portfolio.index(),
        })

    def _get_trades(self, portfolio, df: pl.DataFrame) -> pl.DataFrame:
        """Get trades as Polars dataframe."""
        trades = portfolio.trades.records_readable
        return pl.DataFrame(trades)

    def _combine_equity_curves(self, equity_curves: list[pl.DataFrame]) -> pl.DataFrame:
        """Combine equity curves from multiple chunks."""
        # Simple concatenation (in production, would align by timestamp)
        return pl.concat(equity_curves)

    def _combine_trades(self, trades_list: list[pl.DataFrame]) -> pl.DataFrame:
        """Combine trades from multiple chunks."""
        return pl.concat(trades_list)
