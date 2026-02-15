"""
Data pipeline for fetching, normalizing, and storing market data
"""
from datetime import datetime
from pathlib import Path
from typing import Optional

import polars as pl

from .sources.base import DataSource
from .sources.yfinance_source import YFinanceSource
from .sources.akshare_source import AkShareSource
from .sources.ccxt_source import CCXTSource
from .normalize.adjustments import apply_adjustments, forward_fill_prices
from .normalize.calendar import get_trading_sessions
from .normalize.quality import check_ohlc_consistency, clean_data
from .store.parquet_store import ParquetStore
from ..common.logging import get_logger
from ..common.timeutils import to_utc
from ..common.perf import measure_time

logger = get_logger(__name__)


class DataPipeline:
    """
    Data pipeline for fetching, normalizing, and storing market data.
    """

    def __init__(
        self,
        data_dir: Path,
        spec: dict,
    ):
        """
        Initialize data pipeline.

        Args:
            data_dir: Data directory
            spec: Strategy specification
        """
        self.data_dir = data_dir
        self.spec = spec
        self.store = ParquetStore(data_dir)

        # Initialize data source based on spec
        self.source = self._create_source()

    def _create_source(self) -> DataSource:
        """Create data source from spec."""
        source_name = self.spec["data"]["source"]

        venue = self.spec["instrument"].get("venue", "default")

        if source_name == "yfinance":
            return YFinanceSource(venue)
        elif source_name == "akshare":
            return AkShareSource(venue)
        elif source_name == "ccxt":
            return CCXTSource(venue)
        else:
            raise ValueError(f"Unknown data source: {source_name}")

    def fetch_and_store(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        freq: str,
        forward_fill: bool = False,
    ) -> dict[str, str]:
        """
        Fetch and store data for multiple symbols.

        Args:
            symbols: List of symbols to fetch
            start: Start datetime (UTC)
            end: End datetime (UTC)
            freq: Frequency
            forward_fill: Whether to forward fill missing bars

        Returns:
            Dictionary of symbol -> status
        """
        results = {}

        for symbol in symbols:
            try:
                with measure_time(f"fetch_{symbol}"):
                    df = self.source.fetch_bars(symbol, start, end, freq)

                if df.is_empty():
                    results[symbol] = "no_data"
                    continue

                # Normalize and clean
                df = self._normalize(df, symbol)
                df = clean_data(df)

                if forward_fill:
                    df = forward_fill_prices(df)

                # Check quality
                df = check_ohlc_consistency(df)

                # Store
                market = self.spec["instrument"]["asset_type"]
                venue = self.spec["instrument"]["venue"]
                self.store.write_bars(df, market, venue, freq, mode="overwrite")

                results[symbol] = "success"

            except Exception as e:
                logger.error("fetch_error", symbol=symbol, error=str(e))
                results[symbol] = f"error: {str(e)}"

        return results

    def _normalize(
        self,
        df: pl.DataFrame,
        symbol: str,
    ) -> pl.DataFrame:
        """
        Normalize dataframe to standard format.

        Args:
            df: Raw dataframe
            symbol: Symbol

        Returns:
            Normalized dataframe
        """
        # Ensure required columns
        if "symbol" not in df.columns or df["symbol"].is_null().all():
            df = df.with_columns(pl.lit(symbol).alias("symbol"))

        # Add metadata columns
        df = df.with_columns([
            pl.lit(self.spec["instrument"]["quote_currency"]).alias("currency"),
            pl.lit(True).alias("is_tradable"),  # TODO: check for suspensions
            pl.lit(self.spec["instrument"]["venue"]).alias("venue"),
            pl.lit(self.spec["instrument"]["asset_type"]).alias("market"),
            pl.lit(self.spec["data"]["frequency"]).alias("freq"),
        ])

        # Apply price adjustments if needed
        price_mode = self.spec["data"].get("price_mode", "raw")
        df = apply_adjustments(df, price_mode)

        return df

    def get_data(
        self,
        symbols: Optional[list[str]] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pl.DataFrame:
        """
        Get stored data.

        Args:
            symbols: List of symbols (None = all)
            start: Start datetime
            end: End datetime

        Returns:
            Dataframe with OHLCV data
        """
        market = self.spec["instrument"]["asset_type"]
        venue = self.spec["instrument"]["venue"]
        freq = self.spec["data"]["frequency"]

        if symbols:
            dfs = []
            for symbol in symbols:
                df = self.store.read_bars(market, venue, freq, symbol, start, end)
                if not df.is_empty():
                    dfs.append(df)
            return pl.concat(dfs) if dfs else pl.DataFrame()
        else:
            return self.store.read_bars(market, venue, freq, None, start, end)

    def get_data_version(self) -> Optional[str]:
        """
        Get current data version hash.

        Returns:
            Data version hash
        """
        market = self.spec["instrument"]["asset_type"]
        venue = self.spec["instrument"]["venue"]
        freq = self.spec["data"]["frequency"]

        return self.store.compute_data_version(market, venue, freq)

    def query(self, sql: str, params: Optional[dict] = None) -> pl.DataFrame:
        """
        Query data using DuckDB.

        Args:
            sql: SQL query
            params: Query parameters

        Returns:
            Query results
        """
        return self.store.query(sql, params)
