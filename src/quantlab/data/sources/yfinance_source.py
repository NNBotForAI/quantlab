"""
YFinance data source for US stocks
"""
from datetime import datetime, timedelta
from typing import Optional

import polars as pl
import yfinance as yf

from .base import DataSource
from ...common.logging import get_logger
from ...common.timeutils import to_utc

logger = get_logger(__name__)


class YFinanceSource(DataSource):
    """YFinance data source for US equities."""

    def __init__(self, venue: str = "yahoo"):
        super().__init__(venue)

    def fetch_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        freq: str,
        **kwargs,
    ) -> pl.DataFrame:
        """
        Fetch bars from YFinance.

        Args:
            symbol: Ticker symbol
            start: Start datetime (UTC)
            end: End datetime (UTC)
            freq: Frequency
            **kwargs: Additional parameters

        Returns:
            Normalized dataframe
        """
        # Map frequency to YFinance interval
        interval_map = {
            "1D": "1d",
            "1H": "1h",
            "5m": "5m",
            "15m": "15m",
        }
        interval = interval_map.get(freq, "1d")

        # YFinance expects timezone-aware dates
        start_naive = start.replace(tzinfo=None)
        end_naive = end.replace(tzinfo=None)

        logger.info("fetch_start", symbol=symbol, start=start_naive, end=end_naive, interval=interval)

        # Fetch data
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_naive, end=end_naive, interval=interval, auto_adjust=True)

        if df.empty:
            logger.warning("no_data", symbol=symbol)
            return pl.DataFrame(schema=self._get_schema())

        # Convert to Polars and normalize
        return self.normalize(pl.from_pandas(df.reset_index()))

    def fetch_symbols(self) -> list[str]:
        """Get list of available symbols (placeholder)."""
        # YFinance doesn't have a public API for listing symbols
        # In production, this would use a symbol list file or database
        logger.warning("symbols_placeholder", message="YFinance source requires pre-defined symbol list")
        return []

    def normalize(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Normalize YFinance dataframe.

        Args:
            df: Raw dataframe

        Returns:
            Normalized dataframe
        """
        # Map YFinance columns to standard names
        column_map = {
            "Date": "ts_utc",
            "Datetime": "ts_utc",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }

        # Rename columns
        for old_name, new_name in column_map.items():
            if old_name in df.columns:
                df = df.rename({old_name: new_name})

        # Ensure all required columns exist
        schema = self._get_schema()
        for col in schema.keys():
            if col not in df.columns:
                df = df.with_columns(pl.lit(None).cast(schema[col]).alias(col))

        # Convert timestamp to UTC
        if "ts_utc" in df.columns:
            df = df.with_columns(
                pl.col("ts_utc").dt.convert_time_zone("UTC").alias("ts_utc")
            )

        # Add symbol column (will be filled later)
        df = df.with_columns(pl.lit(None).cast(pl.String).alias("symbol"))

        return df.select([
            "ts_utc", "symbol", "open", "high", "low", "close", "volume"
        ])

    def _get_schema(self) -> dict:
        """Get standard OHLCV schema."""
        return {
            "ts_utc": pl.Datetime(time_zone="UTC"),
            "symbol": pl.String,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
        }
