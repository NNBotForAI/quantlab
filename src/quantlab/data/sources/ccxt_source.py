"""
CCXT data source for crypto
"""
from datetime import datetime, timedelta
from typing import Optional

import ccxt
import polars as pl

from .base import DataSource
from ...common.logging import get_logger
from ...common.timeutils import to_utc

logger = get_logger(__name__)


class CCXTSource(DataSource):
    """CCXT data source for crypto exchanges."""

    def __init__(self, venue: str = "binance"):
        super().__init__(venue)
        # Initialize CCXT exchange
        exchange_class = getattr(ccxt, venue)
        self.exchange = exchange_class({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'},
        })

    def fetch_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        freq: str,
        **kwargs,
    ) -> pl.DataFrame:
        """
        Fetch bars from CCXT exchange.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            start: Start datetime (UTC)
            end: End datetime (UTC)
            freq: Frequency
            **kwargs: Additional parameters

        Returns:
            Normalized dataframe
        """
        # Map frequency to CCXT timeframe
        timeframe_map = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "1H": "1h",
            "1D": "1d",
        }
        timeframe = timeframe_map.get(freq, "1h")

        # Convert to timestamps (milliseconds)
        start_ts = int(start.timestamp() * 1000)
        end_ts = int(end.timestamp() * 1000)

        logger.info("fetch_start", symbol=symbol, start=start, end=end, timeframe=timeframe)

        try:
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=start_ts,
                params={'endTime': end_ts}
            )

            if not ohlcv:
                logger.warning("no_data", symbol=symbol)
                return pl.DataFrame(schema=self._get_schema())

            # Convert to dataframe
            df = pl.DataFrame(ohlcv, schema=[
                "timestamp", "open", "high", "low", "close", "volume"
            ])

            return self.normalize(df, symbol)

        except Exception as e:
            logger.error("fetch_error", symbol=symbol, error=str(e))
            return pl.DataFrame(schema=self._get_schema())

    def fetch_symbols(self) -> list[str]:
        """Fetch available trading pairs."""
        try:
            markets = self.exchange.load_markets()
            return list(markets.keys())
        except Exception as e:
            logger.error("symbols_fetch_error", error=str(e))
            return []

    def normalize(self, df: pl.DataFrame, symbol: str) -> pl.DataFrame:
        """
        Normalize CCXT dataframe.

        Args:
            df: Raw dataframe
            symbol: Symbol string

        Returns:
            Normalized dataframe
        """
        # Convert timestamp from milliseconds to datetime
        df = df.with_columns(
            (pl.col("timestamp") / 1000).cast(pl.Int64)
            .cast(pl.Datetime(time_unit="s", time_zone="UTC"))
            .alias("ts_utc")
        )

        # Drop original timestamp column
        df = df.drop("timestamp")

        # Add symbol column
        df = df.with_columns(pl.lit(symbol).alias("symbol"))

        # Select required columns
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
