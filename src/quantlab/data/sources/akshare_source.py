"""
AkShare data source for Chinese stocks
"""
from datetime import datetime, timedelta
from typing import Optional

import akshare as ak
import polars as pl

from .base import DataSource
from ...common.logging import get_logger
from ...common.timeutils import to_utc

logger = get_logger(__name__)


class AkShareSource(DataSource):
    """AkShare data source for Chinese equities."""

    def __init__(self, venue: str = "sina"):
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
        Fetch bars from AkShare.

        Args:
            symbol: Stock symbol (e.g., "000001" or "000001.SZ")
            start: Start datetime (UTC)
            end: End datetime (UTC)
            freq: Frequency
            **kwargs: Additional parameters

        Returns:
            Normalized dataframe
        """
        # Convert symbol format if needed
        if "." in symbol:
            symbol_code = symbol.split(".")[0]
        else:
            symbol_code = symbol

        # AkShare expects dates in YYYYMMDD format
        start_date = start.strftime("%Y%m%d")
        end_date = end.strftime("%Y%m%d")

        logger.info("fetch_start", symbol=symbol, start=start_date, end=end_date, freq=freq)

        try:
            # Fetch data using stock_zh_a_hist
            df = ak.stock_zh_a_hist(
                symbol=symbol_code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=""
            )

            if df.empty:
                logger.warning("no_data", symbol=symbol)
                return pl.DataFrame(schema=self._get_schema())

            # Convert to Polars and normalize
            return self.normalize(pl.from_pandas(df), symbol)

        except Exception as e:
            logger.error("fetch_error", symbol=symbol, error=str(e))
            return pl.DataFrame(schema=self._get_schema())

    def fetch_symbols(self) -> list[str]:
        """Fetch available symbols."""
        try:
            # Fetch A-share stock list
            df = ak.stock_zh_a_spot_em()
            return df["代码"].to_list()
        except Exception as e:
            logger.error("symbols_fetch_error", error=str(e))
            return []

    def normalize(self, df: pl.DataFrame, symbol: str) -> pl.DataFrame:
        """
        Normalize AkShare dataframe.

        Args:
            df: Raw dataframe
            symbol: Symbol string

        Returns:
            Normalized dataframe
        """
        # Map AkShare columns to standard names
        column_map = {
            "日期": "ts_utc",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
            "成交额": "turnover",
        }

        # Rename columns
        for old_name, new_name in column_map.items():
            if old_name in df.columns:
                df = df.rename({old_name: new_name})

        # Ensure required columns exist
        schema = self._get_schema()
        for col in schema.keys():
            if col not in df.columns:
                df = df.with_columns(pl.lit(None).cast(schema[col]).alias(col))

        # Convert timestamp to UTC
        if "ts_utc" in df.columns:
            # AkShare dates are naive, assume Shanghai timezone
            df = df.with_columns(
                pl.col("ts_utc")
                .str.strptime(pl.Date, "%Y-%m-%d")
                .cast(pl.Datetime)
                .dt.replace_time_zone("Asia/Shanghai")
                .dt.convert_time_zone("UTC")
                .alias("ts_utc")
            )

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
