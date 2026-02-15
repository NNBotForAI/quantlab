"""
Base data source interface
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import polars as pl


class DataSource(ABC):
    """Abstract base class for data sources."""

    def __init__(self, venue: str):
        """
        Initialize data source.

        Args:
            venue: Venue/exchange name
        """
        self.venue = venue

    @abstractmethod
    def fetch_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        freq: str,
        **kwargs,
    ) -> pl.DataFrame:
        """
        Fetch OHLCV bars.

        Args:
            symbol: Instrument symbol
            start: Start datetime (UTC)
            end: End datetime (UTC)
            freq: Frequency (1D, 1H, 5m, etc.)
            **kwargs: Additional parameters

        Returns:
            DataFrame with columns: ts_utc, symbol, open, high, low, close, volume
        """
        pass

    @abstractmethod
    def fetch_symbols(self) -> list[str]:
        """
        Fetch available symbols.

        Returns:
            List of symbol strings
        """
        pass

    @abstractmethod
    def normalize(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Normalize dataframe to standard format.

        Args:
            df: Raw dataframe

        Returns:
            Normalized dataframe with required columns
        """
        pass
