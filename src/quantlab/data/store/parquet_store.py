"""
Parquet data store with DuckDB querying
"""
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import duckdb
import polars as pl

from ...common.logging import get_logger
from ...common.io import safe_path_join, atomic_write
from ...common.hashing import hash_file
from .metadata_sqlite import MetadataStore

logger = get_logger(__name__)


class ParquetStore:
    """
    Partitioned Parquet storage for OHLCV data.

    Partition structure: data/bars/market=.../venue=.../freq=.../symbol=.../part-*.parquet
    """

    def __init__(
        self,
        base_path: Path,
        metadata_db: Optional[Path] = None,
    ):
        """
        Initialize Parquet store.

        Args:
            base_path: Base storage path
            metadata_db: Path to metadata database (defaults to base_path/metadata.db)
        """
        self.base_path = base_path
        self.bars_path = base_path / "bars"
        self.bars_path.mkdir(parents=True, exist_ok=True)

        if metadata_db is None:
            metadata_db = base_path / "metadata.db"

        self.metadata = MetadataStore(metadata_db)

    def _get_partition_path(
        self,
        market: str,
        venue: str,
        freq: str,
        symbol: str,
    ) -> Path:
        """
        Get partition path for a symbol.

        Args:
            market: Market type
            venue: Venue name
            freq: Frequency
            symbol: Symbol

        Returns:
            Partition path
        """
        return safe_path_join(
            self.bars_path,
            f"market={market}",
            f"venue={venue}",
            f"freq={freq}",
            f"symbol={symbol}",
        )

    def write_bars(
        self,
        df: pl.DataFrame,
        market: str,
        venue: str,
        freq: str,
        symbol: Optional[str] = None,
        mode: str = "overwrite",
    ) -> None:
        """
        Write OHLCV bars to parquet.

        Args:
            df: OHLCV dataframe with columns: ts_utc, symbol, open, high, low, close, volume,
                 currency, is_tradable, venue, market, freq
            market: Market type
            venue: Venue name
            freq: Frequency
            symbol: Symbol (optional, will use df.symbol if not provided)
            mode: Write mode ("overwrite" or "append")
        """
        # Ensure required columns
        required_cols = [
            "ts_utc", "symbol", "open", "high", "low", "close", "volume",
            "currency", "is_tradable", "venue", "market", "freq"
        ]

        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Fill missing symbol
        if symbol and "symbol" not in df.columns:
            df = df.with_columns(pl.lit(symbol).cast(pl.String).alias("symbol"))

        # Get unique symbols
        symbols = df["symbol"].unique().to_list()

        for sym in symbols:
            sym_df = df.filter(pl.col("symbol") == sym)

            partition_path = self._get_partition_path(market, venue, freq, sym)
            partition_path.mkdir(parents=True, exist_ok=True)

            # Write parquet
            file_path = partition_path / "part-0.parquet"

            if mode == "overwrite" or not file_path.exists():
                atomic_write(file_path, sym_df.write_parquet(None).to_buffer())
            else:
                # Append: read existing, combine, write back
                existing_df = pl.read_parquet(file_path)
                combined_df = pl.concat([existing_df, sym_df])
                atomic_write(file_path, combined_df.write_parquet(None).to_buffer())

            # Save metadata
            file_hash = hash_file(file_path)
            self.metadata.save_partition_metadata(
                market=market,
                venue=venue,
                freq=freq,
                symbol=sym,
                partition_path=str(partition_path),
                file_hash=file_hash,
            )

        logger.info("bars_written", symbols=symbols, market=market, venue=venue, freq=freq)

    def read_bars(
        self,
        market: str,
        venue: str,
        freq: str,
        symbol: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pl.DataFrame:
        """
        Read OHLCV bars from parquet.

        Args:
            market: Market type
            venue: Venue name
            freq: Frequency
            symbol: Symbol (optional, read all symbols if None)
            start: Start datetime (UTC)
            end: End datetime (UTC)

        Returns:
            OHLCV dataframe
        """
        if symbol:
            partition_path = self._get_partition_path(market, venue, freq, symbol)
            parquet_path = partition_path / "part-0.parquet"

            if not parquet_path.exists():
                logger.warning("partition_not_found", symbol=symbol, path=str(parquet_path))
                return pl.DataFrame()

            df = pl.read_parquet(parquet_path)
        else:
            # Read all symbols for market/venue/freq
            pattern = str(self._get_partition_path(market, venue, freq, "*") / "*.parquet")
            try:
                df = pl.read_parquet(pattern)
            except Exception as e:
                logger.warning("no_partitions_found", pattern=pattern, error=str(e))
                return pl.DataFrame()

        # Filter by time range
        if start:
            df = df.filter(pl.col("ts_utc") >= start)
        if end:
            df = df.filter(pl.col("ts_utc") <= end)

        return df

    def query(
        self,
        sql: str,
        params: Optional[dict] = None,
    ) -> pl.DataFrame:
        """
        Query data using DuckDB (scans parquet directly).

        Args:
            sql: SQL query string
            params: Query parameters

        Returns:
            Query results as Polars DataFrame
        """
        con = duckdb.connect()

        if params is None:
            params = {}

        result = con.execute(sql, params).pl()
        con.close()

        return result

    def compute_data_version(
        self,
        market: str,
        venue: str,
        freq: str,
    ) -> Optional[str]:
        """
        Compute data version hash from partitions.

        Args:
            market: Market type
            venue: Venue name
            freq: Frequency

        Returns:
            Data version hash
        """
        return self.metadata.compute_data_version(market, venue, freq)

    def get_available_symbols(
        self,
        market: str,
        venue: str,
        freq: str,
    ) -> list[str]:
        """
        Get list of available symbols.

        Args:
            market: Market type
            venue: Venue name
            freq: Frequency

        Returns:
            List of symbols
        """
        partitions = self.metadata.get_partition_metadata(market, venue, freq)
        return [p["symbol"] for p in partitions]

    def delete_partition(
        self,
        market: str,
        venue: str,
        freq: str,
        symbol: str,
    ) -> None:
        """
        Delete a partition.

        Args:
            market: Market type
            venue: Venue name
            freq: Frequency
            symbol: Symbol
        """
        partition_path = self._get_partition_path(market, venue, freq, symbol)

        # Delete partition directory
        if partition_path.exists():
            import shutil
            shutil.rmtree(partition_path)

        # Delete from metadata
        with self.metadata.metadata.db_path as conn:
            conn.execute(
                """
                DELETE FROM partition_metadata
                WHERE market = ? AND venue = ? AND freq = ? AND symbol = ?
                """,
                (market, venue, freq, symbol)
            )
            conn.commit()

        logger.info("partition_deleted", symbol=symbol, path=str(partition_path))
