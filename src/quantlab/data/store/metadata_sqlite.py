"""
SQLite metadata store
"""
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from ...common.logging import get_logger

logger = get_logger(__name__)


class MetadataStore:
    """SQLite-based metadata storage."""

    def __init__(self, db_path: Path):
        """
        Initialize metadata store.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS data_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version_hash TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    description TEXT,
                    partitions TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS partition_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market TEXT NOT NULL,
                    venue TEXT NOT NULL,
                    freq TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    partition_path TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(market, venue, freq, symbol)
                )
            """)

            conn.commit()

    def save_data_version(
        self,
        version_hash: str,
        description: Optional[str] = None,
        partitions: Optional[list[str]] = None,
    ) -> None:
        """
        Save data version.

        Args:
            version_hash: Version hash
            description: Version description
            partitions: List of partition paths
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO data_versions (version_hash, created_at, description, partitions)
                VALUES (?, ?, ?, ?)
                """,
                (version_hash, datetime.utcnow().isoformat(), description, ",".join(partitions or []))
            )
            conn.commit()

        logger.info("data_version_saved", version=version_hash)

    def get_latest_data_version(self) -> Optional[str]:
        """Get latest data version hash."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT version_hash FROM data_versions ORDER BY created_at DESC LIMIT 1"
            )
            row = cursor.fetchone()
            return row[0] if row else None

    def save_partition_metadata(
        self,
        market: str,
        venue: str,
        freq: str,
        symbol: str,
        partition_path: str,
        file_hash: str,
    ) -> None:
        """
        Save partition metadata.

        Args:
            market: Market type
            venue: Venue name
            freq: Frequency
            symbol: Symbol
            partition_path: Partition path
            file_hash: File hash
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO partition_metadata
                (market, venue, freq, symbol, partition_path, file_hash, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (market, venue, freq, symbol, partition_path, file_hash, datetime.utcnow().isoformat())
            )
            conn.commit()

    def get_partition_metadata(
        self,
        market: str,
        venue: str,
        freq: str,
        symbol: Optional[str] = None,
    ) -> list[dict]:
        """
        Get partition metadata.

        Args:
            market: Market type
            venue: Venue name
            freq: Frequency
            symbol: Symbol (optional)

        Returns:
            List of partition metadata dicts
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            if symbol:
                cursor = conn.execute(
                    """
                    SELECT * FROM partition_metadata
                    WHERE market = ? AND venue = ? AND freq = ? AND symbol = ?
                    """,
                    (market, venue, freq, symbol)
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM partition_metadata
                    WHERE market = ? AND venue = ? AND freq = ?
                    """,
                    (market, venue, freq)
                )

            return [dict(row) for row in cursor.fetchall()]

    def compute_data_version(
        self,
        market: str,
        venue: str,
        freq: str,
    ) -> Optional[str]:
        """
        Compute data version hash from partition metadata.

        Args:
            market: Market type
            venue: Venue name
            freq: Frequency

        Returns:
            Data version hash
        """
        partitions = self.get_partition_metadata(market, venue, freq)

        if not partitions:
            return None

        # Combine all file hashes
        import hashlib

        sha256 = hashlib.sha256()
        for p in sorted(partitions, key=lambda x: x["symbol"]):
            sha256.update(p["file_hash"].encode())

        return sha256.hexdigest()
