"""
Disk cache with smart key management
"""
import diskcache as dc
from pathlib import Path
from typing import Any, Optional

from .hashing import create_run_id, hash_spec, hash_files


class FeatureCache:
    """
    Cache for feature matrices and intermediate results.

    Cache key = hash(spec + data_version + code_version + stage_name)
    """

    def __init__(self, cache_dir: Path, run_id: Optional[str] = None):
        """
        Initialize feature cache.

        Args:
            cache_dir: Cache directory
            run_id: Run ID for namespacing
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or "default"
        self.cache = dc.Cache(str(cache_dir / self.run_id))

    def _make_key(
        self,
        stage: str,
        spec_hash: str,
        data_version: str,
        code_version: str,
        extra: str = "",
    ) -> str:
        """
        Create cache key.

        Args:
            stage: Stage name
            spec_hash: Strategy spec hash
            data_version: Data version hash
            code_version: Code version hash
            extra: Additional key suffix

        Returns:
            Cache key string
        """
        key_parts = [stage, spec_hash, data_version, code_version]
        if extra:
            key_parts.append(extra)
        return ":".join(key_parts)

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        return self.cache.get(key)

    def set(self, key: str, value: Any, expire: Optional[int] = None) -> None:
        """Set value in cache."""
        self.cache.set(key, value, expire=expire)

    def get_or_compute(
        self,
        key: str,
        compute_fn,
        expire: Optional[int] = None,
    ) -> Any:
        """
        Get from cache or compute and store.

        Args:
            key: Cache key
            compute_fn: Function to compute value if not cached
            expire: Expiration time in seconds

        Returns:
            Cached or computed value
        """
        value = self.get(key)
        if value is not None:
            return value

        value = compute_fn()
        self.set(key, value, expire=expire)
        return value

    def get_features(
        self,
        spec_hash: str,
        data_version: str,
        code_version: str,
        stage: str = "features",
    ) -> Optional[Any]:
        """
        Get cached features.

        Args:
            spec_hash: Strategy spec hash
            data_version: Data version hash
            code_version: Code version hash
            stage: Stage name

        Returns:
            Cached features or None
        """
        key = self._make_key(stage, spec_hash, data_version, code_version)
        return self.get(key)

    def set_features(
        self,
        features: Any,
        spec_hash: str,
        data_version: str,
        code_version: str,
        stage: str = "features",
        expire: Optional[int] = None,
    ) -> None:
        """
        Cache features.

        Args:
            features: Features to cache
            spec_hash: Strategy spec hash
            data_version: Data version hash
            code_version: Code version hash
            stage: Stage name
            expire: Expiration time
        """
        key = self._make_key(stage, spec_hash, data_version, code_version)
        self.set(key, features, expire=expire)

    def clear(self) -> None:
        """Clear all cached values."""
        self.cache.clear()

    def clear_stage(self, stage: str) -> None:
        """Clear all values for a stage."""
        for key in list(self.cache.iterkeys()):
            if isinstance(key, str) and key.startswith(f"{stage}:"):
                self.cache.delete(key)

    def close(self) -> None:
        """Close cache."""
        self.cache.close()
