"""
Feature matrix caching and storage
"""
from datetime import datetime
from pathlib import Path
from typing import Optional

import polars as pl

from ..common.cache import FeatureCache
from ..common.hashing import hash_spec
from ..common.io import atomic_write, safe_path_join
from ..common.logging import get_logger
from ..common.perf import measure_time

logger = get_logger(__name__)


class FeatureStore:
    """
    Store and cache feature matrices.

    Feature matrices stored as parquet: results/{run_id}/features/*.parquet
    """

    def __init__(
        self,
        results_dir: Path,
        run_id: str,
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize feature store.

        Args:
            results_dir: Results directory
            run_id: Run ID
            cache_dir: Cache directory (defaults to results_dir/cache)
        """
        self.results_dir = results_dir
        self.run_id = run_id
        self.features_dir = results_dir / run_id / "features"
        self.features_dir.mkdir(parents=True, exist_ok=True)

        if cache_dir is None:
            cache_dir = results_dir / "cache"

        self.cache = FeatureCache(cache_dir, run_id=run_id)

    def get_features(
        self,
        spec: dict,
        data_version: str,
        code_version: str,
        stage: str = "features",
        use_cache: bool = True,
    ) -> Optional[pl.DataFrame]:
        """
        Get features from cache or storage.

        Args:
            spec: Strategy specification
            data_version: Data version hash
            code_version: Code version hash
            stage: Stage name
            use_cache: Whether to use cache

        Returns:
            Feature dataframe or None
        """
        spec_hash = hash_spec(spec)

        # Try cache first
        if use_cache:
            cached = self.cache.get_features(
                spec_hash=spec_hash,
                data_version=data_version,
                code_version=code_version,
                stage=stage,
            )
            if cached is not None:
                logger.info("features_cache_hit", stage=stage)
                return cached

        # Try storage
        feature_path = self.features_dir / f"{stage}.parquet"
        if feature_path.exists():
            logger.info("features_storage_load", stage=stage)
            return pl.read_parquet(feature_path)

        return None

    def save_features(
        self,
        df: pl.DataFrame,
        spec: dict,
        data_version: str,
        code_version: str,
        stage: str = "features",
        cache_expire: Optional[int] = None,
    ) -> None:
        """
        Save features to storage and cache.

        Args:
            df: Feature dataframe
            spec: Strategy specification
            data_version: Data version hash
            code_version: Code version hash
            stage: Stage name
            cache_expire: Cache expiration in seconds
        """
        spec_hash = hash_spec(spec)

        # Save to storage
        feature_path = self.features_dir / f"{stage}.parquet"
        atomic_write(feature_path, df.write_parquet(None).to_buffer())
        logger.info("features_saved", stage=stage, path=str(feature_path))

        # Save to cache
        self.cache.set_features(
            features=df,
            spec_hash=spec_hash,
            data_version=data_version,
            code_version=code_version,
            stage=stage,
            expire=cache_expire,
        )

    def compute_and_cache(
        self,
        compute_fn,
        spec: dict,
        data_version: str,
        code_version: str,
        stage: str = "features",
        use_cache: bool = True,
        cache_expire: Optional[int] = None,
    ) -> pl.DataFrame:
        """
        Compute features (or load from cache) and save.

        Args:
            compute_fn: Function to compute features
            spec: Strategy specification
            data_version: Data version hash
            code_version: Code version hash
            stage: Stage name
            use_cache: Whether to use cache
            cache_expire: Cache expiration

        Returns:
            Feature dataframe
        """
        # Try to get from cache/storage
        cached = self.get_features(spec, data_version, code_version, stage, use_cache)
        if cached is not None:
            return cached

        # Compute features
        logger.info("computing_features", stage=stage)
        with measure_time(f"compute_{stage}"):
            df = compute_fn()

        # Save to cache and storage
        self.save_features(df, spec, data_version, code_version, stage, cache_expire)

        return df

    def get_feature_path(self, stage: str) -> Path:
        """
        Get feature file path.

        Args:
            stage: Stage name

        Returns:
            Path to feature file
        """
        return self.features_dir / f"{stage}.parquet"

    def clear_cache(self) -> None:
        """Clear feature cache."""
        self.cache.clear()

    def list_stages(self) -> list[str]:
        """
        List available feature stages.

        Returns:
            List of stage names
        """
        stages = []
        for path in self.features_dir.glob("*.parquet"):
            stages.append(path.stem)
        return stages
