"""
Universe providers for different markets
"""
from typing import Optional

import polars as pl

from ..common.logging import get_logger
from ..data.pipeline import DataPipeline

logger = get_logger(__name__)


class UniverseProvider:
    """Base class for universe providers."""

    def __init__(self, spec: dict):
        """
        Initialize universe provider.

        Args:
            spec: Strategy specification
        """
        self.spec = spec
        self.asset_type = spec["instrument"]["asset_type"]

    def get_universe(self) -> list[str]:
        """
        Get universe of symbols.

        Returns:
            List of symbols
        """
        raise NotImplementedError


class CNUniverseProvider(UniverseProvider):
    """Universe provider for Chinese A-shares."""

    def __init__(self, spec: dict):
        super().__init__(spec)
        self.symbols = spec["instrument"].get("symbols", [])

    def get_universe(self) -> list[str]:
        """
        Get A-share universe.

        Returns:
            List of symbols from config
        """
        if not self.symbols:
            logger.warning("empty_universe", message="No symbols specified in config")
        return self.symbols


class USUniverseProvider(UniverseProvider):
    """Universe provider for US stocks."""

    def __init__(self, spec: dict):
        super().__init__(spec)
        self.symbols = spec["instrument"].get("symbols", [])

    def get_universe(self) -> list[str]:
        """
        Get US stock universe.

        Returns:
            List of symbols from config (placeholder for SP500/S&P500)
        """
        if not self.symbols:
            logger.warning(
                "universe_placeholder",
                message="Using config symbols - implement SP500 loader for production"
            )
        return self.symbols


class CryptoUniverseProvider(UniverseProvider):
    """Universe provider for crypto assets."""

    def __init__(self, spec: dict, data_pipeline: Optional[DataPipeline] = None):
        """
        Initialize crypto universe provider.

        Args:
            spec: Strategy specification
            data_pipeline: Data pipeline for fetching top volume assets
        """
        super().__init__(spec)
        self.symbols = spec["instrument"].get("symbols", [])
        self.data_pipeline = data_pipeline

    def get_universe(self) -> list[str]:
        """
        Get crypto universe.

        Returns:
            List of symbols (top volume or from config)
        """
        if not self.symbols and self.data_pipeline:
            # Try to fetch top volume pairs from CCXT
            try:
                from ..data.sources.ccxt_source import CCXTSource

                venue = self.spec["instrument"]["venue"]
                source = CCXTSource(venue)

                logger.info("fetching_top_volume_pairs", venue=venue)
                all_symbols = source.fetch_symbols()

                # Filter to USDT pairs and sort by volume (placeholder)
                usdt_pairs = [s for s in all_symbols if "/USDT" in s]
                self.symbols = usdt_pairs[:50]  # Top 50

                logger.info("top_volume_pairs", count=len(self.symbols))

            except Exception as e:
                logger.error(
                    "top_volume_fetch_error",
                    error=str(e),
                    fallback="Using config symbols"
                )

        if not self.symbols:
            logger.warning("empty_universe", message="No crypto symbols available")
        return self.symbols


class CustomUniverseProvider(UniverseProvider):
    """Custom universe provider from user-supplied list."""

    def __init__(self, spec: dict, symbols: list[str]):
        """
        Initialize custom universe provider.

        Args:
            spec: Strategy specification
            symbols: User-supplied symbol list
        """
        super().__init__(spec)
        self.symbols = symbols

    def get_universe(self) -> list[str]:
        """
        Get custom universe.

        Returns:
            List of user-supplied symbols
        """
        return self.symbols


def create_universe_provider(
    spec: dict,
    data_pipeline: Optional[DataPipeline] = None,
    symbols: Optional[list[str]] = None,
) -> UniverseProvider:
    """
    Create universe provider based on asset type.

    Args:
        spec: Strategy specification
        data_pipeline: Data pipeline (for crypto universe)
        symbols: Custom symbol list (for CustomUniverseProvider)

    Returns:
        Universe provider instance
    """
    asset_type = spec["instrument"]["asset_type"]

    if symbols:
        return CustomUniverseProvider(spec, symbols)

    if asset_type == "CN_STOCK":
        return CNUniverseProvider(spec)
    elif asset_type == "US_STOCK":
        return USUniverseProvider(spec)
    elif asset_type in ["CRYPTO_SPOT", "CRYPTO_PERP"]:
        return CryptoUniverseProvider(spec, data_pipeline)
    else:
        raise ValueError(f"Unknown asset type: {asset_type}")


def filter_universe(
    df: pl.DataFrame,
    universe: list[str],
    symbol_col: str = "symbol",
) -> pl.DataFrame:
    """
    Filter dataframe to universe.

    Args:
        df: Input dataframe
        universe: List of allowed symbols
        symbol_col: Symbol column name

    Returns:
        Filtered dataframe
    """
    return df.filter(pl.col(symbol_col).is_in(universe))
