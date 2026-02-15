"""
Base execution model
"""
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import polars as pl

from ..common.logging import get_logger

logger = get_logger(__name__)


class ExecutionModel(ABC):
    """Abstract base class for execution models."""

    def __init__(self, spec: dict):
        """
        Initialize execution model.

        Args:
            spec: Strategy specification
        """
        self.spec = spec
        self.asset_type = spec["instrument"]["asset_type"]

        # Load instrument constraints
        self.lot_size = spec["instrument"].get("lot_size", 1)
        self.allow_fractional = spec["instrument"].get("allow_fractional", False)
        self.shortable = spec["instrument"].get("shortable", True)
        self.leverage = spec["instrument"].get("leverage", 1)

        # Load fee model
        self.commission_bps = spec["backtest"].get("commission", 0.001)
        self.slippage_bps = spec["backtest"].get("slippage", 0.0005)

    @abstractmethod
    def apply_lot_sizing(
        self,
        quantities: np.ndarray,
    ) -> np.ndarray:
        """
        Apply lot sizing constraints.

        Args:
            quantities: Array of raw quantities

        Returns:
            Array of lot-sized quantities
        """
        pass

    @abstractmethod
    def apply_precision(
        self,
        quantities: np.ndarray,
    ) -> np.ndarray:
        """
        Apply quantity precision constraints.

        Args:
            quantities: Array of quantities

        Returns:
            Array of precision-adjusted quantities
        """
        pass

    @abstractmethod
    def apply_shortability(
        self,
        signals: np.ndarray,
    ) -> np.ndarray:
        """
        Apply shortability constraints.

        Args:
            signals: Array of signals (-1, 0, 1)

        Returns:
            Array of adjusted signals
        """
        pass

    @abstractmethod
    def calculate_fees(
        self,
        prices: np.ndarray,
        quantities: np.ndarray,
        side: str,
    ) -> np.ndarray:
        """
        Calculate trading fees.

        Args:
            prices: Array of prices
            quantities: Array of quantities
            side: "buy" or "sell"

        Returns:
            Array of fee amounts
        """
        pass

    def apply_all_constraints(
        self,
        df: pl.DataFrame,
        signals_col: str = "signal",
    ) -> pl.DataFrame:
        """
        Apply all execution constraints.

        Args:
            df: Input dataframe
            signals_col: Signal column name

        Returns:
            Dataframe with applied constraints
        """
        # Convert to numpy for vectorized operations
        signals = df[signals_col].to_numpy()
        prices = df["close"].to_numpy()
        volumes = df["volume"].to_numpy()

        # Apply constraints
        signals = self.apply_shortability(signals)

        # Calculate quantities (simple equal-weight for now)
        # In production, use position sizing logic
        quantities = signals * 100  # Placeholder: 100 shares/coins per signal

        # Apply lot sizing and precision
        quantities = self.apply_lot_sizing(quantities)
        quantities = self.apply_precision(quantities)

        # Calculate fees
        buy_fees = self.calculate_fees(prices, quantities, "buy")
        sell_fees = self.calculate_fees(prices, -quantities, "sell")

        # Add columns to dataframe
        df = df.with_columns([
            pl.Series("signal_adjusted", signals),
            pl.Series("quantity", quantities),
            pl.Series("buy_fee", buy_fees),
            pl.Series("sell_fee", sell_fees),
        ])

        return df


class GenericExecutionModel(ExecutionModel):
    """Generic execution model for most asset types."""

    def __init__(self, spec: dict):
        super().__init__(spec)
        self.min_qty = 1e-6  # Minimum quantity

    def apply_lot_sizing(self, quantities: np.ndarray) -> np.ndarray:
        """Apply lot sizing."""
        if self.allow_fractional or self.lot_size == 0:
            return quantities

        # Round to lot size
        return np.round(quantities / self.lot_size) * self.lot_size

    def apply_precision(self, quantities: np.ndarray) -> np.ndarray:
        """Apply quantity precision."""
        # Determine precision based on lot size
        if self.allow_fractional:
            decimals = 8  # Crypto typically 8 decimals
        elif self.lot_size < 1:
            decimals = int(abs(np.log10(self.lot_size))) + 2
        else:
            decimals = 0

        return np.round(quantities, decimals)

    def apply_shortability(self, signals: np.ndarray) -> np.ndarray:
        """Apply shortability constraints."""
        if self.shortable:
            return signals

        # No shorting allowed
        return np.where(signals > 0, signals, 0)

    def calculate_fees(
        self,
        prices: np.ndarray,
        quantities: np.ndarray,
        side: str,
    ) -> np.ndarray:
        """Calculate fees using basis points."""
        notional = np.abs(prices * quantities)
        return notional * (self.commission_bps / 10000)
