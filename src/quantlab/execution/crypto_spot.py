"""
Execution model for crypto spot trading
"""
import numpy as np

from .base import ExecutionModel

from ..common.logging import get_logger

logger = get_logger(__name__)


class CryptoSpotExecutionModel(ExecutionModel):
    """
    Execution model for crypto spot trading.

    Constraints:
    - Lot size: varies by exchange/asset (often 0.0001 BTC, 0.01 ETH, etc.)
    - Fractional trading allowed
    - Shorting allowed (margin trading)
    - Instant settlement
    - Maker/taker fees: typically 0.1% / 0.1%
    """

    def __init__(self, spec: dict):
        super().__init__(spec)

        # Crypto-specific constraints
        self.lot_size = spec["instrument"].get("lot_size", 0.0001)  # Very small
        self.allow_fractional = True
        self.shortable = spec["instrument"].get("shortable", True)

        # Crypto-specific fees (maker/taker)
        self.maker_bps = spec["backtest"].get("commission", 0.1)  # 0.1%
        self.taker_bps = self.maker_bps  # Assume same for now
        self.assume_taker = True  # Conservative: assume taker orders

    def apply_lot_sizing(self, quantities: np.ndarray) -> np.ndarray:
        """
        Apply lot sizing.

        Args:
            quantities: Array of quantities

        Returns:
            Lot-sized quantities
        """
        if self.lot_size <= 0:
            return quantities

        # Round to lot size
        return np.round(quantities / self.lot_size) * self.lot_size

    def apply_precision(self, quantities: np.ndarray) -> np.ndarray:
        """
        Apply precision (typically 8 decimals).

        Args:
            quantities: Array of quantities

        Returns:
            Precision-adjusted quantities
        """
        return np.round(quantities, 8)

    def apply_shortability(self, signals: np.ndarray) -> np.ndarray:
        """
        Apply shortability.

        Args:
            signals: Array of signals

        Returns:
            Adjusted signals
        """
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
        """
        Calculate crypto fees.

        Fees:
        - Maker: 0.1% (if limit order fills as maker)
        - Taker: 0.1% (if market order or crosses spread)

        Args:
            prices: Array of prices
            quantities: Array of quantities
            side: "buy" or "sell"

        Returns:
            Array of fee amounts
        """
        notional = np.abs(prices * quantities)

        # Assume taker for simplicity (conservative)
        fee_bps = self.taker_bps if self.assume_taker else self.maker_bps

        return notional * (fee_bps / 10000)
