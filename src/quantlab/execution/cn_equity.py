"""
Execution model for Chinese equities
"""
import numpy as np

from .base import ExecutionModel

from ..common.logging import get_logger

logger = get_logger(__name__)


class CNEquityExecutionModel(ExecutionModel):
    """
    Execution model for Chinese A-shares.

    Constraints:
    - Lot size: 100 shares (minimum)
    - No fractional shares
    - No shorting (in general)
    - T+1 settlement
    - Commission: ~0.03% bps (varies by broker)
    """

    def __init__(self, spec: dict):
        super().__init__(spec)

        # CN-specific constraints
        self.lot_size = max(self.lot_size, 100)  # Minimum 100 shares
        self.allow_fractional = False
        self.shortable = False  # Generally no shorting for A-shares

        # CN-specific fees
        self.commission_bps = spec["backtest"].get("commission", 0.03)  # 0.03%
        self.stamp_duty_bps = 0.1  # 0.1% stamp duty on sell only

    def apply_lot_sizing(self, quantities: np.ndarray) -> np.ndarray:
        """
        Apply lot sizing (100 shares per lot).

        Args:
            quantities: Array of quantities

        Returns:
            Lot-sized quantities
        """
        # Round to nearest 100
        return np.round(quantities / 100) * 100

    def apply_precision(self, quantities: np.ndarray) -> np.ndarray:
        """
        Apply precision (integer shares only).

        Args:
            quantities: Array of quantities

        Returns:
            Integer quantities
        """
        return np.round(quantities).astype(int)

    def apply_shortability(self, signals: np.ndarray) -> np.ndarray:
        """
        Apply shortability (no shorting allowed).

        Args:
            signals: Array of signals

        Returns:
            Adjusted signals (long-only)
        """
        return np.where(signals > 0, signals, 0)

    def calculate_fees(
        self,
        prices: np.ndarray,
        quantities: np.ndarray,
        side: str,
    ) -> np.ndarray:
        """
        Calculate CN equity fees.

        Fees:
        - Commission: 0.03% (buy and sell)
        - Stamp duty: 0.1% (sell only)

        Args:
            prices: Array of prices
            quantities: Array of quantities
            side: "buy" or "sell"

        Returns:
            Array of fee amounts
        """
        notional = np.abs(prices * quantities)

        # Commission (both buy and sell)
        commission = notional * (self.commission_bps / 10000)

        # Stamp duty (sell only)
        if side == "sell":
            stamp_duty = notional * (self.stamp_duty_bps / 10000)
        else:
            stamp_duty = np.zeros_like(notional)

        return commission + stamp_duty
