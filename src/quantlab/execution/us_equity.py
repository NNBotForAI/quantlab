"""
Execution model for US equities
"""
import numpy as np

from .base import ExecutionModel

from ..common.logging import get_logger

logger = get_logger(__name__)


class USEquityExecutionModel(ExecutionModel):
    """
    Execution model for US equities.

    Constraints:
    - Lot size: 1 share (can trade fractional)
    - Fractional shares allowed (for many brokers)
    - Shorting allowed
    - T+2 settlement
    - Commission: typically ~$0.005/share or percentage-based
    """

    def __init__(self, spec: dict):
        super().__init__(spec)

        # US-specific constraints
        self.lot_size = spec["instrument"].get("lot_size", 1)
        self.allow_fractional = spec["instrument"].get("allow_fractional", True)
        self.shortable = spec["instrument"].get("shortable", True)

        # US-specific fees
        self.commission_bps = spec["backtest"].get("commission", 0.01)  # 0.01%
        self.sec_fee_bps = 0.0023  # SEC fee on sell only

    def apply_lot_sizing(self, quantities: np.ndarray) -> np.ndarray:
        """
        Apply lot sizing.

        Args:
            quantities: Array of quantities

        Returns:
            Lot-sized quantities
        """
        if self.allow_fractional:
            return quantities

        # Round to nearest share
        return np.round(quantities)

    def apply_precision(self, quantities: np.ndarray) -> np.ndarray:
        """
        Apply precision.

        Args:
            quantities: Array of quantities

        Returns:
            Precision-adjusted quantities
        """
        if self.allow_fractional:
            # Fractional shares: 4 decimal places
            return np.round(quantities, 4)
        else:
            # Integer shares
            return np.round(quantities).astype(int)

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
        Calculate US equity fees.

        Fees:
        - Commission: 0.01% (buy and sell)
        - SEC fee: 0.0023% (sell only)

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

        # SEC fee (sell only)
        if side == "sell":
            sec_fee = notional * (self.sec_fee_bps / 10000)
        else:
            sec_fee = np.zeros_like(notional)

        return commission + sec_fee
