"""
Execution model for crypto perpetual futures
"""
import numpy as np

from .base import ExecutionModel

from ..common.logging import get_logger

logger = get_logger(__name__)


class CryptoPerpExecutionModel(ExecutionModel):
    """
    Execution model for crypto perpetual futures.

    Constraints:
    - Lot size: varies by exchange/contract (often 1 contract)
    - Fractional contracts generally not allowed
    - Shorting allowed (perpetuals are designed for it)
    - Instant settlement
    - Maker/taker fees: typically 0.02% / 0.04%
    - Funding rate applies to open positions
    """

    def __init__(self, spec: dict):
        super().__init__(spec)

        # Crypto perp-specific constraints
        self.lot_size = spec["instrument"].get("lot_size", 1)  # 1 contract
        self.allow_fractional = False
        self.shortable = True  # Perps always allow shorting
        self.leverage = spec["instrument"].get("leverage", 1)

        # Crypto perp-specific fees (maker/taker)
        self.maker_bps = spec["backtest"].get("commission", 0.02)  # 0.02%
        self.taker_bps = spec["backtest"].get("commission", 0.04)  # 0.04%
        self.assume_taker = True

    def apply_lot_sizing(self, quantities: np.ndarray) -> np.ndarray:
        """
        Apply lot sizing (whole contracts only).

        Args:
            quantities: Array of quantities

        Returns:
            Lot-sized quantities
        """
        # Round to nearest whole contract
        return np.round(quantities / self.lot_size) * self.lot_size

    def apply_precision(self, quantities: np.ndarray) -> np.ndarray:
        """
        Apply precision (integer contracts).

        Args:
            quantities: Array of quantities

        Returns:
            Integer contract quantities
        """
        return np.round(quantities).astype(int)

    def apply_shortability(self, signals: np.ndarray) -> np.ndarray:
        """
        Apply shortability (perps always allow shorting).

        Args:
            signals: Array of signals

        Returns:
            Signals (unchanged)
        """
        # Perpetuals always allow shorting
        return signals

    def calculate_fees(
        self,
        prices: np.ndarray,
        quantities: np.ndarray,
        side: str,
    ) -> np.ndarray:
        """
        Calculate crypto perp fees.

        Fees:
        - Maker: 0.02% (if limit order fills as maker)
        - Taker: 0.04% (if market order or crosses spread)

        Note: Fees apply to notional value of position.

        Args:
            prices: Array of prices
            quantities: Array of quantities
            side: "buy" or "sell"

        Returns:
            Array of fee amounts
        """
        notional = np.abs(prices * quantities * self.leverage)

        # Assume taker for simplicity (conservative)
        fee_bps = self.taker_bps if self.assume_taker else self.maker_bps

        return notional * (fee_bps / 10000)

    def calculate_funding(
        self,
        prices: np.ndarray,
        quantities: np.ndarray,
        funding_rate: float,
    ) -> np.ndarray:
        """
        Calculate funding payments.

        Args:
            prices: Array of mark prices
            quantities: Array of position quantities
            funding_rate: Funding rate (e.g., 0.0001 = 0.01% per 8 hours)

        Returns:
            Array of funding payments (positive = receive, negative = pay)
        """
        # Funding is applied to notional position value
        notional = prices * quantities * self.leverage
        return notional * funding_rate
