"""
Transaction cost modeling
"""
from typing import Optional

import numpy as np
import polars as pl

from ..common.logging import get_logger

logger = get_logger(__name__)


def calculate_slippage_cost(
    prices: np.ndarray,
    volumes: np.ndarray,
    quantities: np.ndarray,
    bid_ask_spread: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Calculate slippage costs.

    Args:
        prices: Array of prices
        volumes: Array of volumes
        quantities: Array of quantities
        bid_ask_spread: Array of bid-ask spreads (optional)

    Returns:
        Array of slippage costs
    """
    if bid_ask_spread is not None:
        # Use provided bid-ask spread
        spread_pct = bid_ask_spread / prices
    else:
        # Estimate spread based on volume (higher volume = lower spread)
        avg_volume = np.mean(volumes)
        spread_pct = 0.001 * (avg_volume / volumes)  # Inverse relationship

    # Slippage is half the spread (assuming random order fill)
    slippage_costs = np.abs(quantities) * prices * (spread_pct / 2)

    return slippage_costs


def calculate_commission_cost(
    prices: np.ndarray,
    quantities: np.ndarray,
    commission_rate: float,
    fee_structure: str = "bps",
) -> np.ndarray:
    """
    Calculate commission costs.

    Args:
        prices: Array of prices
        quantities: Array of quantities
        commission_rate: Commission rate (e.g., 0.001 = 0.1%)
        fee_structure: "bps" (basis points) or "absolute"

    Returns:
        Array of commission costs
    """
    if fee_structure == "bps":
        commission_rate = commission_rate / 10000  # Convert bps to decimal

    notional = np.abs(quantities * prices)
    return notional * commission_rate


def calculate_impact_cost(
    prices: np.ndarray,
    volumes: np.ndarray,
    quantities: np.ndarray,
    participation_rate: float = 0.01,
) -> np.ndarray:
    """
    Calculate market impact costs using square root model.

    Args:
        prices: Array of prices
        volumes: Array of volumes
        quantities: Array of quantities
        participation_rate: Participation rate in market volume

    Returns:
        Array of impact costs
    """
    # Calculate trading intensity
    trading_intensity = np.abs(quantities) / (volumes + 1e-8)  # Add small value to avoid division by 0

    # Square root impact model
    impact_pct = participation_rate * np.sqrt(trading_intensity)

    # Impact cost is proportional to price movement
    impact_costs = np.abs(quantities) * prices * impact_pct

    return impact_costs


def calculate_total_cost(
    prices: np.ndarray,
    volumes: np.ndarray,
    quantities: np.ndarray,
    commission_rate: float,
    bid_ask_spread: Optional[np.ndarray] = None,
    participation_rate: float = 0.01,
) -> np.ndarray:
    """
    Calculate total transaction costs.

    Args:
        prices: Array of prices
        volumes: Array of volumes
        quantities: Array of quantities
        commission_rate: Commission rate
        bid_ask_spread: Bid-ask spread (optional)
        participation_rate: Market impact participation rate

    Returns:
        Array of total costs
    """
    slippage = calculate_slippage_cost(prices, volumes, quantities, bid_ask_spread)
    commission = calculate_commission_cost(prices, quantities, commission_rate)
    impact = calculate_impact_cost(prices, volumes, quantities, participation_rate)

    return slippage + commission + impact


def apply_costs_to_returns(
    returns: np.ndarray,
    prices: np.ndarray,
    volumes: np.ndarray,
    quantities: np.ndarray,
    commission_rate: float,
    bid_ask_spread: Optional[np.ndarray] = None,
    participation_rate: float = 0.01,
) -> np.ndarray:
    """
    Apply transaction costs to returns.

    Args:
        returns: Array of pre-cost returns
        prices: Array of prices
        volumes: Array of volumes
        quantities: Array of quantities
        commission_rate: Commission rate
        bid_ask_spread: Bid-ask spread (optional)
        participation_rate: Market impact participation rate

    Returns:
        Array of post-cost returns
    """
    costs = calculate_total_cost(
        prices, volumes, quantities, commission_rate,
        bid_ask_spread, participation_rate
    )

    # Calculate notional value to convert costs to returns
    notional = np.abs(quantities) * prices

    # Convert costs to return impact
    cost_returns = costs / notional

    # Subtract costs from returns (only when trading occurs)
    mask = quantities != 0
    adjusted_returns = returns.copy()
    adjusted_returns[mask] -= cost_returns[mask]

    return adjusted_returns


class CostModel:
    """Cost model for transaction cost estimation."""

    def __init__(
        self,
        commission_rate: float = 0.001,
        bid_ask_spread: Optional[float] = 0.001,
        participation_rate: float = 0.01,
    ):
        """
        Initialize cost model.

        Args:
            commission_rate: Commission rate in bps
            bid_ask_spread: Bid-ask spread in bps (optional)
            participation_rate: Market impact participation rate
        """
        self.commission_rate = commission_rate / 10000  # Convert bps to decimal
        self.bid_ask_spread = bid_ask_spread / 10000 if bid_ask_spread else None
        self.participation_rate = participation_rate

    def calculate_costs(
        self,
        df: pl.DataFrame,
        quantities: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate costs for dataframe.

        Args:
            df: OHLCV dataframe
            quantities: Array of quantities

        Returns:
            Array of total costs
        """
        prices = df["close"].to_numpy()
        volumes = df["volume"].to_numpy()

        return calculate_total_cost(
            prices, volumes, quantities,
            self.commission_rate,
            np.full(len(prices), self.bid_ask_spread) if self.bid_ask_spread else None,
            self.participation_rate
        )
