"""
Portfolio construction and management
"""
from typing import Optional

import numpy as np
import polars as pl

from ..common.logging import get_logger

logger = get_logger(__name__)


class PortfolioManager:
    """
    Portfolio construction and management.

    Handles position sizing, rebalancing, and risk management.
    """

    def __init__(self, spec: dict):
        """
        Initialize portfolio manager.

        Args:
            spec: Strategy specification
        """
        self.spec = spec
        self.initial_capital = spec["backtest"]["initial_capital"]
        self.max_position_size = spec.get("max_position_size", 0.1)  # 10% max per position
        self.max_leverage = spec.get("max_leverage", 1.0)

    def size_positions(
        self,
        signals: np.ndarray,
        prices: np.ndarray,
        method: str = "equal_weight",
        **kwargs,
    ) -> np.ndarray:
        """
        Size positions based on signals.

        Args:
            signals: Array of signals (-1, 0, 1)
            prices: Array of prices
            method: Position sizing method ("equal_weight", "risk_parity", "vol_target")
            **kwargs: Additional method-specific parameters

        Returns:
            Array of position sizes
        """
        if method == "equal_weight":
            return self._equal_weight_position_sizing(signals, prices)
        elif method == "risk_parity":
            return self._risk_parity_position_sizing(signals, prices, **kwargs)
        elif method == "vol_target":
            return self._vol_target_position_sizing(signals, prices, **kwargs)
        else:
            raise ValueError(f"Unknown position sizing method: {method}")

    def _equal_weight_position_sizing(
        self,
        signals: np.ndarray,
        prices: np.ndarray,
    ) -> np.ndarray:
        """
        Equal-weight position sizing.

        Args:
            signals: Array of signals
            prices: Array of prices

        Returns:
            Array of position sizes
        """
        n_assets = len(signals)
        if n_assets == 0:
            return np.array([])

        # Calculate equal weight
        weights = np.ones(n_assets) / n_assets

        # Apply signal direction
        weights = weights * signals

        # Cap position sizes based on max position size
        weights = np.clip(weights, -self.max_position_size, self.max_position_size)

        # Calculate dollar value
        capital_per_asset = self.initial_capital * weights

        # Convert to quantities
        quantities = capital_per_asset / prices

        return quantities

    def _risk_parity_position_sizing(
        self,
        signals: np.ndarray,
        prices: np.ndarray,
        volatilities: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Risk parity position sizing.

        Args:
            signals: Array of signals
            prices: Array of prices
            volatilities: Array of volatilities (optional, estimated if None)

        Returns:
            Array of position sizes
        """
        if volatilities is None:
            # Estimate volatilities from prices
            returns = np.diff(np.log(prices))
            volatilities = np.full(len(signals), np.std(returns))

        # Invert volatilities to get inverse risk weights
        inv_vol = 1.0 / (volatilities + 1e-8)  # Add small value to avoid division by 0

        # Normalize weights
        weights = inv_vol / np.sum(inv_vol)

        # Apply signal direction
        weights = weights * signals

        # Cap position sizes
        weights = np.clip(weights, -self.max_position_size, self.max_position_size)

        # Calculate dollar value
        capital_per_asset = self.initial_capital * weights

        # Convert to quantities
        quantities = capital_per_asset / prices

        return quantities

    def _vol_target_position_sizing(
        self,
        signals: np.ndarray,
        prices: np.ndarray,
        target_vol: float = 0.15,  # 15% annualized
        lookback: int = 252,  # Annualize from daily
    ) -> np.ndarray:
        """
        Volatility targeting position sizing.

        Args:
            signals: Array of signals
            prices: Array of prices
            target_vol: Target annualized volatility
            lookback: Lookback period for volatility estimation

        Returns:
            Array of position sizes
        """
        # Estimate volatility
        returns = np.diff(np.log(prices))
        if len(returns) < lookback:
            lookback = len(returns)

        if lookback > 0:
            vol = np.std(returns[-lookback:]) * np.sqrt(252)  # Annualized
        else:
            vol = target_vol  # Default to target if insufficient data

        # Calculate position size to achieve target volatility
        if vol > 0:
            leverage = target_vol / vol
            leverage = min(leverage, self.max_leverage)  # Cap leverage
        else:
            leverage = 1.0

        # Base weights
        base_weights = np.ones(len(signals)) / len(signals)
        weights = base_weights * leverage

        # Apply signal direction
        weights = weights * signals

        # Cap position sizes
        weights = np.clip(weights, -self.max_position_size, self.max_position_size)

        # Calculate dollar value
        capital_per_asset = self.initial_capital * weights

        # Convert to quantities
        quantities = capital_per_asset / prices

        return quantities

    def rebalance_portfolio(
        self,
        current_positions: np.ndarray,
        target_weights: np.ndarray,
        prices: np.ndarray,
        rebalance_threshold: float = 0.05,
    ) -> np.ndarray:
        """
        Rebalance portfolio based on threshold.

        Args:
            current_positions: Current position quantities
            target_weights: Target weights
            prices: Current prices
            rebalance_threshold: Threshold for rebalancing

        Returns:
            Array of rebalanced position quantities
        """
        current_values = current_positions * prices
        current_weights = current_values / np.sum(current_values)

        # Calculate deviation from target
        deviation = np.abs(target_weights - current_weights)

        # Only rebalance if deviation exceeds threshold
        rebalance_mask = deviation > rebalance_threshold

        if not np.any(rebalance_mask):
            # No rebalancing needed
            return current_positions

        # Calculate new positions based on target weights
        total_value = np.sum(current_values)
        target_values = total_value * target_weights
        new_quantities = target_values / prices

        return new_quantities

    def calculate_portfolio_metrics(
        self,
        equity_curve: pl.DataFrame,
        weights: np.ndarray,
    ) -> dict:
        """
        Calculate portfolio-level metrics.

        Args:
            equity_curve: Portfolio equity curve
            weights: Asset weights

        Returns:
            Dictionary of portfolio metrics
        """
        equity = equity_curve["equity"].to_numpy()

        # Calculate correlation-adjusted risk metrics
        portfolio_vol = np.std(np.diff(np.log(equity)))

        # Effective number of positions (diversification measure)
        effective_n = 1.0 / np.sum(weights ** 2) if np.sum(weights ** 2) > 0 else 0

        return {
            "portfolio_volatility": portfolio_vol,
            "effective_number_positions": effective_n,
            "concentration_ratio": 1.0 / effective_n if effective_n > 0 else 0,
            "tracking_error": 0.0,  # Placeholder
            "information_ratio": 0.0,  # Placeholder
        }
