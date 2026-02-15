"""
Parameter search space definitions
"""
from typing import Dict, List, Union, Any

import optuna
from optuna.distributions import BaseDistribution


class ParameterSpace:
    """
    Define parameter search spaces for optimization.
    """

    def __init__(self, spec: dict):
        """
        Initialize parameter space from spec.

        Args:
            spec: Strategy specification containing optimization parameters
        """
        self.spec = spec
        self.search_space = self._parse_search_space()

    def _parse_search_space(self) -> Dict[str, BaseDistribution]:
        """
        Parse search space from spec.

        Returns:
            Dictionary mapping parameter names to Optuna distributions
        """
        space = {}

        # Get optimization parameters from spec
        if "optimization" not in self.spec:
            return space

        opt_params = self.spec["optimization"]

        for param_name, param_def in opt_params.items():
            dist = self._create_distribution(param_def)
            if dist is not None:
                space[param_name] = dist

        return space

    def _create_distribution(self, param_def: Dict[str, Any]) -> BaseDistribution:
        """
        Create Optuna distribution from parameter definition.

        Args:
            param_def: Parameter definition with type and bounds

        Returns:
            Optuna distribution object
        """
        param_type = param_def.get("type", "float")
        lower_bound = param_def.get("lower")
        upper_bound = param_def.get("upper")
        step = param_def.get("step", None)

        if param_type == "float":
            if step:
                return optuna.distributions.FloatDistribution(lower_bound, upper_bound, step=step)
            else:
                return optuna.distributions.FloatDistribution(lower_bound, upper_bound)

        elif param_type == "int":
            return optuna.distributions.IntDistribution(lower_bound, upper_bound, step=param_def.get("step", 1))

        elif param_type == "categorical":
            choices = param_def.get("choices", [])
            return optuna.distributions.CategoricalDistribution(choices)

        elif param_type == "log":
            return optuna.distributions.FloatDistribution(lower_bound, upper_bound, log=True)

        else:
            raise ValueError(f"Unknown parameter type: {param_type}")

    def get_space(self) -> Dict[str, BaseDistribution]:
        """
        Get the search space.

        Returns:
            Dictionary mapping parameter names to distributions
        """
        return self.search_space

    def sample_point(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Sample a point from the search space.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary mapping parameter names to values
        """
        params = {}
        for param_name, distribution in self.search_space.items():
            params[param_name] = trial._suggest(param_name, distribution)
        return params

    def get_default_params(self) -> Dict[str, Any]:
        """
        Get default parameter values.

        Returns:
            Dictionary of default parameter values
        """
        defaults = {}
        if "optimization" not in self.spec:
            return defaults

        opt_params = self.spec["optimization"]
        for param_name, param_def in opt_params.items():
            defaults[param_name] = param_def.get("default", 0)
        return defaults


def create_common_spaces(asset_type: str) -> Dict[str, BaseDistribution]:
    """
    Create common parameter spaces for different asset types.

    Args:
        asset_type: Type of asset (CN_STOCK, US_STOCK, CRYPTO_SPOT, CRYPTO_PERP)

    Returns:
        Dictionary of common parameters for the asset type
    """
    common_spaces = {
        "momentum": {
            "period": optuna.distributions.IntDistribution(5, 50),
        },
        "rsi": {
            "period": optuna.distributions.IntDistribution(7, 30),
            "overbought": optuna.distributions.FloatDistribution(60, 80),
            "oversold": optuna.distributions.FloatDistribution(20, 40),
        },
        "macd": {
            "fast_period": optuna.distributions.IntDistribution(8, 16),
            "slow_period": optuna.distributions.IntDistribution(20, 40),
            "signal_period": optuna.distributions.IntDistribution(5, 15),
        },
        "bollinger": {
            "period": optuna.distributions.IntDistribution(10, 30),
            "std_dev": optuna.distributions.FloatDistribution(1.0, 3.0),
        },
        "atr": {
            "period": optuna.distributions.IntDistribution(7, 21),
        }
    }

    # Adjust ranges based on asset type
    if asset_type in ["CRYPTO_SPOT", "CRYPTO_PERP"]:
        # Crypto assets often have higher volatility
        common_spaces["rsi"]["overbought"] = optuna.distributions.FloatDistribution(65, 85)
        common_spaces["rsi"]["oversold"] = optuna.distributions.FloatDistribution(15, 35)

    elif asset_type == "US_STOCK":
        # US stocks might have different optimal ranges
        common_spaces["momentum"]["period"] = optuna.distributions.IntDistribution(10, 40)

    elif asset_type == "CN_STOCK":
        # A-shares might have different characteristics
        common_spaces["momentum"]["period"] = optuna.distributions.IntDistribution(5, 30)

    return common_spaces


def merge_spaces(
    base_space: Dict[str, BaseDistribution],
    additional_space: Dict[str, BaseDistribution]
) -> Dict[str, BaseDistribution]:
    """
    Merge two parameter spaces.

    Args:
        base_space: Base parameter space
        additional_space: Additional parameters to add

    Returns:
        Merged parameter space
    """
    merged = base_space.copy()
    merged.update(additional_space)
    return merged
