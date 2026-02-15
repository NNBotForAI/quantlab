"""
Sensitivity analysis for parameter robustness
"""
from typing import Dict, Any, List, Tuple

import numpy as np
import polars as pl

from ..common.logging import get_logger
from ..common.perf import measure_time

logger = get_logger(__name__)


class SensitivityAnalysis:
    """
    Sensitivity analysis to test parameter robustness.
    """

    def __init__(self, spec: dict):
        """
        Initialize sensitivity analysis.

        Args:
            spec: Strategy specification
        """
        self.spec = spec
        self.base_params = spec.get("sensitivity", {}).get("base_params", {})
        self.param_ranges = spec.get("sensitivity", {}).get("param_ranges", {})
        self.n_points = spec.get("sensitivity", {}).get("n_points", 5)

    def run_analysis(
        self,
        df: pl.DataFrame,
        strategy_func,
        metric_name: str = "sharpe_ratio",
    ) -> dict:
        """
        Run sensitivity analysis.

        Args:
            df: OHLCV dataframe
            strategy_func: Function that takes (df, params) and returns (signals, metrics)
            metric_name: Metric to analyze sensitivity for

        Returns:
            Dictionary with sensitivity results
        """
        logger.info("sensitivity_analysis_start")

        # Get parameters to analyze
        params_to_analyze = self._get_params_to_analyze()

        results = {}
        param_combinations = []

        with measure_time("sensitivity_analysis"):
            # For each parameter, vary it while keeping others constant
            for param_name, param_range in params_to_analyze.items():
                param_values, metric_values = [], []

                for value in param_range:
                    params = self.base_params.copy()
                    params[param_name] = value

                    try:
                        _, metrics = strategy_func(df, params)
                        metric_value = metrics.get(metric_name, 0)

                        param_values.append(value)
                        metric_values.append(metric_value)

                    except Exception as e:
                        logger.warning("param_eval_error", param=param_name, value=value, error=str(e))
                        param_values.append(value)
                        metric_values.append(float('nan'))

                results[param_name] = {
                    "values": param_values,
                    "metric_values": metric_values,
                    "sensitivity": self._calculate_sensitivity(param_values, metric_values),
                }

                # Add to combinations for multi-parameter analysis
                for i, value in enumerate(param_values):
                    combo = self.base_params.copy()
                    combo[param_name] = value
                    param_combinations.append((combo, metric_values[i]))

        # Multi-parameter sensitivity (if more than one parameter)
        multi_param_results = {}
        if len(params_to_analyze) > 1:
            multi_param_results = self._multi_parameter_analysis(
                df, strategy_func, metric_name, params_to_analyze
            )

        logger.info("sensitivity_analysis_complete")

        return {
            "single_parameter": results,
            "multi_parameter": multi_param_results,
            "param_combinations": param_combinations,
            "completed_at": datetime.utcnow().isoformat(),
        }

    def _get_params_to_analyze(self) -> Dict[str, List[Any]]:
        """
        Get parameters to analyze based on spec.

        Returns:
            Dictionary mapping parameter names to value ranges
        """
        params = {}

        for param_name, range_def in self.param_ranges.items():
            if "range" in range_def:
                # Linear range
                start, end, num = range_def["range"]
                params[param_name] = np.linspace(start, end, num).tolist()
            elif "values" in range_def:
                # Explicit values
                params[param_name] = range_def["values"]
            elif "log_range" in range_def:
                # Logarithmic range
                start, end, num = range_def["log_range"]
                params[param_name] = np.logspace(np.log10(start), np.log10(end), num).tolist()
            else:
                # Default: vary around base value
                base_val = self.base_params.get(param_name, 1.0)
                params[param_name] = [
                    base_val * 0.5,
                    base_val * 0.75,
                    base_val,
                    base_val * 1.25,
                    base_val * 1.5
                ]

        return params

    def _calculate_sensitivity(self, param_values: List[float], metric_values: List[float]) -> float:
        """
        Calculate sensitivity as the slope of metric vs parameter.

        Args:
            param_values: Parameter values tested
            metric_values: Corresponding metric values

        Returns:
            Sensitivity value (slope of linear fit)
        """
        # Remove NaN values
        valid_pairs = [(p, m) for p, m in zip(param_values, metric_values) if not np.isnan(m)]
        
        if len(valid_pairs) < 2:
            return 0.0

        param_vals, metric_vals = zip(*valid_pairs)

        # Calculate slope using least squares
        param_vals = np.array(param_vals)
        metric_vals = np.array(metric_vals)

        # Add intercept term
        A = np.vstack([param_vals, np.ones(len(param_vals))]).T
        slope, _ = np.linalg.lstsq(A, metric_vals, rcond=None)[0]

        return float(slope)

    def _multi_parameter_analysis(
        self,
        df: pl.DataFrame,
        strategy_func,
        metric_name: str,
        params_to_analyze: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        """
        Perform multi-parameter sensitivity analysis.

        Args:
            df: OHLCV dataframe
            strategy_func: Strategy function
            metric_name: Metric to analyze
            params_to_analyze: Parameters to analyze

        Returns:
            Multi-parameter analysis results
        """
        if len(params_to_analyze) < 2:
            return {}

        # Generate parameter grid
        param_names = list(params_to_analyze.keys())
        param_values = list(params_to_analyze.values())

        # Create meshgrid for parameter combinations
        from itertools import product
        param_combinations = list(product(*param_values))

        results_grid = []
        metric_matrix = []

        for combo in param_combinations:
            params = self.base_params.copy()
            for i, name in enumerate(param_names):
                params[name] = combo[i]

            try:
                _, metrics = strategy_func(df, params)
                metric_value = metrics.get(metric_name, 0)
            except Exception:
                metric_value = float('nan')

            results_grid.append({
                "params": dict(zip(param_names, combo)),
                "metric": metric_value
            })
            metric_matrix.append(metric_value)

        # If we have 2 parameters, reshape to 2D matrix for heatmap
        if len(param_names) == 2:
            n1, n2 = len(param_values[0]), len(param_values[1])
            metric_matrix = np.array(metric_matrix).reshape(n1, n2)

            # Calculate interaction effects
            interaction_effect = self._calculate_interaction_effect(
                np.array(param_values[0]),
                np.array(param_values[1]),
                metric_matrix
            )
        else:
            interaction_effect = None

        return {
            "param_names": param_names,
            "param_values": param_values,
            "results_grid": results_grid,
            "metric_matrix": metric_matrix.tolist() if len(param_names) == 2 else metric_matrix,
            "interaction_effect": interaction_effect,
        }

    def _calculate_interaction_effect(
        self,
        param1_vals: np.ndarray,
        param2_vals: np.ndarray,
        metric_matrix: np.ndarray
    ) -> float:
        """
        Calculate interaction effect between two parameters.

        Args:
            param1_vals: Values for first parameter
            param2_vals: Values for second parameter
            metric_matrix: Matrix of metric values [param1, param2]

        Returns:
            Interaction effect (measure of non-additivity)
        """
        # Center the matrix around overall mean
        overall_mean = np.mean(metric_matrix)
        centered_matrix = metric_matrix - overall_mean

        # Calculate main effects
        main_effect_1 = np.mean(centered_matrix, axis=1)  # Mean across param2
        main_effect_2 = np.mean(centered_matrix, axis=0)  # Mean across param1

        # Calculate interaction effect as deviation from additive model
        interaction_matrix = centered_matrix - main_effect_1[:, np.newaxis] - main_effect_2[np.newaxis, :]

        # Overall interaction strength as Frobenius norm of interaction matrix
        interaction_strength = np.linalg.norm(interaction_matrix, 'fro') / (metric_matrix.size)

        return interaction_strength

    def generate_heatmap_data(
        self,
        results: dict,
        param_x: str,
        param_y: str,
        metric_name: str = "sharpe_ratio"
    ) -> dict:
        """
        Generate heatmap data for visualization.

        Args:
            results: Sensitivity analysis results
            param_x: X-axis parameter name
            param_y: Y-axis parameter name
            metric_name: Metric to visualize

        Returns:
            Heatmap data dictionary
        """
        if "multi_parameter" not in results or not results["multi_parameter"]:
            return {"error": "Multi-parameter results not available"}

        multi_results = results["multi_parameter"]

        if "param_names" not in multi_results or len(multi_results["param_names"]) < 2:
            return {"error": "Not enough parameters for heatmap"}

        if param_x not in multi_results["param_names"] or param_y not in multi_results["param_names"]:
            return {"error": f"Parameters {param_x} or {param_y} not in analysis"}

        # Find indices
        idx_x = multi_results["param_names"].index(param_x)
        idx_y = multi_results["param_names"].index(param_y)

        # Extract values
        x_vals = multi_results["param_values"][idx_x]
        y_vals = multi_results["param_values"][idx_y]

        # Get metric matrix
        metric_matrix = np.array(multi_results["metric_matrix"])

        # Transpose if needed to match x/y order
        if idx_x == 1:  # If x is second parameter, transpose
            metric_matrix = metric_matrix.T

        return {
            "x_values": x_vals,
            "y_values": y_vals,
            "z_values": metric_matrix.tolist(),
            "x_label": param_x,
            "y_label": param_y,
            "metric": metric_name,
        }
