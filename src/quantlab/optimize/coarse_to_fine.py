"""
Coarse-to-fine optimization strategy
"""
from typing import List, Tuple, Dict, Any

import optuna
import polars as pl

from .search_space import ParameterSpace
from ..common.logging import get_logger

logger = get_logger(__name__)


class CoarseToFineOptimizer:
    """
    Two-phase optimization: Coarse (grid/random) -> Fine (Bayesian).
    """

    def __init__(self, spec: dict):
        """
        Initialize coarse-to-fine optimizer.

        Args:
            spec: Strategy specification
        """
        self.spec = spec
        self.param_space = ParameterSpace(spec)
        self.n_coarse_samples = spec.get("optimization", {}).get("coarse_samples", 50)
        self.n_fine_samples = spec.get("optimization", {}).get("fine_samples", 100)
        self.top_k = spec.get("optimization", {}).get("top_k", 10)

    def optimize(
        self,
        objective_fn,
        data_df: pl.DataFrame,
        n_jobs: int = 1,
    ) -> Tuple[Dict[str, Any], float, List[Tuple[Dict[str, Any], float]]]:
        """
        Perform coarse-to-fine optimization.

        Args:
            objective_fn: Objective function to maximize
            data_df: Input data for backtesting
            n_jobs: Number of parallel jobs

        Returns:
            Tuple of (best_params, best_score, all_trials)
        """
        logger.info("coarse_to_fine_start")

        # Phase A: Coarse optimization
        logger.info("starting_coarse_optimization", samples=self.n_coarse_samples)
        coarse_results = self._coarse_search(objective_fn, data_df, n_jobs)

        # Select top K configurations
        top_configs = self._select_top_configs(coarse_results, self.top_k)

        logger.info("selected_top_configs", count=len(top_configs))

        # Phase B: Fine optimization around top configs
        logger.info("starting_fine_optimization", samples=self.n_fine_samples)
        fine_results = self._fine_search(objective_fn, data_df, top_configs, n_jobs)

        # Combine results
        all_results = coarse_results + fine_results

        # Find best overall
        best_idx = max(range(len(all_results)), key=lambda i: all_results[i][1])
        best_params, best_score = all_results[best_idx]

        logger.info("optimization_complete", best_score=best_score, best_params=best_params)

        return best_params, best_score, all_results

    def _coarse_search(
        self,
        objective_fn,
        data_df: pl.DataFrame,
        n_jobs: int = 1,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Perform coarse search (grid/random sampling).

        Args:
            objective_fn: Objective function
            data_df: Input data
            n_jobs: Number of parallel jobs

        Returns:
            List of (params, score) tuples
        """
        results = []

        # Use Optuna sampler for coarse search
        study = optuna.create_study(direction="maximize")
        search_space = self.param_space.get_space()

        # For coarse search, we'll use random sampling
        for i in range(self.n_coarse_samples):
            # Sample a random configuration
            trial = study.ask(search_space)
            params = {name: trial.params[name] for name in trial.params}

            try:
                # Evaluate the configuration with shorter window for speed
                score = self._evaluate_with_shorter_window(
                    objective_fn, data_df, params
                )
                results.append((params, score))
            except Exception as e:
                logger.warning("coarse_eval_error", params=params, error=str(e))
                results.append((params, float('-inf')))

        return results

    def _fine_search(
        self,
        objective_fn,
        data_df: pl.DataFrame,
        top_configs: List[Dict[str, Any]],
        n_jobs: int = 1,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Perform fine search (Bayesian optimization) around top configs.

        Args:
            objective_fn: Objective function
            data_df: Input data
            top_configs: Top configurations from coarse phase
            n_jobs: Number of parallel jobs

        Returns:
            List of (params, score) tuples
        """
        results = []

        # Create a study with Bayesian optimization
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42)
        )

        # Define objective function for Optuna
        def optuna_objective(trial):
            params = {}
            search_space = self.param_space.get_space()

            # Suggest parameters based on space
            for param_name, distribution in search_space.items():
                if param_name in trial.params:
                    # Use existing parameter
                    params[param_name] = trial.params[param_name]
                else:
                    # Suggest new parameter
                    params[param_name] = trial._suggest(param_name, distribution)

            # Evaluate with full data
            try:
                score = objective_fn(params, data_df)
            except Exception as e:
                logger.warning("fine_eval_error", params=params, error=str(e))
                score = float('-inf')

            return score

        # Optimize for specified number of trials
        study.optimize(optuna_objective, n_trials=self.n_fine_samples)

        # Extract results
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                params = {name: trial.params[name] for name in trial.params}
                results.append((params, trial.value))

        return results

    def _select_top_configs(
        self,
        results: List[Tuple[Dict[str, Any], float]],
        k: int
    ) -> List[Dict[str, Any]]:
        """
        Select top K configurations.

        Args:
            results: List of (params, score) tuples
            k: Number of top configs to select

        Returns:
            List of top parameter dictionaries
        """
        # Sort by score descending
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
        return [params for params, score in sorted_results[:k]]

    def _evaluate_with_shorter_window(
        self,
        objective_fn,
        data_df: pl.DataFrame,
        params: Dict[str, Any],
    ) -> float:
        """
        Evaluate configuration with shorter data window for speed.

        Args:
            objective_fn: Objective function
            data_df: Full input data
            params: Parameters to evaluate

        Returns:
            Score from shortened evaluation
        """
        # Use last 30% of data for faster evaluation
        n_rows = len(data_df)
        start_idx = int(n_rows * 0.7)
        short_df = data_df.slice(start_idx, n_rows - start_idx)

        try:
            score = objective_fn(params, short_df)
        except Exception:
            score = float('-inf')

        return score
