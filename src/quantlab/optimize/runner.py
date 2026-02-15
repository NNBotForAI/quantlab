"""
Optimization runner with parallel execution
"""
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import joblib
import optuna
import polars as pl

from .coarse_to_fine import CoarseToFineOptimizer
from .optuna_objective import create_objective_function
from ..common.logging import get_logger
from ..common.hashing import create_run_id

logger = get_logger(__name__)


class OptimizationRunner:
    """
    Run optimization with parallel execution support.
    """

    def __init__(self, spec: dict, data_df: pl.DataFrame, results_dir: Path):
        """
        Initialize optimization runner.

        Args:
            spec: Strategy specification
            data_df: Input data
            results_dir: Results directory
        """
        self.spec = spec
        self.data_df = data_df
        self.results_dir = results_dir
        self.run_id = create_run_id(spec, "dummy_data_version", "dummy_code_version")

        # Performance settings
        self.parallel_backend = spec["performance"]["parallel_backend"]
        self.max_workers = spec["performance"]["max_workers"]

    def run_optimization(
        self,
        n_trials: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run optimization.

        Args:
            n_trials: Number of optimization trials (defaults to spec)
            timeout: Timeout in seconds (defaults to spec)

        Returns:
            Dictionary with optimization results
        """
        logger.info("optimization_start", run_id=self.run_id)

        # Get number of trials from spec or parameter
        if n_trials is None:
            n_trials = self.spec.get("optimization", {}).get("n_trials", 100)

        if timeout is None:
            timeout = self.spec.get("optimization", {}).get("timeout", 3600)  # 1 hour

        # Determine if using coarse-to-fine approach
        use_coarse_to_fine = self.spec.get("optimization", {}).get("coarse_to_fine", True)

        if use_coarse_to_fine:
            # Use coarse-to-fine approach
            optimizer = CoarseToFineOptimizer(self.spec)
            objective_fn = self._create_simple_objective()

            best_params, best_score, all_trials = optimizer.optimize(
                objective_fn=objective_fn,
                data_df=self.data_df,
                n_jobs=self.max_workers,
            )
        else:
            # Use direct Optuna optimization
            best_params, best_score, all_trials = self._run_direct_optimization(
                n_trials=n_trials,
                timeout=timeout,
            )

        # Save results
        results = {
            "best_params": best_params,
            "best_score": best_score,
            "all_trials": all_trials,
            "run_id": self.run_id,
            "completed_at": datetime.utcnow().isoformat(),
        }

        self._save_results(results)

        logger.info("optimization_complete", run_id=self.run_id, best_score=best_score)

        return results

    def _create_simple_objective(self):
        """
        Create a simplified objective function for coarse-to-fine.
        """
        def objective(params, data_df):
            # This would be a simplified version that runs faster
            # For now, using momentum as example
            from .optuna_objective import momentum_objective
            return momentum_objective(params, data_df)

        return objective

    def _run_direct_optimization(
        self,
        n_trials: int,
        timeout: int,
    ) -> tuple:
        """
        Run direct Optuna optimization.

        Args:
            n_trials: Number of trials
            timeout: Timeout in seconds

        Returns:
            Tuple of (best_params, best_score, all_trials)
        """
        # Create objective function
        objective_fn = create_objective_function(self.spec, self.data_df)

        # Create study
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
        )

        # Parallel execution settings
        if self.parallel_backend == "joblib":
            n_jobs = self.max_workers
        elif self.parallel_backend == "ray":
            # Ray parallelization would require additional setup
            n_jobs = self.max_workers
            logger.warning("ray_parallelization_not_implemented", fallback="using_joblib")
            n_jobs = self.max_workers
        else:
            n_jobs = 1

        logger.info("running_optimization", n_trials=n_trials, n_jobs=n_jobs)

        # Optimize
        if n_jobs == 1:
            # Sequential optimization
            study.optimize(
                objective_fn,
                n_trials=n_trials,
                timeout=timeout,
            )
        else:
            # Parallel optimization with joblib
            def run_trial(trial_id):
                trial = study.ask()
                try:
                    value = objective_fn(trial)
                    study.tell(trial, value)
                    return trial_id, value
                except Exception as e:
                    logger.warning("trial_failed", trial_id=trial_id, error=str(e))
                    study.tell(trial, float('-inf'))
                    return trial_id, float('-inf')

            # Run trials in parallel
            joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(run_trial)(i) for i in range(n_trials)
            )

        # Extract results
        best_params = study.best_params
        best_score = study.best_value

        # Get all trials
        all_trials = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                params = {name: trial.params[name] for name in trial.params}
                all_trials.append((params, trial.value))

        return best_params, best_score, all_trials

    def _save_results(self, results: Dict[str, Any]) -> None:
        """
        Save optimization results.

        Args:
            results: Optimization results dictionary
        """
        import json

        results_path = self.results_dir / self.run_id / "optimization_results.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert any non-serializable objects
        serializable_results = self._make_serializable(results)

        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)

    def _make_serializable(self, obj):
        """
        Convert object to JSON serializable format.

        Args:
            obj: Object to convert

        Returns:
            Serializable version
        """
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._make_serializable(list(obj)))
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        else:
            return obj
