"""
Pipeline stage definitions and orchestration
"""
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional

from ..common.logging import get_logger

logger = get_logger(__name__)


class Stage(Enum):
    """Pipeline stages."""
    DATA_UPDATE = "data:update"
    FEATURE_COMPUTE = "feature_compute"
    BACKTEST_RUN = "backtest:run"
    OPTIMIZE_RUN = "optimize:run"
    ROBUSTNESS_RUN = "robustness:run"
    REPORT_BUILD = "report:build"


@dataclass
class StageConfig:
    """Configuration for a pipeline stage."""
    name: Stage
    enabled: bool
    required: bool = False
    dependencies: list[Stage] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class PipelineOrchestrator:
    """
    Orchestrate pipeline stages with dependency management.
    """

    def __init__(self, spec: dict, results_dir: Path):
        """
        Initialize pipeline orchestrator.

        Args:
            spec: Strategy specification
            results_dir: Results directory
        """
        self.spec = spec
        self.results_dir = results_dir
        self.stages = self._define_stages()

    def _define_stages(self) -> Dict[Stage, StageConfig]:
        """
        Define pipeline stages based on spec.

        Returns:
            Dictionary mapping stages to configurations
        """
        stages = {
            Stage.DATA_UPDATE: StageConfig(
                name=Stage.DATA_UPDATE,
                enabled=True,
                required=True,
            ),
            Stage.FEATURE_COMPUTE: StageConfig(
                name=Stage.FEATURE_COMPUTE,
                enabled=self.spec.get("features", {}).get("enabled", True),
                required=False,
                dependencies=[Stage.DATA_UPDATE],
            ),
            Stage.BACKTEST_RUN: StageConfig(
                name=Stage.BACKTEST_RUN,
                enabled=self.spec.get("backtest", {}).get("enabled", True),
                required=False,
                dependencies=[Stage.DATA_UPDATE, Stage.FEATURE_COMPUTE],
            ),
            Stage.OPTIMIZE_RUN: StageConfig(
                name=Stage.OPTIMIZE_RUN,
                enabled=self.spec.get("optimization", {}).get("enabled", False),
                required=False,
                dependencies=[Stage.DATA_UPDATE],
            ),
            Stage.ROBUSTNESS_RUN: StageConfig(
                name=Stage.ROBUSTNESS_RUN,
                enabled=self.spec.get("validation", {}).get("enabled", False),
                required=False,
                dependencies=[Stage.BACKTEST_RUN],
            ),
            Stage.REPORT_BUILD: StageConfig(
                name=Stage.REPORT_BUILD,
                enabled=self.spec.get("report", {}).get("enabled", True),
                required=False,
                dependencies=[Stage.BACKTEST_RUN, Stage.ROBUSTNESS_RUN],
            ),
        }

        return stages

    def get_execution_order(self) -> list[Stage]:
        """
        Get topological sort of stages based on dependencies.

        Returns:
            List of stages in execution order
        """
        # Topological sort
        visited = set()
        execution_order = []

        def visit(stage: Stage):
            if stage in visited:
                return

            visited.add(stage)

            stage_config = self.stages[stage]

            # Visit dependencies first
            for dep in stage_config.dependencies:
                visit(dep)

            execution_order.append(stage)

        # Visit all stages
        for stage in self.stages:
            if self.stages[stage].enabled:
                visit(stage)

        return execution_order

    def validate_dependencies(self) -> bool:
        """
        Validate that all dependencies are met.

        Returns:
            True if dependencies are valid
        """
        for stage_name, stage_config in self.stages.items():
            if not stage_config.enabled:
                continue

            for dep in stage_config.dependencies:
                if dep not in self.stages:
                    logger.error(
                        "invalid_dependency",
                        stage=stage_name.value,
                        dependency=dep.value
                    )
                    return False

                if not self.stages[dep].enabled:
                    logger.warning(
                        "dependency_disabled",
                        stage=stage_name.value,
                        dependency=dep.value
                    )

        return True

    def get_stage_status(
        self,
        run_id: str,
    ) -> Dict[Stage, str]:
        """
        Get status of all stages for a run.

        Args:
            run_id: Run ID

        Returns:
            Dictionary mapping stages to status strings
        """
        run_dir = self.results_dir / run_id

        if not run_dir.exists():
            return {stage: "pending" for stage in self.stages}

        status = {}
        for stage_name in self.stages:
            stage_marker = run_dir / f".{stage_name.value}.complete"

            if stage_marker.exists():
                status[stage_name] = "complete"
            else:
                status[stage_name] = "pending"

        return status

    def mark_stage_complete(self, run_id: str, stage: Stage) -> None:
        """
        Mark a stage as complete.

        Args:
            run_id: Run ID
            stage: Stage to mark as complete
        """
        run_dir = self.results_dir / run_id
        stage_marker = run_dir / f".{stage.value}.complete"

        with open(stage_marker, 'w') as f:
            f.write(f"{datetime.utcnow().isoformat()}\n")

        logger.info("stage_complete", stage=stage.value, run_id=run_id)

    def get_stage_output_dir(
        self,
        run_id: str,
        stage: Stage,
    ) -> Path:
        """
        Get output directory for a stage.

        Args:
            run_id: Run ID
            stage: Stage

        Returns:
            Path to stage output directory
        """
        run_dir = self.results_dir / run_id
        stage_dir = run_dir / stage.value

        stage_dir.mkdir(parents=True, exist_ok=True)

        return stage_dir

    def save_stage_metadata(
        self,
        run_id: str,
        stage: Stage,
        metadata: Dict[str, Any],
    ) -> None:
        """
        Save metadata for a stage.

        Args:
            run_id: Run ID
            stage: Stage
            metadata: Metadata to save
        """
        stage_dir = self.get_stage_output_dir(run_id, stage)
        metadata_path = stage_dir / "metadata.json"

        with open(metadata_path, 'w') as f:
            import json
            json.dump(metadata, f, indent=2, default=str)

        logger.info("stage_metadata_saved", stage=stage.value, run_id=run_id)

    def run_pipeline(self, run_id: str) -> int:
        """
        Run the full pipeline.

        Args:
            run_id: Run ID

        Returns:
            Exit code (0 = success, non-zero = error)
        """
        logger.info("pipeline_start", run_id=run_id)

        # Validate dependencies
        if not self.validate_dependencies():
            logger.error("dependency_validation_failed")
            return 1

        # Get execution order
        execution_order = self.get_execution_order()

        logger.info("pipeline_execution_order", stages=[s.value for s in execution_order])

        # Run each stage
        for stage in execution_order:
            logger.info("pipeline_stage_start", stage=stage.value)

            try:
                # Import and run appropriate stage function
                from .run import (
                    command_data_update,
                    command_backtest_run,
                    command_optimize_run,
                )

                # Map stages to commands
                stage_commands = {
                    Stage.DATA_UPDATE: ("data:update", command_data_update),
                    Stage.BACKTEST_RUN: ("backtest:run", command_backtest_run),
                    Stage.OPTIMIZE_RUN: ("optimize:run", command_optimize_run),
                }

                if stage in stage_commands:
                    cmd_name, cmd_func = stage_commands[stage]

                    # Create args object
                    class Args:
                        def __init__(self):
                            self.config = "config.json"  # Would be spec path in practice
                            self.output = str(self.results_dir)

                    # Run command
                    exit_code = cmd_func(Args())

                    if exit_code != 0:
                        logger.error("pipeline_stage_failed", stage=stage.value, exit_code=exit_code)
                        return 1

                # Mark stage as complete
                self.mark_stage_complete(run_id, stage)

                logger.info("pipeline_stage_complete", stage=stage.value)

            except Exception as e:
                logger.exception("pipeline_stage_error", stage=stage.value, error=str(e))
                return 1

        logger.info("pipeline_complete", run_id=run_id)
        return 0
