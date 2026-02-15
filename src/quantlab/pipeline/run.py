"""
Main CLI entry point for QuantLab pipeline
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..common.logging import get_logger, setup_logging, set_run_id
from ..common.hashing import hash_spec, create_run_id
from ..data.pipeline import DataPipeline
from ..features.feature_store import FeatureStore
from ..features.universe import create_universe_provider
from ..features.signals import momentum_signal
from ..backtest.vectorbt_engine import VectorBTBacktestEngine
from ..optimize.runner import OptimizationRunner
from ..robustness.walk_forward import WalkForwardAnalysis
from ..robustness.sensitivity import SensitivityAnalysis
from ..robustness.bootstrap import BootstrapAnalysis
from ..robustness.regime_split import RegimeAnalysis
from ..robustness.leakage_checks import LeakageDetection
from ..report.build_report import ReportBuilder

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="QuantLab Multi-Market Quantitative Trading Platform"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Data update command
    update_parser = subparsers.add_parser("data:update", help="Update market data")
    update_parser.add_argument("--config", "-c", required=True, help="Strategy config JSON file")
    update_parser.add_argument("--output", "-o", default="results", help="Output directory")

    # Backtest command
    backtest_parser = subparsers.add_parser("backtest:run", help="Run backtest")
    backtest_parser.add_argument("--config", "-c", required=True, help="Strategy config JSON file")
    backtest_parser.add_argument("--output", "-o", default="results", help="Output directory")
    backtest_parser.add_argument("--dry-run", action="store_true", help="Validate config without execution")

    # Optimize command
    optimize_parser = subparsers.add_parser("optimize:run", help="Run optimization")
    optimize_parser.add_argument("--config", "-c", required=True, help="Strategy config JSON file")
    optimize_parser.add_argument("--output", "-o", default="results", help="Output directory")
    optimize_parser.add_argument("--trials", "-n", type=int, help="Number of optimization trials")
    optimize_parser.add_argument("--timeout", "-t", type=int, help="Timeout in seconds")

    # Robustness command
    robust_parser = subparsers.add_parser("robustness:run", help="Run robustness analysis")
    robust_parser.add_argument("--config", "-c", required=True, help="Strategy config JSON file")
    robust_parser.add_argument("--output", "-o", default="results", help="Output directory")
    robust_parser.add_argument("--walk-forward", action="store_true", help="Run walk-forward analysis")
    robust_parser.add_argument("--sensitivity", action="store_true", help="Run sensitivity analysis")
    robust_parser.add_argument("--bootstrap", action="store_true", help="Run bootstrap analysis")
    robust_parser.add_argument("--regime", action="store_true", help="Run regime analysis")
    robust_parser.add_argument("--leakage", action="store_true", help="Run leakage detection")

    # Report command
    report_parser = subparsers.add_parser("report:build", help="Build report")
    report_parser.add_argument("--run-id", "-r", required=True, help="Run ID to build report for")
    report_parser.add_argument("--output", "-o", default="results", help="Output directory")

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """
    Load and validate strategy configuration.

    Args:
        config_path: Path to config JSON file

    Returns:
        Configuration dictionary
    """
    config_path_obj = Path(config_path)

    if not config_path_obj.exists():
        logger.error("config_not_found", path=config_path)
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path_obj, 'r') as f:
        config = json.load(f)

    # Validate required fields
    required_fields = ["strategy_name", "instrument", "data", "backtest"]
    for field in required_fields:
        if field not in config:
            logger.error("missing_required_field", field=field)
            raise ValueError(f"Missing required field: {field}")

    return config


def command_data_update(args) -> int:
    """
    Execute data update command.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 = success, non-zero = error)
    """
    logger.info("command_start", command="data:update")

    try:
        # Load configuration
        spec = load_config(args.config)

        # Create results directory
        results_dir = Path(args.output)
        results_dir.mkdir(parents=True, exist_ok=True)

        # Create run ID
        run_id = create_run_id(spec, "pending", "pending")
        set_run_id(run_id)

        # Setup logging
        log_dir = results_dir / "logs"
        setup_logging(log_dir)

        logger.info("run_start", run_id=run_id, spec=spec["strategy_name"])

        # Initialize data pipeline
        data_pipeline = DataPipeline(Path("data"), spec)

        # Get universe
        universe_provider = create_universe_provider(spec)
        symbols = universe_provider.get_universe()

        if not symbols:
            logger.error("empty_universe")
            return 1

        # Fetch and store data
        start_date = datetime.fromisoformat(spec["data"].get("start_date", "2020-01-01"))
        end_date = datetime.fromisoformat(spec["data"].get("end_date", "2024-12-31"))
        freq = spec["data"]["frequency"]

        results = data_pipeline.fetch_and_store(
            symbols=symbols,
            start=start_date,
            end=end_date,
            freq=freq,
            forward_fill=True,
        )

        # Save run metadata
        metadata = {
            "run_id": run_id,
            "spec": spec,
            "data_fetch_results": results,
            "data_version": data_pipeline.get_data_version(),
            "code_version": "pending",  # Would be computed from git in production
            "completed_at": datetime.utcnow().isoformat(),
        }

        metadata_path = results_dir / run_id / "metadata.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info("command_complete", command="data:update", results=results)

        return 0

    except Exception as e:
        logger.exception("command_error", command="data:update", error=str(e))
        return 1


def command_backtest_run(args) -> int:
    """
    Execute backtest command.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code
    """
    logger.info("command_start", command="backtest:run")

    try:
        # Load configuration
        spec = load_config(args.config)

        if args.dry_run:
            logger.info("dry_run", config=spec)
            return 0

        # Create results directory
        results_dir = Path(args.output)
        results_dir.mkdir(parents=True, exist_ok=True)

        # Create run ID
        run_id = create_run_id(spec, "pending", "pending")
        set_run_id(run_id)

        # Setup logging
        log_dir = results_dir / "logs"
        setup_logging(log_dir)

        logger.info("run_start", run_id=run_id, spec=spec["strategy_name"])

        # Initialize data pipeline
        data_pipeline = DataPipeline(Path("data"), spec)

        # Get data
        universe_provider = create_universe_provider(spec)
        symbols = universe_provider.get_universe()

        data_df = data_pipeline.get_data(symbols=symbols)

        if data_df.is_empty():
            logger.error("no_data_available")
            return 1

        # Generate signals
        lookback = spec["features"].get("lookback_period", 20)
        signals_df = momentum_signal(
            data_df,
            period=lookback,
            long_threshold=spec["features"]["signals"].get("long_threshold", 0.02),
            short_threshold=spec["features"]["signals"].get("short_threshold", -0.02)
        )

        # Run backtest
        initial_capital = spec["backtest"]["initial_capital"]
        engine = VectorBTBacktestEngine(spec)
        signals = signals_df["signal"].to_numpy()

        backtest_results = engine.run_backtest(data_df, signals, initial_capital)

        # Save results
        results = {
            "run_id": run_id,
            "spec": spec,
            "backtest_results": {
                "metrics": backtest_results["metrics"],
            },
            "data_version": data_pipeline.get_data_version(),
            "code_version": "pending",
            "completed_at": datetime.utcnow().isoformat(),
        }

        results_path = results_dir / run_id / "results.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)

        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info("command_complete", command="backtest:run", metrics=backtest_results["metrics"])

        return 0

    except Exception as e:
        logger.exception("command_error", command="backtest:run", error=str(e))
        return 1


def command_optimize_run(args) -> int:
    """
    Execute optimization command.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code
    """
    logger.info("command_start", command="optimize:run")

    try:
        # Load configuration
        spec = load_config(args.config)

        # Create results directory
        results_dir = Path(args.output)
        results_dir.mkdir(parents=True, exist_ok=True)

        # Create run ID
        run_id = create_run_id(spec, "pending", "pending")
        set_run_id(run_id)

        # Setup logging
        log_dir = results_dir / "logs"
        setup_logging(log_dir)

        # Initialize data pipeline
        data_pipeline = DataPipeline(Path("data"), spec)

        # Get data
        universe_provider = create_universe_provider(spec)
        symbols = universe_provider.get_universe()
        data_df = data_pipeline.get_data(symbols=symbols)

        if data_df.is_empty():
            logger.error("no_data_available")
            return 1

        # Run optimization
        optimizer = OptimizationRunner(spec, data_df, results_dir)
        optimization_results = optimizer.run_optimization(
            n_trials=args.trials,
            timeout=args.timeout,
        )

        logger.info("command_complete", command="optimize:run")

        return 0

    except Exception as e:
        logger.exception("command_error", command="optimize:run", error=str(e))
        return 1


def command_robustness_run(args) -> int:
    """
    Execute robustness analysis command.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code
    """
    logger.info("command_start", command="robustness:run")

    try:
        # Load configuration
        spec = load_config(args.config)

        # Create results directory
        results_dir = Path(args.output)
        results_dir.mkdir(parents=True, exist_ok=True)

        # Create run ID
        run_id = create_run_id(spec, "pending", "pending")
        set_run_id(run_id)

        # Setup logging
        log_dir = results_dir / "logs"
        setup_logging(log_dir)

        # Initialize data pipeline
        data_pipeline = DataPipeline(Path("data"), spec)

        # Get data
        universe_provider = create_universe_provider(spec)
        symbols = universe_provider.get_universe()
        data_df = data_pipeline.get_data(symbols=symbols)

        if data_df.is_empty():
            logger.error("no_data_available")
            return 1

        robustness_results = {}

        # Run selected analyses
        if args.walk_forward:
            wf_analyzer = WalkForwardAnalysis(spec)
            robustness_results["walk_forward"] = wf_analyzer.run_analysis(
                df=data_df,
                strategy_func=lambda df, params: ([], {}),
                initial_capital=spec["backtest"]["initial_capital"]
            )

        if args.sensitivity:
            sens_analyzer = SensitivityAnalysis(spec)
            robustness_results["sensitivity"] = sens_analyzer.run_analysis(
                df=data_df,
                strategy_func=lambda df, params: ([], {}),
                metric_name="sharpe_ratio"
            )

        if args.bootstrap:
            boot_analyzer = BootstrapAnalysis(spec)
            robustness_results["bootstrap"] = boot_analyzer.run_analysis(
                df=data_df,
                strategy_func=lambda df, params: ([], {}),
                metric_name="sharpe_ratio"
            )

        if args.regime:
            regime_analyzer = RegimeAnalysis(spec)
            robustness_results["regime"] = regime_analyzer.run_analysis(
                df=data_df,
                strategy_func=lambda df, params: ([], {}),
            )

        if args.leakage:
            leakage_detector = LeakageDetection(spec)
            # This would need a feature_df - using data_df for now
            robustness_results["leakage"] = leakage_detector.run_all_checks(
                df=data_df,
                features_df=data_df  # Placeholder
            )

        logger.info("command_complete", command="robustness:run")

        return 0

    except Exception as e:
        logger.exception("command_error", command="robustness:run", error=str(e))
        return 1


def command_report_build(args) -> int:
    """
    Execute report build command.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code
    """
    logger.info("command_start", command="report:build")

    try:
        # Load results for run ID
        results_dir = Path(args.output)
        run_dir = results_dir / args.run_id

        if not run_dir.exists():
            logger.error("run_not_found", run_id=args.run_id)
            return 1

        # Load metadata
        metadata_path = run_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}

        # Load results
        results_path = run_dir / "results.json"
        if results_path.exists():
            with open(results_path, 'r') as f:
                results = json.load(f)
        else:
            results = {}

        # Build report
        report_builder = ReportBuilder(metadata.get("spec", {}), results_dir)
        html_report = report_builder.build_report(
            backtest_results=results.get("backtest_results", {}),
        )

        logger.info("command_complete", command="report:build", run_id=args.run_id)

        return 0

    except Exception as e:
        logger.exception("command_error", command="report:build", error=str(e))
        return 1


def main():
    """Main entry point."""
    args = parse_args()

    if not args.command:
        print("No command specified. Use --help for usage information.")
        return 1

    # Route to appropriate command
    command_handlers = {
        "data:update": command_data_update,
        "backtest:run": command_backtest_run,
        "optimize:run": command_optimize_run,
        "robustness:run": command_robustness_run,
        "report:build": command_report_build,
    }

    handler = command_handlers.get(args.command)
    if handler:
        return handler(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
