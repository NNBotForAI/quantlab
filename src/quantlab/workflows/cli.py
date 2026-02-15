#!/usr/bin/env python3
"""
QuantLab Strategy Research Workflow CLI

Command-line interface for running the complete strategy research lifecycle:
1. Idea Generation → 2. Factor Design → 3. Strategy Construction → 
4. Backtest Validation → 5. Parameter Optimization → 6. Robustness Testing
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

from quantlab.workflows.strategy_research_lifecycle import (
    StrategyResearchLifecycle, 
    StrategyResearchConfig,
    run_example_research_cycle
)
from quantlab.common.logging import get_logger, setup_logging

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="QuantLab Strategy Research Workflow - Complete Strategy Development Lifecycle"
    )
    
    # Subcommands for different operations
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Run complete research cycle
    run_parser = subparsers.add_parser("run", help="Run complete strategy research cycle")
    run_parser.add_argument(
        "--config", 
        "-c", 
        required=True, 
        help="Path to strategy research configuration JSON file"
    )
    run_parser.add_argument(
        "--output-dir", 
        "-o", 
        default="results", 
        help="Output directory for results (default: results)"
    )
    run_parser.add_argument(
        "--cache-enabled", 
        action="store_true", 
        help="Enable caching for computations (default: False)"
    )
    
    # Run specific stage
    stage_parser = subparsers.add_parser("run-stage", help="Run specific research stage")
    stage_parser.add_argument(
        "--config", 
        "-c", 
        required=True, 
        help="Path to strategy research configuration JSON file"
    )
    stage_parser.add_argument(
        "--stage", 
        "-s", 
        required=True, 
        choices=[
            "idea-generation", "factor-design", "strategy-construction", 
            "backtest-validation", "parameter-optimization", "robustness-validation"
        ],
        help="Specific stage to run"
    )
    stage_parser.add_argument(
        "--output-dir", 
        "-o", 
        default="results", 
        help="Output directory for results (default: results)"
    )
    
    # Example workflow
    example_parser = subparsers.add_parser("example", help="Run example strategy research")
    example_parser.add_argument(
        "--output-dir", 
        "-o", 
        default="results/example", 
        help="Output directory for example results (default: results/example)"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        logger.error("config_not_found", path=str(config_path))
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    return config


def create_strategy_config_from_workflow(workflow_config: dict, output_dir: str) -> StrategyResearchConfig:
    """Create StrategyResearchConfig from workflow configuration."""
    workflow = workflow_config["strategy_research_workflow"]
    
    # Extract main strategy parameters
    stages = workflow["stages"]
    
    config = StrategyResearchConfig(
        strategy_name=workflow.get("name", "default_strategy"),
        instrument=workflow.get("instrument", {
            "asset_type": "US_STOCK",
            "symbol": "SPY",
            "venue": "NYSE",
            "quote_currency": "USD",
            "lot_size": 1,
            "allow_fractional": True,
            "shortable": True,
            "leverage": 1
        }),
        data_config=workflow.get("data_config", {
            "frequency": "1D",
            "start_date": "2020-01-01",
            "end_date": "2024-12-31",
            "source": "yfinance"
        }),
        idea_generation=stages["stage1_idea_generation"]["config"],
        factor_design=stages["stage2_factor_design"]["config"],
        strategy_construction=stages["stage3_strategy_construction"]["config"],
        backtest_validation=stages["stage4_backtest_validation"]["config"],
        parameter_optimization=stages["stage5_parameter_optimization"]["config"],
        robustness_validation=stages["stage6_robustness_validation"]["config"],
        output_dir=Path(output_dir),
        cache_enabled=workflow.get("cache_enabled", True)
    )
    
    return config


def command_run_complete(args):
    """Run complete strategy research cycle."""
    logger.info("starting_complete_research_cycle", config=args.config)
    
    try:
        # Load workflow configuration
        workflow_config = load_config(args.config)
        
        # Create strategy research configuration
        strategy_config = create_strategy_config_from_workflow(workflow_config, args.output_dir)
        
        # Initialize research lifecycle
        lifecycle = StrategyResearchLifecycle(strategy_config)
        
        # Run complete research cycle
        results = lifecycle.run_complete_research_cycle()
        
        logger.info("research_cycle_completed", strategy=strategy_config.strategy_name)
        print(f"\n✅ Research cycle completed successfully!")
        print(f"Strategy: {results['strategy_name']}")
        print(f"Stages completed: {results['stages_completed']}")
        print(f"Results saved to: {args.output_dir}/")
        
        return 0
        
    except Exception as e:
        logger.exception("research_cycle_failed", error=str(e))
        print(f"\n❌ Research cycle failed: {str(e)}")
        return 1


def command_run_stage(args):
    """Run specific research stage."""
    logger.info("starting_specific_stage", stage=args.stage, config=args.config)
    
    try:
        # Load workflow configuration
        workflow_config = load_config(args.config)
        
        # Create strategy research configuration
        strategy_config = create_strategy_config_from_workflow(workflow_config, args.output_dir)
        
        # Initialize research lifecycle
        lifecycle = StrategyResearchLifecycle(strategy_config)
        
        # Map stage name to method
        stage_methods = {
            "idea-generation": lifecycle.stage1_idea_generation,
            "factor-design": lambda: lifecycle.stage2_factor_design([]),  # Need ideas input
            "strategy-construction": lambda: lifecycle.stage3_strategy_construction({}),  # Need factors input
            "backtest-validation": lambda: lifecycle.stage4_backtest_validation({}),  # Need strategy input
            "parameter-optimization": lambda: lifecycle.stage5_parameter_optimization({}, {}),  # Need strategy and results input
            "robustness-validation": lambda: lifecycle.stage6_robustness_validation({}, {})  # Need strategy and opt results input
        }
        
        # Special handling for stages that depend on previous results
        if args.stage == "idea-generation":
            results = lifecycle.stage1_idea_generation()
            print(f"\n✅ Idea generation completed! Generated {len(results)} ideas.")
        elif args.stage == "factor-design":
            # First generate ideas, then design factors
            ideas = lifecycle.stage1_idea_generation()
            results = lifecycle.stage2_factor_design(ideas)
            print(f"\n✅ Factor design completed! Created {sum(len(v) for v in results.values())} factors.")
        else:
            print(f"\n⚠️  Stage '{args.stage}' requires results from previous stages.")
            print("Consider running the complete workflow instead.")
            return 1
        
        print(f"Results saved to: {args.output_dir}/")
        return 0
        
    except Exception as e:
        logger.exception("stage_run_failed", stage=args.stage, error=str(e))
        print(f"\n❌ Stage run failed: {str(e)}")
        return 1


def command_example(args):
    """Run example strategy research."""
    logger.info("running_example_research")
    
    try:
        # Setup logging
        log_dir = Path(args.output_dir) / "logs"
        setup_logging(log_dir)
        
        print("🚀 Running example strategy research cycle...")
        print("This will demonstrate the complete workflow with a momentum strategy.")
        
        # Run example (this uses hardcoded example config)
        results = run_example_research_cycle()
        
        logger.info("example_completed", strategy=results['strategy_name'])
        print(f"\n✅ Example completed successfully!")
        print(f"Strategy: {results['strategy_name']}")
        print(f"Stages completed: {results['stages_completed']}")
        print(f"Results saved to: {args.output_dir}/")
        
        return 0
        
    except Exception as e:
        logger.exception("example_failed", error=str(e))
        print(f"\n❌ Example failed: {str(e)}")
        return 1


def main():
    """Main entry point."""
    args = parse_args()
    
    if not args.command:
        print("No command specified. Use --help for usage information.")
        return 1
    
    # Setup logging
    setup_logging(Path("logs"))
    
    # Route to appropriate command
    command_handlers = {
        "run": command_run_complete,
        "run-stage": command_run_stage,
        "example": command_example,
    }
    
    handler = command_handlers.get(args.command)
    if handler:
        return handler(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
