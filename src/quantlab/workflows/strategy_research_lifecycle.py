"""
QuantLab Strategy Research Lifecycle
====================================

A complete workflow for strategy research combining insights from quant-learning-24h
with the QuantLab platform capabilities.

This module implements a 6-stage process:
1. Idea Generation → 2. Factor Design → 3. Strategy Construction → 
4. Backtest Validation → 5. Parameter Optimization → 6. Robustness Testing

Each stage builds upon the quant-learning-24h knowledge base while leveraging
QuantLab's performance-optimized architecture.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import numpy as np
import polars as pl
from dataclasses import dataclass

from src.quantlab.data.pipeline import DataPipeline
from src.quantlab.features.indicators import (
    momentum, volatility, rsi, sma, ema, bollinger_bands, atr
)
from src.quantlab.features.signals import momentum_signal
from src.quantlab.features.universe import create_universe_provider
from src.quantlab.backtest.vectorbt_engine import VectorBTBacktestEngine
from src.quantlab.backtest.metrics import calculate_all_metrics
from src.quantlab.optimize.coarse_to_fine import CoarseToFineOptimizer
from src.quantlab.robustness.walk_forward import WalkForwardAnalysis
from src.quantlab.robustness.bootstrap import BootstrapAnalysis
from src.quantlab.robustness.sensitivity import SensitivityAnalysis
from src.quantlab.robustness.regime_split import RegimeAnalysis
from src.quantlab.robustness.leakage_checks import LeakageDetection
from src.quantlab.report.build_report import ReportBuilder
from src.quantlab.common.logging import get_logger, setup_logging
from src.quantlab.common.hashing import hash_spec, create_run_id
from src.quantlab.common.cache import FeatureCache


logger = get_logger(__name__)


@dataclass
class StrategyResearchConfig:
    """Configuration for strategy research lifecycle."""
    
    # Basic strategy info
    strategy_name: str
    instrument: Dict[str, Any]
    
    # Data requirements
    data_config: Dict[str, Any]
    
    # Research stages configuration
    idea_generation: Dict[str, Any]
    factor_design: Dict[str, Any]
    strategy_construction: Dict[str, Any]
    backtest_validation: Dict[str, Any]
    parameter_optimization: Dict[str, Any]
    robustness_validation: Dict[str, Any]
    
    # Output settings
    output_dir: Path = Path("results")
    cache_enabled: bool = True


class StrategyResearchLifecycle:
    """
    Complete strategy research lifecycle implementation.
    
    Combines insights from quant-learning-24h with QuantLab platform
    to create a systematic approach to strategy development.
    """
    
    def __init__(self, config: StrategyResearchConfig):
        """
        Initialize the strategy research lifecycle.
        
        Args:
            config: Strategy research configuration
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        log_dir = self.output_dir / "logs"
        setup_logging(log_dir)
        
        # Initialize components
        self.data_pipeline = DataPipeline(
            data_dir=Path("data"),
            spec={"instrument": config.instrument, "data": config.data_config}
        )
        self.cache = FeatureCache(
            cache_dir=self.output_dir / "cache",
            run_id=create_run_id(config.__dict__, "v1", "v1")
        )
        
        logger.info("strategy_research_initialized", strategy_name=config.strategy_name)
    
    def stage1_idea_generation(self) -> List[Dict[str, Any]]:
        """
        Stage 1: Idea Generation and Inspiration Mining
        
        Based on quant-learning-24h insights:
        - Technical analysis schools (trend/oscillation/form)
        - Fundamental analysis schools (value/growth/quality)
        - Behavioral finance schools (momentum/reversal/smart money)
        - JoinQuant strategies (successful strategy learning methods)
        """
        logger.info("stage_start", stage="idea_generation", strategy=self.config.strategy_name)
        
        # Generate strategy ideas based on market types
        ideas = []
        
        # Technical analysis ideas
        if self.config.idea_generation.get("technical_analysis", True):
            tech_ideas = self._generate_technical_ideas()
            ideas.extend(tech_ideas)
        
        # Fundamental analysis ideas
        if self.config.idea_generation.get("fundamental_analysis", False):
            fundamental_ideas = self._generate_fundamental_ideas()
            ideas.extend(fundamental_ideas)
        
        # Behavioral finance ideas
        if self.config.idea_generation.get("behavioral_finance", True):
            behavioral_ideas = self._generate_behavioral_ideas()
            ideas.extend(behavioral_ideas)
        
        # Market-specific ideas
        asset_type = self.config.instrument["asset_type"]
        if asset_type == "CN_STOCK":
            cn_specific = self._generate_cn_stock_ideas()
            ideas.extend(cn_specific)
        elif asset_type == "US_STOCK":
            us_specific = self._generate_us_stock_ideas()
            ideas.extend(us_specific)
        elif asset_type.startswith("CRYPTO"):
            crypto_specific = self._generate_crypto_ideas()
            ideas.extend(crypto_specific)
        
        # Save ideas
        ideas_path = self.output_dir / f"{self.config.strategy_name}_ideas.json"
        with open(ideas_path, 'w', encoding='utf-8') as f:
            json.dump(ideas, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info("stage_complete", stage="idea_generation", ideas_generated=len(ideas))
        return ideas
    
    def _generate_technical_ideas(self) -> List[Dict[str, Any]]:
        """Generate technical analysis strategy ideas."""
        ideas = []
        
        # Trend following ideas
        trend_ideas = [
            {
                "name": "Dual Moving Average Crossover",
                "category": "trend",
                "description": "Buy when fast MA crosses above slow MA, sell when reversed",
                "logic": "fast_ma > slow_ma",
                "parameters": {"fast_period": 20, "slow_period": 60}
            },
            {
                "name": "Momentum Breakout",
                "category": "trend",
                "description": "Enter positions when price breaks above recent highs/lows",
                "logic": "price > rolling_max(high, period)",
                "parameters": {"breakout_period": 20}
            }
        ]
        ideas.extend(trend_ideas)
        
        # Oscillation ideas
        osc_ideas = [
            {
                "name": "RSI Mean Reversion",
                "category": "oscillation",
                "description": "Buy oversold assets (RSI < 30), sell overbought (RSI > 70)",
                "logic": "rsi < 30 or rsi > 70",
                "parameters": {"rsi_period": 14, "oversold": 30, "overbought": 70}
            },
            {
                "name": "Bollinger Band Reversion",
                "category": "oscillation",
                "description": "Buy when price touches lower band, sell when upper band",
                "logic": "price < lower_bb or price > upper_bb",
                "parameters": {"bb_period": 20, "bb_std": 2.0}
            }
        ]
        ideas.extend(osc_ideas)
        
        # Pattern recognition ideas
        pattern_ideas = [
            {
                "name": "MACD Divergence",
                "category": "pattern",
                "description": "Look for divergences between price and MACD histogram",
                "logic": "price makes new high but MACD doesn't",
                "parameters": {"macd_fast": 12, "macd_slow": 26, "macd_signal": 9}
            }
        ]
        ideas.extend(pattern_ideas)
        
        return ideas
    
    def _generate_fundamental_ideas(self) -> List[Dict[str, Any]]:
        """Generate fundamental analysis strategy ideas."""
        # Note: This would require fundamental data which we'll simulate
        ideas = []
        
        # Value investing ideas
        value_ideas = [
            {
                "name": "Low PE Ratio Strategy",
                "category": "value",
                "description": "Buy stocks with low PE ratios compared to industry average",
                "logic": "pe_ratio < industry_pe_avg * 0.8",
                "parameters": {"pe_threshold": 0.8}
            },
            {
                "name": "High Dividend Yield",
                "category": "value",
                "description": "Invest in high dividend yield stocks",
                "logic": "dividend_yield > threshold",
                "parameters": {"yield_threshold": 0.03}
            }
        ]
        ideas.extend(value_ideas)
        
        return ideas
    
    def _generate_behavioral_ideas(self) -> List[Dict[str, Any]]:
        """Generate behavioral finance strategy ideas."""
        ideas = []
        
        # Momentum ideas
        momentum_ideas = [
            {
                "name": "Price Momentum",
                "category": "behavioral",
                "description": "Follow price trends based on historical momentum",
                "logic": "recent_price_change > threshold",
                "parameters": {"momentum_period": 20, "momentum_threshold": 0.05}
            },
            {
                "name": "Earnings Momentum",
                "category": "behavioral",
                "description": "Follow earnings surprises and revisions",
                "logic": "earnings_revision > threshold",
                "parameters": {"revision_threshold": 0.05}
            }
        ]
        ideas.extend(momentum_ideas)
        
        # Smart money ideas
        smart_money_ideas = [
            {
                "name": "Institutional Holdings",
                "category": "behavioral",
                "description": "Follow institutional buying/selling patterns",
                "logic": "institutional_flow > threshold",
                "parameters": {"flow_threshold": 0.05}
            }
        ]
        ideas.extend(smart_money_ideas)
        
        return ideas
    
    def _generate_cn_stock_ideas(self) -> List[Dict[str, Any]]:
        """Generate China stock market specific ideas."""
        ideas = []
        
        # T+1 specific ideas
        t_plus_1_ideas = [
            {
                "name": "T+1 Arbitrage",
                "category": "china_specific",
                "description": "Exploit T+1 settlement restrictions",
                "logic": "buy_low_sell_high_same_day",
                "parameters": {"volatility_threshold": 0.03}
            },
            {
                "name": "IPO Strategy",
                "category": "china_specific",
                "description": "Trade IPOs on first day listing",
                "logic": "first_day_listing",
                "parameters": {"listing_period": 1}
            }
        ]
        ideas.extend(t_plus_1_ideas)
        
        return ideas
    
    def _generate_us_stock_ideas(self) -> List[Dict[str, Any]]:
        """Generate US stock market specific ideas."""
        ideas = []
        
        # Market structure ideas
        market_structure_ideas = [
            {
                "name": "Market Hours Strategy",
                "category": "us_specific",
                "description": "Exploit pre-market and after-hours patterns",
                "logic": "pre_market_movement > threshold",
                "parameters": {"hours_filter": "pre_market"}
            }
        ]
        ideas.extend(market_structure_ideas)
        
        return ideas
    
    def _generate_crypto_ideas(self) -> List[Dict[str, Any]]:
        """Generate cryptocurrency specific ideas."""
        ideas = []
        
        # 24/7 market ideas
        crypto_24_7_ideas = [
            {
                "name": "Weekend Effect",
                "category": "crypto_specific",
                "description": "Exploit weekend vs weekday volatility differences",
                "logic": "weekend_vs_weekday_diff",
                "parameters": {"day_of_week_filter": "monday"}
            },
            {
                "name": "Liquidity Mining",
                "category": "crypto_specific",
                "description": "Arbitrage liquidity provision opportunities",
                "logic": "liquidity_incentives > threshold",
                "parameters": {"incentive_threshold": 0.01}
            }
        ]
        ideas.extend(crypto_24_7_ideas)
        
        return ideas
    
    def stage2_factor_design(self, selected_ideas: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Stage 2: Factor Design and Feature Engineering
        
        Based on quant-learning-24h insights:
        - Alpha101 factors from market theory
        - Fama-French 3/5-factor models
        - JoinQuant technical indicators
        """
        logger.info("stage_start", stage="factor_design", strategy=self.config.strategy_name)
        
        # Select factors based on chosen ideas
        factors = {
            "momentum_factors": [],
            "reversal_factors": [],
            "volatility_factors": [],
            "value_factors": [],
            "quality_factors": [],
            "smart_money_factors": [],
            "technical_factors": [],
            "sentiment_factors": []
        }
        
        # Get universe data
        universe_provider = create_universe_provider({
            "instrument": self.config.instrument,
            "data": self.config.data_config
        })
        symbols = universe_provider.get_universe()
        
        # Fetch data for factor computation
        start_date = datetime.fromisoformat(self.config.data_config.get("start_date", "2020-01-01"))
        end_date = datetime.fromisoformat(self.config.data_config.get("end_date", "2024-12-31"))
        freq = self.config.data_config["frequency"]
        
        data_df = self.data_pipeline.get_data(
            symbols=symbols,
            start=start_date,
            end=end_date,
            freq=freq
        )
        
        if data_df.is_empty():
            logger.error("no_data_for_factor_computation")
            return factors
        
        # Compute momentum factors
        momentum_params = self.config.factor_design.get("momentum_factors", {})
        if momentum_params.get("enabled", True):
            periods = momentum_params.get("periods", [5, 10, 20, 60, 120])
            for period in periods:
                factor_name = f"momentum_{period}d"
                
                # Use cached computation if available
                spec_hash = hash_spec({"type": "momentum", "period": period})
                factor_data = self.cache.get_or_compute(
                    spec_hash=spec_hash,
                    data_version=self.data_pipeline.get_data_version(),
                    code_version="v1",
                    stage="factor_computation",
                    compute_fn=lambda: momentum(data_df, period=period),
                    use_cache=self.config.cache_enabled
                )
                
                factors["momentum_factors"].append({
                    "name": factor_name,
                    "data": factor_data,
                    "period": period,
                    "computation_time": datetime.now().isoformat()
                })
        
        # Compute volatility factors
        volatility_params = self.config.factor_design.get("volatility_factors", {})
        if volatility_params.get("enabled", True):
            periods = volatility_params.get("periods", [10, 20, 60])
            for period in periods:
                factor_name = f"volatility_{period}d"
                
                spec_hash = hash_spec({"type": "volatility", "period": period})
                factor_data = self.cache.get_or_compute(
                    spec_hash=spec_hash,
                    data_version=self.data_pipeline.get_data_version(),
                    code_version="v1",
                    stage="factor_computation",
                    compute_fn=lambda: volatility(data_df, period=period),
                    use_cache=self.config.cache_enabled
                )
                
                factors["volatility_factors"].append({
                    "name": factor_name,
                    "data": factor_data,
                    "period": period,
                    "computation_time": datetime.now().isoformat()
                })
        
        # Compute technical factors (RSI, Bollinger Bands, etc.)
        technical_params = self.config.factor_design.get("technical_factors", {})
        if technical_params.get("enabled", True):
            # RSI
            rsi_period = technical_params.get("rsi_period", 14)
            if rsi_period:
                spec_hash = hash_spec({"type": "rsi", "period": rsi_period})
                rsi_data = self.cache.get_or_compute(
                    spec_hash=spec_hash,
                    data_version=self.data_pipeline.get_data_version(),
                    code_version="v1",
                    stage="factor_computation",
                    compute_fn=lambda: rsi(data_df, period=rsi_period, use_numba=True),
                    use_cache=self.config.cache_enabled
                )
                
                factors["technical_factors"].append({
                    "name": f"rsi_{rsi_period}d",
                    "data": rsi_data,
                    "period": rsi_period,
                    "computation_time": datetime.now().isoformat()
                })
            
            # Bollinger Bands
            bb_params = technical_params.get("bollinger_bands", {})
            if bb_params.get("enabled", True):
                bb_period = bb_params.get("period", 20)
                bb_std = bb_params.get("std_dev", 2.0)
                
                spec_hash = hash_spec({"type": "bollinger", "period": bb_period, "std": bb_std})
                bb_data = self.cache.get_or_compute(
                    spec_hash=spec_hash,
                    data_version=self.data_pipeline.get_data_version(),
                    code_version="v1",
                    stage="factor_computation",
                    compute_fn=lambda: bollinger_bands(data_df, period=bb_period, std_dev=bb_std),
                    use_cache=self.config.cache_enabled
                )
                
                factors["technical_factors"].append({
                    "name": f"bollinger_{bb_period}d_{bb_std}std",
                    "data": bb_data,
                    "period": bb_period,
                    "std_dev": bb_std,
                    "computation_time": datetime.now().isoformat()
                })
        
        # Save factors
        factors_path = self.output_dir / f"{self.config.strategy_name}_factors.json"
        with open(factors_path, 'w', encoding='utf-8') as f:
            # Convert polars DataFrames to dict for JSON serialization
            serializable_factors = {}
            for category, factor_list in factors.items():
                serializable_factors[category] = []
                for factor in factor_list:
                    serializable_factor = factor.copy()
                    if "data" in serializable_factor and isinstance(serializable_factor["data"], pl.DataFrame):
                        serializable_factor["data"] = serializable_factor["data"].to_dict()
                    serializable_factors[category].append(serializable_factor)
            
            json.dump(serializable_factors, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info("stage_complete", stage="factor_design", factors_created=len(sum(factors.values(), [])))
        return factors
    
    def stage3_strategy_construction(self, factors: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 3: Strategy Construction
        
        Based on quant-learning-24h insights:
        - JoinQuant strategy creation methods
        - Quantitative trading system implementation
        """
        logger.info("stage_start", stage="strategy_construction", strategy=self.config.strategy_name)
        
        # Determine strategy type based on configuration
        strategy_type = self.config.strategy_construction.get("type", "single_factor")
        
        if strategy_type == "single_factor":
            strategy = self._construct_single_factor_strategy(factors)
        elif strategy_type == "multi_factor":
            strategy = self._construct_multi_factor_strategy(factors)
        elif strategy_type == "rotation":
            strategy = self._construct_rotation_strategy(factors)
        elif strategy_type == "timing":
            strategy = self._construct_timing_strategy(factors)
        else:
            strategy = self._construct_single_factor_strategy(factors)
        
        # Save strategy
        strategy_path = self.output_dir / f"{self.config.strategy_name}_strategy.json"
        with open(strategy_path, 'w', encoding='utf-8') as f:
            json.dump(strategy, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info("stage_complete", stage="strategy_construction", strategy_type=strategy_type)
        return strategy
    
    def _construct_single_factor_strategy(self, factors: Dict[str, Any]) -> Dict[str, Any]:
        """Construct a single-factor strategy."""
        # Select the primary factor based on configuration
        primary_factor = self.config.strategy_construction.get("primary_factor", "momentum_20d")
        
        # Find the factor in our computed factors
        factor_data = None
        factor_name = None
        
        for category, factor_list in factors.items():
            for factor in factor_list:
                if factor["name"] == primary_factor:
                    factor_data = factor["data"]
                    factor_name = factor["name"]
                    break
            if factor_data is not None:
                break
        
        if factor_data is None:
            logger.warning("primary_factor_not_found", factor=primary_factor)
            # Use first available momentum factor
            if factors["momentum_factors"]:
                factor_data = factors["momentum_factors"][0]["data"]
                factor_name = factors["momentum_factors"][0]["name"]
            else:
                raise ValueError(f"No suitable factor found for strategy construction")
        
        # Generate signals based on the factor
        entry_threshold = self.config.strategy_construction.get("entry_threshold", 0.02)
        exit_threshold = self.config.strategy_construction.get("exit_threshold", -0.01)
        
        # Convert factor data to signals
        if isinstance(factor_data, pl.DataFrame):
            signals_df = factor_data.with_columns([
                pl.when(pl.col(factor_name.split('_')[0]) > entry_threshold)
                 .then(1)
                 .when(pl.col(factor_name.split('_')[0]) < exit_threshold)
                 .then(-1)
                 .otherwise(0)
                 .alias("signal")
            ])
        else:
            # If factor_data is dict (from JSON), convert appropriately
            signals_df = pl.DataFrame({factor_name.split('_')[0]: [0] * 10})  # placeholder
            signals_df = signals_df.with_columns([
                pl.lit(0).alias("signal")
            ])
        
        # Apply shift to prevent lookahead bias
        from src.quantlab.features.signals import shift_enforce
        signals_df = shift_enforce(signals_df, columns=["signal"], periods=1)
        
        strategy = {
            "type": "single_factor",
            "factor_used": factor_name,
            "entry_threshold": entry_threshold,
            "exit_threshold": exit_threshold,
            "signals": signals_df.to_dict(),
            "position_size": self.config.strategy_construction.get("position_size", 1.0),
            "max_position": self.config.strategy_construction.get("max_position", 10)
        }
        
        return strategy
    
    def _construct_multi_factor_strategy(self, factors: Dict[str, Any]) -> Dict[str, Any]:
        """Construct a multi-factor strategy."""
        # Combine multiple factors using weights
        factor_weights = self.config.strategy_construction.get("factor_weights", {})
        
        # For simplicity, we'll combine momentum and volatility factors
        combined_signals = None
        
        # Weighted combination of factors
        weight_sum = 0
        for factor_category, weight in factor_weights.items():
            if factor_category in factors and factors[factor_category]:
                factor_data = factors[factor_category][0]["data"]  # Use first factor in category
                # Normalize the factor data
                normalized = factor_data.select([
                    pl.col("*").apply(lambda x: (x - x.mean()) / x.std())
                ])
                
                if combined_signals is None:
                    combined_signals = normalized * weight
                else:
                    combined_signals = combined_signals + normalized * weight
                
                weight_sum += abs(weight)
        
        if combined_signals is None:
            # Fallback to single factor if no weighted combination worked
            return self._construct_single_factor_strategy(factors)
        
        # Generate signals from combined factors
        entry_threshold = self.config.strategy_construction.get("entry_threshold", 0.02)
        exit_threshold = self.config.strategy_construction.get("exit_threshold", -0.01)
        
        signals_df = combined_signals.with_columns([
            pl.when(pl.col("*").mean(axis=1) > entry_threshold)
             .then(1)
             .when(pl.col("*").mean(axis=1) < exit_threshold)
             .then(-1)
             .otherwise(0)
             .alias("signal")
        ])
        
        # Apply shift to prevent lookahead bias
        from src.quantlab.features.signals import shift_enforce
        signals_df = shift_enforce(signals_df, columns=["signal"], periods=1)
        
        strategy = {
            "type": "multi_factor",
            "factor_weights": factor_weights,
            "signals": signals_df.to_dict(),
            "position_size": self.config.strategy_construction.get("position_size", 1.0),
            "max_position": self.config.strategy_construction.get("max_position", 10)
        }
        
        return strategy
    
    def _construct_rotation_strategy(self, factors: Dict[str, Any]) -> Dict[str, Any]:
        """Construct a rotation strategy."""
        # This would involve rotating between different assets/classes based on factors
        strategy = {
            "type": "rotation",
            "rotation_frequency": self.config.strategy_construction.get("rotation_frequency", "monthly"),
            "selection_criteria": self.config.strategy_construction.get("selection_criteria", "momentum"),
            "universe_size": self.config.strategy_construction.get("universe_size", 10),
            "position_size": self.config.strategy_construction.get("position_size", 1.0)
        }
        
        return strategy
    
    def _construct_timing_strategy(self, factors: Dict[str, Any]) -> Dict[str, Any]:
        """Construct a market timing strategy."""
        # This would involve adjusting overall exposure based on market conditions
        strategy = {
            "type": "timing",
            "timing_signal": self.config.strategy_construction.get("timing_signal", "volatility_regime"),
            "exposure_levels": self.config.strategy_construction.get("exposure_levels", [0.2, 0.5, 1.0]),
            "position_size": self.config.strategy_construction.get("position_size", 1.0)
        }
        
        return strategy
    
    def stage4_backtest_validation(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 4: Backtest Validation
        
        Based on quant-learning-24h insights:
        - Backtesting system implementation
        - JoinQuant backtest and ranking analysis
        """
        logger.info("stage_start", stage="backtest_validation", strategy=self.config.strategy_name)
        
        # Get data for backtesting
        universe_provider = create_universe_provider({
            "instrument": self.config.instrument,
            "data": self.config.data_config
        })
        symbols = universe_provider.get_universe()
        
        start_date = datetime.fromisoformat(self.config.data_config.get("start_date", "2020-01-01"))
        end_date = datetime.fromisoformat(self.config.data_config.get("end_date", "2024-12-31"))
        freq = self.config.data_config["frequency"]
        
        data_df = self.data_pipeline.get_data(
            symbols=symbols,
            start=start_date,
            end=end_date,
            freq=freq
        )
        
        if data_df.is_empty():
            logger.error("no_data_for_backtest")
            return {"error": "No data available for backtesting"}
        
        # Prepare signals for backtesting
        if strategy["type"] == "single_factor":
            # Extract signals from the strategy
            signals_dict = strategy["signals"]
            signals_df = pl.DataFrame(signals_dict)
            
            # Flatten signals for vectorbt (assuming single symbol for now)
            signals = signals_df["signal"].to_numpy()
        else:
            # For other strategy types, generate signals based on strategy logic
            signals = self._generate_strategy_signals(data_df, strategy)
        
        # Run backtest with both engines
        backtest_results = {}
        
        # Fast validation with VectorBT
        if self.config.backtest_validation.get("use_vectorbt", True):
            engine_config = self.config.backtest_validation.get("vectorbt_config", {})
            spec = {
                "instrument": self.config.instrument,
                "data": self.config.data_config,
                "backtest": self.config.backtest_validation.get("engine_params", {
                    "initial_capital": 100000,
                    "commission": 0.0001,
                    "slippage": 0.0005
                }),
                "performance": {
                    "vectorbt_chunking": engine_config.get("chunk_size", 200)
                }
            }
            
            engine = VectorBTBacktestEngine(spec)
            
            if engine_config.get("use_chunking", False):
                chunk_size = engine_config.get("chunk_size", 200)
                results = engine.run_backtest_chunked(
                    data_df, signals, 
                    initial_capital=spec["backtest"]["initial_capital"],
                    chunk_size=chunk_size
                )
            else:
                results = engine.run_backtest(
                    data_df, signals, 
                    initial_capital=spec["backtest"]["initial_capital"]
                )
            
            backtest_results["vectorbt"] = results
        
        # Save results
        results_path = self.output_dir / f"{self.config.strategy_name}_backtest_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(backtest_results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info("stage_complete", stage="backtest_validation", metrics=backtest_results.get("vectorbt", {}).get("metrics", {}))
        return backtest_results
    
    def _generate_strategy_signals(self, data_df: pl.DataFrame, strategy: Dict[str, Any]) -> np.ndarray:
        """Generate signals based on strategy type."""
        if strategy["type"] == "single_factor":
            # Signals already in strategy
            signals_dict = strategy["signals"]
            signals_df = pl.DataFrame(signals_dict)
            return signals_df["signal"].to_numpy()
        elif strategy["type"] == "multi_factor":
            # Combine multiple factors
            # This is a simplified version - in reality would depend on factor normalization
            return np.zeros(len(data_df))
        else:
            # Default to zeros
            return np.zeros(len(data_df))
    
    def stage5_parameter_optimization(self, strategy: Dict[str, Any], backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 5: Parameter Optimization
        
        Based on quant-learning-24h insights:
        - Risk management optimization theory
        - ML optimization methods
        """
        logger.info("stage_start", stage="parameter_optimization", strategy=self.config.strategy_name)
        
        # Get data for optimization
        universe_provider = create_universe_provider({
            "instrument": self.config.instrument,
            "data": self.config.data_config
        })
        symbols = universe_provider.get_universe()
        
        start_date = datetime.fromisoformat(self.config.data_config.get("start_date", "2020-01-01"))
        end_date = datetime.fromisoformat(self.config.data_config.get("end_date", "2024-12-31"))
        freq = self.config.data_config["frequency"]
        
        data_df = self.data_pipeline.get_data(
            symbols=symbols,
            start=start_date,
            end=end_date,
            freq=freq
        )
        
        if data_df.is_empty():
            logger.error("no_data_for_optimization")
            return {"error": "No data available for optimization"}
        
        # Set up optimization
        optimization_config = self.config.parameter_optimization
        
        # Use the CoarseToFineOptimizer
        optimizer = CoarseToFineOptimizer(
            spec={
                "instrument": self.config.instrument,
                "data": self.config.data_config,
                "backtest": optimization_config.get("backtest_params", {
                    "initial_capital": 100000,
                    "commission": 0.0001,
                    "slippage": 0.0005
                }),
                "optimization": optimization_config
            },
            data=data_df
        )
        
        # Define search space based on strategy type
        if strategy["type"] == "single_factor":
            search_space = {
                "entry_threshold": (-0.1, 0.1),
                "exit_threshold": (-0.1, 0.1),
                "position_size": (0.1, 1.0)
            }
        elif strategy["type"] == "multi_factor":
            search_space = {
                "factor_weight_1": (-1.0, 1.0),
                "factor_weight_2": (-1.0, 1.0),
                "entry_threshold": (-0.1, 0.1),
                "position_size": (0.1, 1.0)
            }
        else:
            search_space = {
                "entry_threshold": (-0.1, 0.1),
                "exit_threshold": (-0.1, 0.1),
                "position_size": (0.1, 1.0)
            }
        
        # Run optimization
        optimization_results = optimizer.optimize(
            search_space=search_space,
            n_coarse=50,
            n_fine=200,
            timeout=optimization_config.get("timeout", 3600)
        )
        
        # Save results
        opt_path = self.output_dir / f"{self.config.strategy_name}_optimization_results.json"
        with open(opt_path, 'w', encoding='utf-8') as f:
            json.dump(optimization_results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info("stage_complete", stage="parameter_optimization", best_score=optimization_results.get("best_value"))
        return optimization_results
    
    def stage6_robustness_validation(self, strategy: Dict[str, Any], optimization_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 6: Robustness Validation
        
        Based on quant-learning-24h insights:
        - Risk management (VaR, stress testing)
        - ML validation (cross-validation, out-of-sample testing)
        """
        logger.info("stage_start", stage="robustness_validation", strategy=self.config.strategy_name)
        
        # Get data for robustness testing
        universe_provider = create_universe_provider({
            "instrument": self.config.instrument,
            "data": self.config.data_config
        })
        symbols = universe_provider.get_universe()
        
        start_date = datetime.fromisoformat(self.config.data_config.get("start_date", "2020-01-01"))
        end_date = datetime.fromisoformat(self.config.data_config.get("end_date", "2024-12-31"))
        freq = self.config.data_config["frequency"]
        
        data_df = self.data_pipeline.get_data(
            symbols=symbols,
            start=start_date,
            end=end_date,
            freq=freq
        )
        
        if data_df.is_empty():
            logger.error("no_data_for_robustness")
            return {"error": "No data available for robustness testing"}
        
        robustness_results = {}
        
        # 1. Walk-forward analysis
        if self.config.robustness_validation.get("walk_forward", {}).get("enabled", True):
            wf_config = self.config.robustness_validation["walk_forward"]
            wf_analyzer = WalkForwardAnalysis({
                "instrument": self.config.instrument,
                "data": self.config.data_config,
                "robustness": wf_config
            })
            
            # Simplified function for walk-forward (would normally be the strategy function)
            def dummy_strategy_func(df, params):
                # Placeholder - in reality would implement the actual strategy
                signals = np.zeros(len(df))
                return signals, {}
            
            wf_results = wf_analyzer.run_analysis(
                df=data_df,
                strategy_func=dummy_strategy_func,
                initial_capital=wf_config.get("initial_capital", 100000)
            )
            robustness_results["walk_forward"] = wf_results
        
        # 2. Bootstrap analysis
        if self.config.robustness_validation.get("bootstrap", {}).get("enabled", True):
            boot_config = self.config.robustness_validation["bootstrap"]
            boot_analyzer = BootstrapAnalysis({
                "instrument": self.config.instrument,
                "data": self.config.data_config,
                "robustness": boot_config
            })
            
            def dummy_strategy_func(df, params):
                signals = np.zeros(len(df))
                return signals, {}
            
            boot_results = boot_analyzer.run_analysis(
                df=data_df,
                strategy_func=dummy_strategy_func,
                metric_name=boot_config.get("metric", "sharpe_ratio")
            )
            robustness_results["bootstrap"] = boot_results
        
        # 3. Sensitivity analysis
        if self.config.robustness_validation.get("sensitivity", {}).get("enabled", True):
            sens_config = self.config.robustness_validation["sensitivity"]
            sens_analyzer = SensitivityAnalysis({
                "instrument": self.config.instrument,
                "data": self.config.data_config,
                "robustness": sens_config
            })
            
            def dummy_strategy_func(df, params):
                signals = np.zeros(len(df))
                return signals, {}
            
            sens_results = sens_analyzer.run_analysis(
                df=data_df,
                strategy_func=dummy_strategy_func,
                metric_name=sens_config.get("metric", "sharpe_ratio")
            )
            robustness_results["sensitivity"] = sens_results
        
        # 4. Regime analysis
        if self.config.robustness_validation.get("regime", {}).get("enabled", True):
            regime_config = self.config.robustness_validation["regime"]
            regime_analyzer = RegimeAnalysis({
                "instrument": self.config.instrument,
                "data": self.config.data_config,
                "robustness": regime_config
            })
            
            def dummy_strategy_func(df, params):
                signals = np.zeros(len(df))
                return signals, {}
            
            regime_results = regime_analyzer.run_analysis(
                df=data_df,
                strategy_func=dummy_strategy_func
            )
            robustness_results["regime"] = regime_results
        
        # 5. Leakage detection
        if self.config.robustness_validation.get("leakage", {}).get("enabled", True):
            leak_config = self.config.robustness_validation["leakage"]
            leak_detector = LeakageDetection({
                "instrument": self.config.instrument,
                "data": self.config.data_config,
                "robustness": leak_config
            })
            
            # Create a features dataframe for leakage detection
            features_df = data_df.select(["close"]).rename({"close": "feature"})
            
            leak_results = leak_detector.run_all_checks(
                df=data_df,
                features_df=features_df
            )
            robustness_results["leakage"] = leak_results
        
        # Save results
        robust_path = self.output_dir / f"{self.config.strategy_name}_robustness_results.json"
        with open(robust_path, 'w', encoding='utf-8') as f:
            json.dump(robustness_results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info("stage_complete", stage="robustness_validation", tests_completed=len(robustness_results))
        return robustness_results
    
    def run_complete_research_cycle(self) -> Dict[str, Any]:
        """
        Run the complete strategy research lifecycle.
        
        Returns:
            Dictionary containing results from all stages
        """
        logger.info("research_cycle_start", strategy=self.config.strategy_name)
        
        # Stage 1: Idea Generation
        ideas = self.stage1_idea_generation()
        
        # Stage 2: Factor Design
        factors = self.stage2_factor_design(ideas)
        
        # Stage 3: Strategy Construction
        strategy = self.stage3_strategy_construction(factors)
        
        # Stage 4: Backtest Validation
        backtest_results = self.stage4_backtest_validation(strategy)
        
        # Stage 5: Parameter Optimization
        optimization_results = self.stage5_parameter_optimization(strategy, backtest_results)
        
        # Stage 6: Robustness Validation
        robustness_results = self.stage6_robustness_validation(strategy, optimization_results)
        
        # Compile final results
        final_results = {
            "strategy_name": self.config.strategy_name,
            "stages_completed": 6,
            "idea_generation": {
                "ideas_count": len(ideas),
                "ideas_sample": ideas[:3]  # First 3 ideas
            },
            "factor_design": {
                "factors_created": sum(len(v) for v in factors.values()),
                "factor_categories": list(factors.keys())
            },
            "strategy_construction": {
                "strategy_type": strategy["type"],
                "parameters": {k: v for k, v in strategy.items() if k != "signals"}  # Exclude large signal data
            },
            "backtest_validation": backtest_results,
            "parameter_optimization": optimization_results,
            "robustness_validation": {
                "tests_completed": len(robustness_results),
                "test_names": list(robustness_results.keys())
            },
            "completed_at": datetime.utcnow().isoformat()
        }
        
        # Save final results
        final_path = self.output_dir / f"{self.config.strategy_name}_final_results.json"
        with open(final_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info("research_cycle_complete", strategy=self.config.strategy_name, results=final_results)
        return final_results


def run_example_research_cycle():
    """
    Example usage of the strategy research lifecycle.
    """
    # Example configuration for a momentum strategy
    config = StrategyResearchConfig(
        strategy_name="example_momentum_strategy",
        instrument={
            "asset_type": "US_STOCK",
            "symbol": "SPY",
            "venue": "NYSE",
            "quote_currency": "USD",
            "lot_size": 1,
            "allow_fractional": True,
            "shortable": True,
            "leverage": 1
        },
        data_config={
            "frequency": "1D",
            "start_date": "2020-01-01",
            "end_date": "2024-12-31",
            "source": "yfinance"
        },
        idea_generation={
            "technical_analysis": True,
            "fundamental_analysis": False,
            "behavioral_finance": True
        },
        factor_design={
            "momentum_factors": {
                "enabled": True,
                "periods": [5, 10, 20]
            },
            "volatility_factors": {
                "enabled": True,
                "periods": [10, 20]
            },
            "technical_factors": {
                "enabled": True,
                "rsi_period": 14,
                "bollinger_bands": {
                    "enabled": True,
                    "period": 20,
                    "std_dev": 2.0
                }
            }
        },
        strategy_construction={
            "type": "single_factor",
            "primary_factor": "momentum_20d",
            "entry_threshold": 0.02,
            "exit_threshold": -0.01,
            "position_size": 1.0,
            "max_position": 10
        },
        backtest_validation={
            "use_vectorbt": True,
            "vectorbt_config": {
                "use_chunking": True,
                "chunk_size": 200
            },
            "engine_params": {
                "initial_capital": 100000,
                "commission": 0.0001,
                "slippage": 0.0005
            }
        },
        parameter_optimization={
            "coarse_search": {
                "sample_size": 50,
                "window_fraction": 0.3
            },
            "fine_tuning": {
                "n_trials": 200,
                "timeout": 3600
            },
            "optimization_constraints": {
                "max_turnover": 500,
                "min_trades_per_month": 5,
                "max_drawdown_limit": 0.25
            }
        },
        robustness_validation={
            "walk_forward": {
                "enabled": True,
                "train_window": 252,
                "test_window": 63,
                "step": 63
            },
            "bootstrap": {
                "enabled": True,
                "n_samples": 1000,
                "confidence_level": 0.95
            },
            "sensitivity": {
                "enabled": True,
                "parameters": ["entry_threshold", "exit_threshold"]
            },
            "regime": {
                "enabled": True,
                "detection_method": "volatility_trend"
            },
            "leakage": {
                "enabled": True
            }
        }
    )
    
    # Run the complete research cycle
    lifecycle = StrategyResearchLifecycle(config)
    results = lifecycle.run_complete_research_cycle()
    
    return results


if __name__ == "__main__":
    # Run example if executed directly
    results = run_example_research_cycle()
    print(f"Research cycle completed! Results saved to results/ directory")
    print(f"Strategy: {results['strategy_name']}")
    print(f"Stages completed: {results['stages_completed']}")
    print(f"Momentum factors created: {results['factor_design']['factors_created']}")
