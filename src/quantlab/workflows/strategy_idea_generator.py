"""
Strategy Idea Generation Templates
================================

Based on quant-learning-24h insights, this module provides
systematic strategy idea generation templates.
"""

import numpy as np
import polars as pl
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum


class StrategyCategory(Enum):
    """Strategy categories based on quant-learning-24h."""
    TREND_FOLLOWING = "trend_following"      # 趋势跟踪
    MEAN_REVERSION = "mean_reversion"          # 均值回归
    MOMENTUM = "momentum"                      # 动量
    REVERSAL = "reversal"                      # 反转
    VALUE = "value"                            # 价值
    GROWTH = "growth"                          # 成长
    QUALITY = "quality"                        # 质量
    SMART_MONEY = "smart_money"                # 聪明钱
    SENTIMENT = "sentiment"                     # 情绪
    ROTATION = "rotation"                       # 轮动
    TIMING = "timing"                         # 择时


@dataclass
class StrategyIdea:
    """Strategy idea template."""
    name: str
    category: StrategyCategory
    description: str
    logic: str
    parameters: Dict[str, any]
    suitable_markets: List[str]
    risk_level: str  # low, medium, high
    time_horizon: str  # intraday, short_term, medium_term, long_term
    performance_expectation: str
    source: str  # 哪个 quant-learning-24h 阶段


class StrategyIdeaGenerator:
    """
    Generate strategy ideas based on quant-learning-24h knowledge.
    """
    
    def __init__(self):
        self.idea_library = self._build_idea_library()
    
    def generate_ideas(
        self,
        market_type: str,
        market_regime: str,
        risk_preference: str,
        count: int = 10
    ) -> List[StrategyIdea]:
        """
        Generate strategy ideas based on market conditions.
        
        Args:
            market_type: CN_STOCK, US_STOCK, CRYPTO_SPOT, CRYPTO_PERP
            market_regime: trending, ranging, volatile
            risk_preference: low, medium, high
            count: Number of ideas to generate
        
        Returns:
            List of strategy ideas
        """
        # Filter ideas by market type
        filtered_ideas = [
            idea for idea in self.idea_library
            if market_type in idea.suitable_markets
        ]
        
        # Sort by relevance to market regime
        scored_ideas = []
        for idea in filtered_ideas:
            score = self._calculate_relevance_score(
                idea, market_regime, risk_preference
            )
            scored_ideas.append((idea, score))
        
        # Sort by score and return top N
        scored_ideas.sort(key=lambda x: x[1], reverse=True)
        
        return [idea for idea, score in scored_ideas[:count]]
    
    def _calculate_relevance_score(
        self,
        idea: StrategyIdea,
        market_regime: str,
        risk_preference: str
    ) -> float:
        """Calculate relevance score for an idea."""
        score = 0.0
        
        # Market regime matching
        if market_regime == "trending":
            if idea.category in [StrategyCategory.TREND_FOLLOWING, StrategyCategory.MOMENTUM]:
                score += 3.0
        elif market_regime == "ranging":
            if idea.category in [StrategyCategory.MEAN_REVERSION, StrategyCategory.REVERSAL]:
                score += 3.0
        elif market_regime == "volatile":
            if idea.category in [StrategyCategory.ROTATION, StrategyCategory.TIMING]:
                score += 2.0
        
        # Risk preference matching
        if risk_preference == "low":
            if idea.risk_level == "low":
                score += 2.0
            elif idea.risk_level == "medium":
                score += 1.0
        elif risk_preference == "medium":
            if idea.risk_level == "medium":
                score += 2.0
        elif risk_preference == "high":
            if idea.risk_level == "high":
                score += 2.0
        
        # Time horizon bonus (prefer medium_term for balance)
        if idea.time_horizon == "medium_term":
            score += 0.5
        
        return score
    
    def _build_idea_library(self) -> List[StrategyIdea]:
        """Build strategy idea library from quant-learning-24h."""
        ideas = []
        
        # === Technical Analysis Ideas ===
        
        # 1. Dual Moving Average Crossover
        ideas.append(StrategyIdea(
            name="双均线交叉系统",
            category=StrategyCategory.TREND_FOLLOWING,
            description="快线上穿慢线买入，下穿卖出。经典的趋势跟踪策略。",
            logic="MA(fast) > MA(slow) → LONG; MA(fast) < MA(slow) → SHORT",
            parameters={
                "fast_period": {"min": 5, "max": 20, "default": 5},
                "slow_period": {"min": 20, "max": 100, "default": 20},
                "position_size": {"min": 1, "max": 10, "default": 1}
            },
            suitable_markets=["CN_STOCK", "US_STOCK", "CRYPTO_SPOT", "CRYPTO_PERP"],
            risk_level="low",
            time_horizon="short_term",
            performance_expectation="稳定收益，适合趋势行情",
            source="quant-learning-24h/stage01_market_basics"
        ))
        
        # 2. MACD Zero Crossover
        ideas.append(StrategyIdea(
            name="MACD 零轴突破",
            category=StrategyCategory.TREND_FOLLOWING,
            description="MACD DIF 线上穿 DEA 线且位于零轴上方买入，下穿零轴卖出。",
            logic="DIF > DEA and DIF > 0 → LONG; DIF < DEA and DIF < 0 → SHORT",
            parameters={
                "fast_period": {"min": 8, "max": 16, "default": 12},
                "slow_period": {"min": 20, "max": 32, "default": 26},
                "signal_period": {"min": 6, "max": 12, "default": 9},
                "position_size": {"min": 1, "max": 10, "default": 1}
            },
            suitable_markets=["CN_STOCK", "US_STOCK", "CRYPTO_SPOT", "CRYPTO_PERP"],
            risk_level="low",
            time_horizon="short_term",
            performance_expectation="中期趋势跟踪，信号相对较少",
            source="quant-learning-24h/stage01_market_basics"
        ))
        
        # 3. Bollinger Band Breakout
        ideas.append(StrategyIdea(
            name="布林带突破",
            category=StrategyCategory.TREND_FOLLOWING,
            description="价格突破上轨做多，跌破下轨做空，回归中轨平仓。",
            logic="Price > UpperBand → LONG; Price < LowerBand → SHORT; |Price - Middle| < threshold → CLOSE",
            parameters={
                "period": {"min": 10, "max": 30, "default": 20},
                "std_dev": {"min": 1.5, "max": 3.0, "default": 2.0},
                "exit_threshold": {"min": 0.5, "max": 1.5, "default": 1.0},
                "position_size": {"min": 1, "max": 10, "default": 1}
            },
            suitable_markets=["CN_STOCK", "US_STOCK", "CRYPTO_SPOT", "CRYPTO_PERP"],
            risk_level="medium",
            time_horizon="short_term",
            performance_expectation="波动率扩张期效果好，震荡市容易止损",
            source="quant-learning-24h/stage01_market_basics"
        ))
        
        # 4. RSI Mean Reversion
        ideas.append(StrategyIdea(
            name="RSI 均值回归",
            category=StrategyCategory.MEAN_REVERSION,
            description="RSI 超卖时买入，超买时卖出。适合震荡市。",
            logic="RSI < oversold → LONG; RSI > overbought → SHORT",
            parameters={
                "period": {"min": 7, "max": 21, "default": 14},
                "oversold": {"min": 20, "max": 35, "default": 30},
                "overbought": {"min": 65, "max": 80, "default": 70},
                "position_size": {"min": 1, "max": 10, "default": 1}
            },
            suitable_markets=["CN_STOCK", "US_STOCK", "CRYPTO_SPOT"],
            risk_level="low",
            time_horizon="intraday",
            performance_expectation="震荡市收益稳定，趋势市容易亏损",
            source="quant-learning-24h/stage01_market_basics"
        ))
        
        # 5. KDJ Golden Cross
        ideas.append(StrategyIdea(
            name="KDJ 金叉死叉",
            category=StrategyCategory.REVERSAL,
            description="K 线上穿 D 线金叉买入，下穿死叉卖出。短期反转信号。",
            logic="K > D → LONG; K < D → SHORT",
            parameters={
                "n": {"min": 6, "max": 12, "default": 9},
                "m1": {"min": 2, "max": 4, "default": 3},
                "m2": {"min": 2, "max": 4, "default": 3},
                "position_size": {"min": 1, "max": 10, "default": 1}
            },
            suitable_markets=["CN_STOCK", "US_STOCK"],
            risk_level="medium",
            time_horizon="intraday",
            performance_expectation="短期反转效果明显，需要及时止损",
            source="quant-learning-24h/stage21_joinquant_platform"
        ))
        
        # 6. Williams %R
        ideas.append(StrategyIdea(
            name="威廉指标反转",
            category=StrategyCategory.REVERSAL,
            description="WR < -80 超卖买入，WR > -20 超买卖出。",
            logic="WR < -80 → LONG; WR > -20 → SHORT",
            parameters={
                "period": {"min": 7, "max": 21, "default": 14},
                "oversold": {"min": -90, "max": -75, "default": -80},
                "overbought": {"min": -25, "max": -10, "default": -20},
                "position_size": {"min": 1, "max": 10, "default": 1}
            },
            suitable_markets=["CN_STOCK", "US_STOCK", "CRYPTO_SPOT"],
            risk_level="medium",
            time_horizon="short_term",
            performance_expectation="快速反转，波动率低时效果差",
            source="quant-learning-24h/stage01_market_basics"
        ))
        
        # === Fundamental Analysis Ideas ===
        
        # 7. Low PE Strategy
        ideas.append(StrategyIdea(
            name="低PE价值策略",
            category=StrategyCategory.VALUE,
            description="买入 PE 低于历史平均且低于行业平均的股票。",
            logic="PE < Historical_Mean and PE < Industry_Mean → LONG",
            parameters={
                "pe_threshold_percentile": {"min": 10, "max": 30, "default": 20},
                "holding_period": {"min": 30, "max": 180, "default": 90},
                "position_size": {"min": 1, "max": 10, "default": 1}
            },
            suitable_markets=["CN_STOCK", "US_STOCK"],
            risk_level="low",
            time_horizon="medium_term",
            performance_expectation="长期收益稳定，适合价值投资",
            source="quant-learning-24h/stage03_asset_pricing"
        ))
        
        # 8. High Dividend Yield
        ideas.append(StrategyIdea(
            name="高股息策略",
            category=StrategyCategory.VALUE,
            description="买入股息率高且分红稳定的公司。",
            logic="Dividend_Yield > Threshold and Dividend_Growth_Stable → LONG",
            parameters={
                "yield_threshold": {"min": 3, "max": 8, "default": 5},
                "min_dividend_years": {"min": 3, "max": 10, "default": 5},
                "position_size": {"min": 1, "max": 10, "default": 1}
            },
            suitable_markets=["CN_STOCK", "US_STOCK"],
            risk_level="low",
            time_horizon="long_term",
            performance_expectation="稳健收益，适合长期持有",
            source="quant-learning-24h/stage03_asset_pricing"
        ))
        
        # 9. PB Recovery
        ideas.append(StrategyIdea(
            name="PB修复策略",
            category=StrategyCategory.VALUE,
            description="买入 PB < 1 且 ROE > 15% 的公司。",
            logic="PB < 1 and ROE > 15% → LONG",
            parameters={
                "pb_threshold": {"min": 0.8, "max": 1.2, "default": 1.0},
                "roe_threshold": {"min": 12, "max": 20, "default": 15},
                "position_size": {"min": 1, "max": 10, "default": 1}
            },
            suitable_markets=["CN_STOCK"],
            risk_level="low",
            time_horizon="medium_term",
            performance_expectation="底部反转，周期股效果好",
            source="quant-learning-24h/stage03_asset_pricing"
        ))
        
        # 10. High Growth
        ideas.append(StrategyIdea(
            name="高成长策略",
            category=StrategyCategory.GROWTH,
            description="买入营收和利润增速最快的公司。",
            logic="Revenue_Growth > 30% and Profit_Growth > 20% → LONG",
            parameters={
                "revenue_growth_threshold": {"min": 20, "max": 50, "default": 30},
                "profit_growth_threshold": {"min": 15, "max": 30, "default": 20},
                "position_size": {"min": 1, "max": 10, "default": 1}
            },
            suitable_markets=["CN_STOCK", "US_STOCK"],
            risk_level="high",
            time_horizon="medium_term",
            performance_expectation="高收益高波动，需严格控制风险",
            source="quant-learning-24h/stage03_asset_pricing"
        ))
        
        # 11. PEG Strategy
        ideas.append(StrategyIdea(
            name="PEG策略",
            category=StrategyCategory.GROWTH,
            description="买入 PEG < 1 的高增长公司。",
            logic="PEG = PE / Growth_Rate < 1 → LONG",
            parameters={
                "peg_threshold": {"min": 0.8, "max": 1.2, "default": 1.0},
                "min_growth_rate": {"min": 15, "max": 30, "default": 20},
                "position_size": {"min": 1, "max": 10, "default": 1}
            },
            suitable_markets=["CN_STOCK", "US_STOCK"],
            risk_level="medium",
            time_horizon="medium_term",
            performance_expectation="兼顾价值和成长，平衡性较好",
            source="quant-learning-24h/stage03_asset_pricing"
        ))
        
        # 12. High ROE
        ideas.append(StrategyIdea(
            name="高ROE策略",
            category=StrategyCategory.QUALITY,
            description="买入 ROE 持续 > 20% 的优质公司。",
            logic="ROE > 20% for 3 years → LONG",
            parameters={
                "roe_threshold": {"min": 15, "max": 25, "default": 20},
                "min_years": {"min": 2, "max": 5, "default": 3},
                "position_size": {"min": 1, "max": 10, "default": 1}
            },
            suitable_markets=["CN_STOCK", "US_STOCK"],
            risk_level="low",
            time_horizon="long_term",
            performance_expectation="白马股，长期稳定收益",
            source="quant-learning-24h/stage03_asset_pricing"
        ))
        
        # === Behavioral Finance Ideas ===
        
        # 13. Price Momentum
        ideas.append(StrategyIdea(
            name="价格动量策略",
            category=StrategyCategory.MOMENTUM,
            description="过去 20 天涨幅最高的股票继续上涨。",
            logic="Return_20d > Threshold → LONG",
            parameters={
                "lookback_period": {"min": 5, "max": 60, "default": 20},
                "momentum_threshold": {"min": 1, "max": 10, "default": 5},
                "top_n": {"min": 5, "max": 20, "default": 10},
                "position_size": {"min": 1, "max": 10, "default": 1}
            },
            suitable_markets=["CN_STOCK", "US_STOCK", "CRYPTO_SPOT", "CRYPTO_PERP"],
            risk_level="medium",
            time_horizon="short_term",
            performance_expectation="趋势延续，但在转折期容易亏损",
            source="quant-learning-24h/stage01_market_basics"
        ))
        
        # 14. Sector Rotation
        ideas.append(StrategyIdea(
            name="行业轮动策略",
            category=StrategyCategory.ROTATION,
            description="买入动量最强的行业 ETF。",
            logic="Sector_Momentum_Rank → Top 3 Sectors → LONG",
            parameters={
                "lookback_period": {"min": 20, "max": 120, "default": 60},
                "top_n_sectors": {"min": 2, "max": 5, "default": 3},
                "rebalance_frequency": {"min": 5, "max": 30, "default": 20},
                "position_size": {"min": 1, "max": 10, "default": 1}
            },
            suitable_markets=["CN_STOCK", "US_STOCK"],
            risk_level="medium",
            time_horizon="short_term",
            performance_expectation="捕捉风格轮动，降低个股风险",
            source="quant-learning-24h/stage01_market_basics"
        ))
        
        # 15. 52-Week High
        ideas.append(StrategyIdea(
            name="52周新高策略",
            category=StrategyCategory.MOMENTUM,
            description="创 52 周新高的股票继续上涨。",
            logic="Price > 52_Week_High * 0.98 → LONG",
            parameters={
                "lookback_days": {"min": 200, "max": 365, "default": 252},
                "proximity_threshold": {"min": 0.95, "max": 0.99, "default": 0.98},
                "position_size": {"min": 1, "max": 10, "default": 1}
            },
            suitable_markets=["CN_STOCK", "US_STOCK"],
            risk_level="high",
            time_horizon="short_term",
            performance_expectation="突破行情，但需警惕假突破",
            source="quant-learning-24h/stage01_market_basics"
        ))
        
        # 16. Overreaction Reversal
        ideas.append(StrategyIdea(
            name="过度反应反转",
            category=StrategyCategory.REVERSAL,
            description="单日跌幅 > -7% 的股票次日大概率反弹。",
            logic="Return_1d < -7% → LONG",
            parameters={
                "decline_threshold": {"min": -10, "max": -5, "default": -7},
                "holding_period": {"min": 1, "max": 5, "default": 3},
                "position_size": {"min": 1, "max": 10, "default": 1}
            },
            suitable_markets=["CN_STOCK", "US_STOCK"],
            risk_level="medium",
            time_horizon="intraday",
            performance_expectation="恐慌性下跌后的反弹，需快速止损",
            source="quant-learning-24h/stage06_ml_applications"
        ))
        
        # 17. Volatility Reversal
        ideas.append(StrategyIdea(
            name="波动率反转",
            category=StrategyCategory.REVERSAL,
            description="高波动后趋于平静，低波动后爆发。",
            logic="ATR > 2*ATR_MA → Mean Reversion; ATR < 0.5*ATR_MA → Volatility Expansion",
            parameters={
                "atr_period": {"min": 10, "max": 30, "default": 14},
                "atr_ma_period": {"min": 20, "max": 60, "default": 30},
                "high_threshold": {"min": 1.5, "max": 3.0, "default": 2.0},
                "low_threshold": {"min": 0.3, "max": 0.7, "default": 0.5},
                "position_size": {"min": 1, "max": 10, "default": 1}
            },
            suitable_markets=["CN_STOCK", "US_STOCK", "CRYPTO_SPOT", "CRYPTO_PERP"],
            risk_level="medium",
            time_horizon="short_term",
            performance_expectation="波动率均值回归，效果稳定",
            source="quant-learning-24h/stage11_market_microstructure"
        ))
        
        # === Smart Money Ideas ===
        
        # 18. Northbound Capital Flow
        ideas.append(StrategyIdea(
            name="北向资金流入",
            category=StrategyCategory.SMART_MONEY,
            description="跟随外资流向。",
            logic="Net_Flow > Threshold for N days → LONG",
            parameters={
                "flow_threshold": {"min": 0.5, "max": 2.0, "default": 1.0},
                "consecutive_days": {"min": 3, "max": 10, "default": 5},
                "position_size": {"min": 1, "max": 10, "default": 1}
            },
            suitable_markets=["CN_STOCK"],
            risk_level="low",
            time_horizon="short_term",
            performance_expectation="外资偏好，短期效果好",
            source="quant-learning-24h/stage21_joinquant_platform"
        ))
        
        # 19. Institutional Holdings Change
        ideas.append(StrategyIdea(
            name="机构持仓变化",
            category=StrategyCategory.SMART_MONEY,
            description="机构增持的股票后续表现更好。",
            logic="Holdings_Change > Threshold → LONG",
            parameters={
                "change_threshold": {"min": 0.5, "max": 3.0, "default": 1.0},
                "min_holding_ratio": {"min": 2, "max": 10, "default": 5},
                "position_size": {"min": 1, "max": 10, "default": 1}
            },
            suitable_markets=["CN_STOCK", "US_STOCK"],
            risk_level="low",
            time_horizon="medium_term",
            performance_expectation="机构信息优势，中期效果好",
            source="quant-learning-24h/stage06_ml_applications"
        ))
        
        # 20. Major Shareholder Increase
        ideas.append(StrategyIdea(
            name="大股东增持",
            category=StrategyCategory.SMART_MONEY,
            description="大股东或高管增持彰显信心。",
            logic="Holdings_Increase > Threshold → LONG",
            parameters={
                "increase_threshold": {"min": 3, "max": 10, "default": 5},
                "insider_only": {"min": 0, "max": 1, "default": 0},
                "position_size": {"min": 1, "max": 10, "default": 1}
            },
            suitable_markets=["CN_STOCK", "US_STOCK"],
            risk_level="medium",
            time_horizon="medium_term",
            performance_expectation="信心信号，但需结合基本面",
            source="quant-learning-24h/stage21_joinquant_platform"
        ))
        
        # === Crypto-Specific Ideas ===
        
        # 21. Weekend Effect
        ideas.append(StrategyIdea(
            name="周末效应",
            category=StrategyCategory.REVERSAL,
            description="加密货币在周末和 weekdays 表现不同。",
            logic="Weekend → Different_Strategy; Weekday → Different_Strategy",
            parameters={
                "weekend_strategy": {"min": 0, "max": 3, "default": 1},
                "weekday_strategy": {"min": 0, "max": 3, "default": 2},
                "position_size": {"min": 1, "max": 10, "default": 1}
            },
            suitable_markets=["CRYPTO_SPOT", "CRYPTO_PERP"],
            risk_level="medium",
            time_horizon="intraday",
            performance_expectation="24/7 市场的独特模式",
            source="quant-learning-24h/stage13_quant_trading_system"
        ))
        
        # 22. Funding Rate Arbitrage
        ideas.append(StrategyIdea(
            name="资金费率套利",
            category=StrategyCategory.ROTATION,
            description="永续合约资金费率偏离时的套利机会。",
            logic="|Funding_Rate - Implied_Rate| > Threshold → Arbitrage",
            parameters={
                "rate_threshold": {"min": 0.005, "max": 0.02, "default": 0.01},
                "position_size": {"min": 1, "max": 10, "default": 1}
            },
            suitable_markets=["CRYPTO_PERP"],
            risk_level="low",
            time_horizon="intraday",
            performance_expectation="低风险稳定收益，但机会少",
            source="quant-learning-24h/stage13_quant_trading_system"
        ))
        
        # 23. Liquidity Mining
        ideas.append(StrategyIdea(
            name="流动性挖矿",
            category=StrategyCategory.ROTATION,
            description="在 DeFi 协议中提供流动性赚取奖励。",
            logic="Liquidity_Provision → Earn_Yield + Trading_Fees",
            parameters={
                "min_apy": {"min": 5, "max": 50, "default": 15},
                "impermanent_loss_threshold": {"min": 0.5, "max": 2.0, "default": 1.0},
                "position_size": {"min": 1, "max": 10, "default": 1}
            },
            suitable_markets=["CRYPTO_SPOT"],
            risk_level="high",
            time_horizon="medium_term",
            performance_expectation="高收益但无常损失风险",
            source="quant-learning-24h/stage13_quant_trading_system"
        ))
        
        # === Market Microstructure Ideas ===
        
        # 24. Opening/Closing Effect
        ideas.append(StrategyIdea(
            name="开盘收盘效应",
            category=StrategyCategory.REVERSAL,
            description="利用开盘和收盘的特殊模式。",
            logic="9:25-9:35 or 14:50-15:00 → Specific_Patterns",
            parameters={
                "entry_time_start": {"min": "09:25", "max": "09:35", "default": "09:25"},
                "entry_time_end": {"min": "09:35", "max": "09:45", "default": "09:35"},
                "position_size": {"min": 1, "max": 10, "default": 1}
            },
            suitable_markets=["CN_STOCK"],
            risk_level="high",
            time_horizon="intraday",
            performance_expectation="短期模式，需精确执行",
            source="quant-learning-24h/stage11_market_microstructure"
        ))
        
        # 25. Volume-Price Relationship
        ideas.append(StrategyIdea(
            name="量价关系",
            category=StrategyCategory.MOMENTUM,
            description="放量上涨，缩量下跌。",
            logic="Price_Up and Volume_Increase → LONG; Price_Down and Volume_Decrease → SHORT",
            parameters={
                "volume_ma_period": {"min": 5, "max": 20, "default": 10},
                "volume_increase_threshold": {"min": 1.2, "max": 2.0, "default": 1.5},
                "position_size": {"min": 1, "max": 10, "default": 1}
            },
            suitable_markets=["CN_STOCK", "US_STOCK", "CRYPTO_SPOT"],
            risk_level="medium",
            time_horizon="intraday",
            performance_expectation="量价配合，效果较好",
            source="quant-learning-24h/stage01_market_basics"
        ))
        
        return ideas


# Usage example
if __name__ == "__main__":
    generator = StrategyIdeaGenerator()
    
    # Generate ideas for specific market conditions
    ideas = generator.generate_ideas(
        market_type="CN_STOCK",
        market_regime="ranging",
        risk_preference="low",
        count=5
    )
    
    print(f"生成了 {len(ideas)} 个策略创意：\n")
    for i, idea in enumerate(ideas, 1):
        print(f"{i}. {idea.name}")
        print(f"   类别: {idea.category.value}")
        print(f"   描述: {idea.description}")
        print(f"   风险: {idea.risk_level}")
        print(f"   时间框架: {idea.time_horizon}")
        print(f"   来源: {idea.source}")
        print()
