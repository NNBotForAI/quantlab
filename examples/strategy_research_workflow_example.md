# 策略研究工作流使用示例

基于 `quant-learning-24h` 学习成果和 QuantLab 平台的完整策略研发流程演示。

## 示例1：美股动量策略研究

```python
from quantlab.workflows.strategy_research_lifecycle import StrategyResearchLifecycle, StrategyResearchConfig
from datetime import datetime

# 配置美股动量策略研究
config = StrategyResearchConfig(
    strategy_name="us_stock_momentum",
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
            "periods": [5, 10, 20, 60]
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
    },
    output_dir="results/us_stock_momentum",
    cache_enabled=True
)

# 运行完整研究流程
lifecycle = StrategyResearchLifecycle(config)
results = lifecycle.run_complete_research_cycle()

print(f"美股动量策略研究完成！")
print(f"策略名称: {results['strategy_name']}")
print(f"阶段完成数: {results['stages_completed']}")
print(f"生成因子数: {results['factor_design']['factors_created']}")
```

## 示例2：A股轮动策略研究

```python
# A股轮动策略配置
cn_config = StrategyResearchConfig(
    strategy_name="cn_stock_rotation",
    instrument={
        "asset_type": "CN_STOCK",
        "symbol": "000300.SH",  # 沪深300
        "venue": "SSE",
        "quote_currency": "CNY",
        "lot_size": 100,  # A股特有
        "allow_fractional": False,  # A股特有
        "shortable": False,  # A股T+1限制
        "leverage": 1
    },
    data_config={
        "frequency": "1D",
        "start_date": "2020-01-01",
        "end_date": "2024-12-31",
        "source": "akshare"
    },
    idea_generation={
        "technical_analysis": True,
        "fundamental_analysis": True,
        "behavioral_finance": True,
        "market_specific": ["CN_STOCK"]  # A股特殊因素
    },
    factor_design={
        "momentum_factors": {
            "enabled": True,
            "periods": [5, 10, 20]
        },
        "value_factors": {
            "enabled": True,  # A股价值投资有效
            "metrics": ["PE", "PB", "dividend_yield"]
        },
        "sentiment_factors": {
            "enabled": True,  # A股情绪影响较大
            "metrics": ["northbound_flow", "margin_balance"]  # 北向资金、融资余额
        }
    },
    strategy_construction={
        "type": "rotation",  # 轮动策略
        "rotation_frequency": "weekly",
        "selection_criteria": "momentum",
        "universe_size": 10,
        "position_size": 1.0
    },
    # ... 其他配置类似
)

cn_lifecycle = StrategyResearchLifecycle(cn_config)
cn_results = cn_lifecycle.run_complete_research_cycle()
```

## 示例3：加密货币趋势策略研究

```python
# 加密货币趋势策略配置
crypto_config = StrategyResearchConfig(
    strategy_name="crypto_trend_following",
    instrument={
        "asset_type": "CRYPTO_SPOT",
        "symbol": "BTC/USDT",
        "venue": "binance",
        "quote_currency": "USDT",
        "lot_size": 0.0001,
        "allow_fractional": True,
        "shortable": True,
        "leverage": 3  # 加密货币杠杆
    },
    data_config={
        "frequency": "1H",  # 加密货币高频交易
        "start_date": "2022-01-01",
        "end_date": "2024-12-31",
        "source": "ccxt"
    },
    idea_generation={
        "technical_analysis": True,
        "behavioral_finance": True,
        "market_specific": ["CRYPTO"]  # 加密货币特殊因素
    },
    factor_design={
        "trend_indicators": {
            "enabled": True,
            "methods": ["SMA", "EMA", "MACD", "ADX"]
        },
        "volatility_indicators": {
            "enabled": True,
            "methods": ["ATR", "bollinger", "volatility_channel"]
        },
        "volume_indicators": {
            "enabled": True,
            "methods": ["OBV", "VWAP", "volume_price"]
        }
    },
    strategy_construction={
        "type": "multi_factor",  # 多因子策略应对高波动
        "factor_combination": ["linear_weighting", "nonlinear_combo"],
        "factor_selection": ["ic_stats", "ir_stats"]
    },
    # ... 其他配置
)

crypto_lifecycle = StrategyResearchLifecycle(crypto_config)
crypto_results = crypto_lifecycle.run_complete_research_cycle()
```

## 量化学习成果整合

此工作流深度整合了 `quant-learning-24h` 的21个阶段学习成果：

1. **市场基础理论**: Alpha101因子设计
2. **资产定价**: Fama-French多因子模型
3. **风险管理**: VaR、压力测试、优化理论
4. **机器学习应用**: 参数优化、交叉验证方法
5. **期权定价**: 波动率因子设计
6. **市场微观结构**: 交易成本建模
7. **果仁网实战**: 策略构建、回测、排名分析经验

## 策略评估与分级

根据 `quant-learning-24h` 的风险管理理念，系统会自动对策略进行分级：

- **A级策略**: 夏普>2, 卡尔马>2, 回撤<15%, 高一致性 → 可考虑实盘部署
- **B级策略**: 夏普1-2, 卡尔马1-2, 回撤15-25%, 中等一致性 → 小仓位试运行
- **C级策略**: 夏普0.5-1, 卡尔马0.5-1, 回撤25-35%, 低一致性 → 继续优化
- **D级策略**: 夏普0-0.5, 卡尔马0-0.5, 回撤35-50%, 无一致性 → 重新设计
- **F级策略**: 夏普<0, 卡尔马<0, 回撤>50%, 失败 → 彻底放弃

## 总结

这套策略研究工作流将 `quant-learning-24h` 的理论学习成果与 QuantLab 平台的技术优势相结合，实现了：

- **理论指导实践**: 基于扎实的量化金融理论
- **系统化流程**: 从创意到部署的完整闭环
- **高性能计算**: Polars + Arrow + DuckDB + VectorBT + Numba
- **多市场适配**: A股/美股/加密货币差异化处理
- **风险管控**: 全面的鲁棒性验证体系

这是从量化学习到实际应用的重要桥梁。