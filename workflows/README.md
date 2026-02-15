# QuantLab Strategy Research Workflow

基于 `quant-learning-24h` 学习成果构建的完整策略研发流程，整合了从创意生成到鲁棒性验证的端到端解决方案。

## 🎯 核心理念

结合 `quant-learning-24h` 的21个学习阶段成果，构建系统化的策略研发流程：
- **理论基础**: 市场基础、资产定价、风险管理、机器学习应用
- **实践洞察**: 果仁网量化平台实战经验
- **技术优势**: QuantLab平台的高性能计算能力

## 🔄 研发流程

### 1. 创意生成 (Idea Generation)
基于 `quant-learning-24h` 的多种分析流派：
- **技术分析**: 趋势/震荡/形态流派
- **基本面分析**: 价值/成长/质量流派  
- **行为金融**: 动量/反转/聪明钱流派
- **果仁网策略学习**: 优秀策略借鉴方法

### 2. 因子设计 (Factor Design)
利用 `quant-learning-24h` 的理论框架：
- **Alpha101因子**: 市场理论衍生
- **Fama-French因子**: 三因子、五因子模型
- **技术指标因子**: 果仁网平台积累
- **多维度因子**: 动量、反转、波动率、价值、质量、聪明钱、情绪

### 3. 策略构建 (Strategy Construction)
根据 `quant-learning-24h` 的实践经验：
- **单因子策略**: 简洁高效
- **多因子策略**: 机器学习融合
- **轮动策略**: 市场状态适应
- **择时策略**: 风险管理导向

### 4. 回测验证 (Backtest Validation)
集成 `quant-learning-24h` 的回测知识：
- **快速验证**: VectorBT引擎
- **精确验证**: Backtrader引擎
- **多市场适配**: A股/美股/加密货币
- **成本建模**: 佣金、滑点、冲击成本

### 5. 参数优化 (Parameter Optimization)
运用 `quant-learning-24h` 的优化理论：
- **粗到细优化**: 网格搜索 + 贝叶斯优化
- **多目标函数**: 夏普 + 卡尔马 + 换手率
- **过拟合防范**: 时间序列交叉验证
- **约束条件**: 最大回撤、换手率限制

### 6. 鲁棒性验证 (Robustness Validation)
应用 `quant-learning-24h` 的风险理念：
- **滚动验证**: 时间序列鲁棒性
- **自举检验**: 统计显著性
- **状态分析**: 牛熊市/高低波
- **敏感性分析**: 参数稳定性
- **泄露检测**: 前瞻偏差检查

## 📁 文件结构

```
workflows/
├── strategy_research_lifecycle.py    # 主要工作流实现
├── cli.py                           # 命令行接口
└── strategy_research_lifecycle.json # 工作流配置模板

configs/
└── workflows/
    └── strategy_research_workflow.json # 完整配置文件

src/quantlab/
└── workflows/
    ├── strategy_research_lifecycle.py # 核心实现
    └── cli.py                        # CLI入口
```

## 🚀 快速开始

### 1. 使用预设配置运行完整流程

```bash
# 安装（如果在虚拟环境中）
pip install -e .

# 运行完整研究周期
quantlab-research run --config configs/workflows/strategy_research_workflow.json --output-dir results/my_strategy

# 运行示例
quantlab-research example --output-dir results/example
```

### 2. 使用Python API

```python
from quantlab.workflows.strategy_research_lifecycle import StrategyResearchLifecycle, StrategyResearchConfig

# 配置研究参数
config = StrategyResearchConfig(
    strategy_name="my_momentum_strategy",
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
    # ... 其他配置
)

# 初始化研究流程
lifecycle = StrategyResearchLifecycle(config)

# 运行完整流程
results = lifecycle.run_complete_research_cycle()
```

### 3. 运行特定阶段

```bash
# 运行创意生成阶段
quantlab-research run-stage --config configs/workflows/strategy_research_workflow.json --stage idea-generation

# 运行因子设计阶段
quantlab-research run-stage --config configs/workflows/strategy_research_workflow.json --stage factor-design
```

## 📊 输出结果

每个阶段都会生成相应结果：

- **ideas**: 策略创意列表
- **factors**: 因子矩阵和统计
- **strategy**: 策略配置和信号
- **backtest**: 性能指标和资金曲线
- **optimization**: 最优参数和置信区间
- **robustness**: 验证报告和稳定性指标
- **final**: 综合评估和建议

## 🏆 评分系统

根据 `quant-learning-24h` 的风险管理理念：

- **A级**: 夏普>2, 卡尔马>2, 回撤<15%, 高一致性
- **B级**: 夏普1-2, 卡尔马1-2, 回撤15-25%, 中等一致性
- **C级**: 夏普0.5-1, 卡尔马0.5-1, 回撤25-35%, 低一致性
- **D级**: 夏普0-0.5, 卡尔马0-0.5, 回撤35-50%, 无一致性
- **F级**: 夏普<0, 卡尔马<0, 回撤>50%, 失败

## 🔗 与 quant-learning-24h 集成

- **理论基础**: 所有策略构建基于前21阶段的学习成果
- **实践验证**: 使用果仁网平台的实际经验
- **技术实现**: QuantLab平台的高性能架构
- **风险管控**: 完整的风险管理体系

## 🎯 后续步骤

1. **策略部署**: A/B级策略进入实盘监控
2. **参数微调**: C级策略进一步优化
3. **策略重构**: D/F级策略重新设计
4. **持续监控**: 上线策略的表现跟踪

---

**基于 `quant-learning-24h` 成果，构建于 QuantLab 平台之上，实现从理论到实践的完整闭环。**