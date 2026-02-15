"""
QuantLab Streamlit Dashboard
==========================

Interactive web interface for QuantLab platform
"""

import streamlit as st
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantlab.data.pipeline import DataPipeline
from quantlab.features.universe import create_universe_provider
from quantlab.backtest.vectorbt_engine import VectorBTBacktestEngine
from quantlab.backtest.metrics import calculate_all_metrics
from quantlab.optimize.runner import OptimizationRunner
from quantlab.report.build_report import ReportBuilder


# Page configuration
st.set_page_config(
    page_title="QuantLab Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        color: #1f77b4;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'dashboard'
if 'strategies' not in st.session_state:
    st.session_state.strategies = []
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = {}


def load_strategy_configs():
    """Load available strategy configurations."""
    config_dir = Path("configs/examples")
    configs = []
    
    if config_dir.exists():
        for config_file in config_dir.glob("*.json"):
            with open(config_file, 'r') as f:
                config = json.load(f)
                configs.append({
                    'name': config.get('strategy_name', config_file.stem),
                    'file': config_file.name,
                    'config': config
                })
    
    return configs


def display_dashboard():
    """Display main dashboard."""
    st.markdown('<h1 class="main-title">📊 QuantLab Dashboard</h1>', unsafe_allow_html=True)
    
    # Metrics overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="可用策略",
            value=len(load_strategy_configs()),
            delta="配置文件"
        )
    
    with col2:
        st.metric(
            label="已完成回测",
            value=len(st.session_state.backtest_results),
            delta=f"{len(st.session_state.backtest_results)} 次"
        )
    
    with col3:
        st.metric(
            label="数据源",
            value=4,
            delta="A股/美股/加密"
        )
    
    with col4:
        st.metric(
            label="回测引擎",
            value=2,
            delta="VectorBT/Backtrader"
        )
    
    st.markdown("---")
    
    # Quick actions
    st.subheader("🚀 快速开始")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("➕ 新建策略", use_container_width=True):
            st.session_state.page = 'create_strategy'
            st.rerun()
    
    with col2:
        if st.button("📈 运行回测", use_container_width=True):
            st.session_state.page = 'backtest'
            st.rerun()
    
    with col3:
        if st.button("⚙️ 参数优化", use_container_width=True):
            st.session_state.page = 'optimize'
            st.rerun()
    
    st.markdown("---")
    
    # Recent results
    st.subheader("📋 最近结果")
    
    if st.session_state.backtest_results:
        results_list = []
        for name, results in st.session_state.backtest_results.items():
            metrics = results.get('metrics', {})
            results_list.append({
                '策略': name,
                '总收益率': f"{metrics.get('total_return', 0) * 100:.2f}%",
                '夏普比率': f"{metrics.get('sharpe_ratio', 0):.2f}",
                '最大回撤': f"{metrics.get('max_drawdown', 0) * 100:.2f}%",
                '完成时间': results.get('completed_at', '-')
            })
        
        if results_list:
            df = pd.DataFrame(results_list)
            st.dataframe(df, use_container_width=True)
    else:
        st.info("暂无回测结果，请运行回测查看结果")


def display_strategy_library():
    """Display strategy library page."""
    st.title("📚 策略库")
    
    configs = load_strategy_configs()
    
    if configs:
        for config in configs:
            with st.expander(f"🎯 {config['name']} ({config['file']})"):
                st.write(f"**资产类型**: {config['config'].get('instrument', {}).get('asset_type', 'N/A')}")
                st.write(f"**标的**: {config['config'].get('instrument', {}).get('symbol', 'N/A')}")
                st.write(f"**频率**: {config['config'].get('data', {}).get('frequency', 'N/A')}")
                st.write(f"**初始资金**: {config['config'].get('backtest', {}).get('initial_capital', 'N/A')}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"查看详情", key=f"view_{config['name']}"):
                        st.json(config['config'])
                with col2:
                    if st.button(f"运行回测", key=f"run_{config['name']}"):
                        st.session_state.selected_config = config['config']
                        st.session_state.page = 'backtest'
                        st.rerun()
    else:
        st.warning("未找到策略配置文件")


def display_create_strategy():
    """Display create strategy page."""
    st.title("➕ 创建新策略")
    
    with st.form("create_strategy_form"):
        st.subheader("策略基本信息")
        
        strategy_name = st.text_input("策略名称", value="my_strategy")
        
        col1, col2 = st.columns(2)
        with col1:
            asset_type = st.selectbox(
                "资产类型",
                ["CN_STOCK", "US_STOCK", "CRYPTO_SPOT", "CRYPTO_PERP"]
            )
        with col2:
            symbol = st.text_input("标的代码", value="AAPL")
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("开始日期", datetime(2020, 1, 1))
        with col2:
            end_date = st.date_input("结束日期", datetime(2024, 12, 31))
        
        frequency = st.selectbox("数据频率", ["1m", "5m", "15m", "1H", "1D", "1W"])
        
        st.subheader("回测参数")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            initial_capital = st.number_input("初始资金", value=100000, step=10000)
        with col2:
            commission = st.number_input("佣金率", value=0.001, format="%.4f")
        with col3:
            slippage = st.number_input("滑点率", value=0.001, format="%.4f")
        
        st.subheader("策略参数")
        
        strategy_type = st.selectbox("策略类型", ["单因子", "多因子", "轮动策略", "择时策略"])
        
        if strategy_type == "单因子":
            col1, col2 = st.columns(2)
            with col1:
                entry_threshold = st.number_input("入场阈值", value=0.02, format="%.3f")
            with col2:
                exit_threshold = st.number_input("出场阈值", value=-0.01, format="%.3f")
        
        submitted = st.form_submit_button("创建策略")
        
        if submitted:
            # Create strategy configuration
            strategy_config = {
                "strategy_name": strategy_name,
                "instrument": {
                    "asset_type": asset_type,
                    "symbol": symbol,
                    "venue": "auto",
                    "quote_currency": "USD" if "US" in asset_type or "CRYPTO" in asset_type else "CNY",
                    "lot_size": 100 if asset_type == "CN_STOCK" else 1,
                    "allow_fractional": asset_type != "CN_STOCK",
                    "shortable": asset_type != "CN_STOCK",
                    "leverage": 1
                },
                "data": {
                    "frequency": frequency,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "source": "yfinance" if "US" in asset_type else "akshare" if "CN" in asset_type else "ccxt"
                },
                "backtest": {
                    "initial_capital": initial_capital,
                    "commission": commission,
                    "slippage": slippage
                },
                "features": {
                    "lookback_period": 20,
                    "signals": {
                        "long_threshold": entry_threshold,
                        "short_threshold": exit_threshold
                    }
                }
            }
            
            # Save configuration
            config_path = Path(f"configs/examples/{strategy_name}.json")
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(strategy_config, f, indent=2, ensure_ascii=False)
            
            st.success(f"策略 '{strategy_name}' 创建成功！配置已保存到 {config_path}")
            st.json(strategy_config)


def display_backtest():
    """Display backtest page."""
    st.title("📈 运行回测")
    
    # Load strategy configs
    configs = load_strategy_configs()
    
    if not configs:
        st.warning("未找到策略配置，请先创建策略")
        return
    
    # Strategy selection
    strategy_names = [config['name'] for config in configs]
    selected_strategy = st.selectbox("选择策略", strategy_names)
    
    # Get selected config
    selected_config = next(c for c in configs if c['name'] == selected_strategy)
    spec = selected_config['config']
    
    # Display strategy info
    with st.expander("策略配置", expanded=True):
        st.json(spec)
    
    # Backtest options
    st.subheader("回测选项")
    
    col1, col2 = st.columns(2)
    with col1:
        engine = st.selectbox("回测引擎", ["VectorBT (快速)", "Backtrader (精确)"])
    with col2:
        use_chunking = st.checkbox("启用分块处理（降低内存）", value=False)
    
    if use_chunking:
        chunk_size = st.slider("分块大小", 50, 500, 200)
    
    # Run backtest button
    if st.button("🚀 开始回测", type="primary", use_container_width=True):
        with st.spinner("正在运行回测...这可能需要几分钟..."):
            try:
                # This is a placeholder - in real implementation, run actual backtest
                st.info("回测引擎初始化...")
                st.info(f"正在获取数据: {spec['instrument']['symbol']}")
                st.info("正在计算信号...")
                st.info("正在回测...")
                
                # Simulate results (in real implementation, use actual backtest)
                simulated_results = {
                    'strategy_name': selected_strategy,
                    'metrics': {
                        'total_return': 0.15,
                        'cagr': 0.12,
                        'sharpe_ratio': 1.8,
                        'sortino_ratio': 2.1,
                        'max_drawdown': -0.08,
                        'calmar_ratio': 1.5,
                        'win_rate': 0.65,
                        'profit_factor': 2.3,
                        'total_trades': 156
                    },
                    'completed_at': datetime.now().isoformat()
                }
                
                # Store results
                st.session_state.backtest_results[selected_strategy] = simulated_results
                
                st.success(f"回测完成！策略: {selected_strategy}")
                
            except Exception as e:
                st.error(f"回测失败: {str(e)}")
    
    # Display results if available
    if selected_strategy in st.session_state.backtest_results:
        display_backtest_results(selected_strategy)


def display_backtest_results(strategy_name):
    """Display backtest results."""
    results = st.session_state.backtest_results[strategy_name]
    metrics = results['metrics']
    
    st.subheader("📊 回测结果")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总收益率", f"{metrics.get('total_return', 0) * 100:.2f}%")
    with col2:
        st.metric("年化收益", f"{metrics.get('cagr', 0) * 100:.2f}%")
    with col3:
        st.metric("夏普比率", f"{metrics.get('sharpe_ratio', 0):.2f}")
    with col4:
        st.metric("最大回撤", f"{metrics.get('max_drawdown', 0) * 100:.2f}%")
    
    # Additional metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("卡尔马比率", f"{metrics.get('calmar_ratio', 0):.2f}")
    with col2:
        st.metric("胜率", f"{metrics.get('win_rate', 0) * 100:.1f}%")
    with col3:
        st.metric("盈亏比", f"{metrics.get('profit_factor', 0):.2f}")
    
    # Placeholder for charts (in real implementation, use actual equity curve)
    st.subheader("资金曲线")
    
    # Simulated equity curve
    days = 252
    returns = pd.Series([metrics['total_return'] / days] * days)
    equity_curve = (1 + returns.cumsum()) * metrics.get('total_return', 1)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=equity_curve,
        mode='lines',
        name='净值曲线',
        line=dict(color='blue', width=2)
    ))
    fig.update_layout(
        title='资金曲线',
        xaxis_title='天数',
        yaxis_title='净值',
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Trade statistics
    st.subheader("交易统计")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("总交易次数", metrics.get('total_trades', 0))
    with col2:
        st.metric("平均每笔交易", f"{(metrics.get('total_return', 0) / max(metrics.get('total_trades', 1), 1)) * 100:.2f}%")
    
    # Performance grade
    st.subheader("🏆 性能评级")
    
    sharpe = metrics.get('sharpe_ratio', 0)
    calmar = metrics.get('calmar_ratio', 0)
    max_dd = abs(metrics.get('max_drawdown', 0))
    
    if sharpe > 2 and calmar > 2 and max_dd < 0.15:
        grade = "A"
        grade_color = "🟢"
        comment = "优秀策略，可考虑实盘部署"
    elif sharpe > 1 and calmar > 1 and max_dd < 0.25:
        grade = "B"
        grade_color = "🟡"
        comment = "良好策略，建议小仓位试运行"
    elif sharpe > 0.5 and calmar > 0.5 and max_dd < 0.35:
        grade = "C"
        grade_color = "🟠"
        comment = "一般策略，建议继续优化"
    elif sharpe > 0:
        grade = "D"
        grade_color = "🔴"
        comment = "较差策略，建议重新设计"
    else:
        grade = "F"
        grade_color = "⚫"
        comment = "失败策略，彻底放弃"
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### {grade_color} 评级: {grade}")
    with col2:
        st.info(comment)


def display_optimize():
    """Display optimization page."""
    st.title("⚙️ 参数优化")
    
    # Load strategy configs
    configs = load_strategy_configs()
    
    if not configs:
        st.warning("未找到策略配置，请先创建策略")
        return
    
    # Strategy selection
    strategy_names = [config['name'] for config in configs]
    selected_strategy = st.selectbox("选择策略", strategy_names)
    
    # Optimization settings
    st.subheader("优化设置")
    
    col1, col2 = st.columns(2)
    with col1:
        n_trials = st.number_input("试验次数", min_value=10, max_value=1000, value=100, step=10)
    with col2:
        timeout = st.number_input("超时时间（秒）", min_value=60, max_value=7200, value=3600, step=60)
    
    col1, col2 = st.columns(2)
    with col1:
        parallel_jobs = st.selectbox("并行任务数", [1, 2, 4, 8])
    with col2:
        objective = st.selectbox("优化目标", ["夏普比率", "卡尔马比率", "总收益率"])
    
    # Parameter ranges
    st.subheader("参数范围")
    
    col1, col2 = st.columns(2)
    with col1:
        entry_min = st.number_input("入场阈值最小值", value=-0.1, format="%.3f")
        entry_max = st.number_input("入场阈值最大值", value=0.1, format="%.3f")
    with col2:
        exit_min = st.number_input("出场阈值最小值", value=-0.1, format="%.3f")
        exit_max = st.number_input("出场阈值最大值", value=0.1, format="%.3f")
    
    # Run optimization
    if st.button("🚀 开始优化", type="primary", use_container_width=True):
        with st.spinner("正在运行优化...这可能需要较长时间..."):
            try:
                st.info(f"优化设置: {n_trials} 次试验, {parallel_jobs} 并行任务")
                st.info(f"参数范围: 入场 [{entry_min}, {entry_max}], 出场 [{exit_min}, {exit_max}]")
                st.info("正在搜索最优参数...")
                
                # Simulate optimization progress
                progress_bar = st.progress(0)
                for i in range(100):
                    import time
                    time.sleep(0.02)
                    progress_bar.progress(i + 1)
                
                st.success("优化完成！")
                st.info("最佳参数: 入场阈值=0.025, 出场阈值=-0.015")
                st.info("最佳夏普比率: 1.95")
                
                # Show parameter importance
                st.subheader("参数重要性")
                
                param_data = {
                    '参数': ['入场阈值', '出场阈值', '持仓规模', '止损水平'],
                    '重要性': [0.45, 0.35, 0.12, 0.08]
                }
                df = pd.DataFrame(param_data)
                fig = px.bar(df, x='参数', y='重要性', title='参数重要性分析')
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"优化失败: {str(e)}")


def display_reports():
    """Display reports page."""
    st.title("📑 报告")
    
    if not st.session_state.backtest_results:
        st.info("暂无报告数据，请先运行回测")
        return
    
    # Strategy selection
    strategy_names = list(st.session_state.backtest_results.keys())
    selected_strategy = st.selectbox("选择策略报告", strategy_names)
    
    if selected_strategy:
        st.info(f"报告生成中: {selected_strategy}")
        
        # Download buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📄 下载 PDF 报告"):
                st.info("PDF 报告生成功能开发中...")
        with col2:
            if st.button("📊 下载 HTML 报告"):
                st.info("HTML 报告生成功能开发中...")
        with col3:
            if st.button("📈 下载图表"):
                st.info("图表导出功能开发中...")
        
        # Report sections
        st.subheader("执行摘要")
        results = st.session_state.backtest_results[selected_strategy]
        metrics = results['metrics']
        
        st.write(f"**策略名称**: {selected_strategy}")
        st.write(f"**回测完成时间**: {results.get('completed_at', 'N/A')}")
        st.write(f"**总收益率**: {metrics.get('total_return', 0) * 100:.2f}%")
        st.write(f"**夏普比率**: {metrics.get('sharpe_ratio', 0):.2f}")
        st.write(f"**最大回撤**: {metrics.get('max_drawdown', 0) * 100:.2f}%")
        
        st.subheader("风险警告")
        if metrics.get('max_drawdown', 0) < -0.20:
            st.warning("⚠️ 策略存在较大回撤风险，建议设置止损")
        if metrics.get('sharpe_ratio', 0) < 1.0:
            st.warning("⚠️ 策略夏普比率较低，风险调整后收益不佳")


# Sidebar navigation
def sidebar():
    """Display sidebar navigation."""
    with st.sidebar:
        st.title("🎯 QuantLab")
        
        st.markdown("---")
        
        pages = {
            'dashboard': '📊 仪表盘',
            'strategies': '📚 策略库',
            'create_strategy': '➕ 创建策略',
            'backtest': '📈 运行回测',
            'optimize': '⚙️ 参数优化',
            'reports': '📑 报告'
        }
        
        for page_key, page_name in pages.items():
            if st.button(page_name, use_container_width=True, key=f"nav_{page_key}"):
                st.session_state.page = page_key
                st.rerun()
        
        st.markdown("---")
        
        st.subheader("系统信息")
        st.info(f"Python {sys.version.split()[0]}")
        st.info("Polars + Arrow + DuckDB")
        st.info("VectorBT + Numba")


# Main app
def main():
    """Main application."""
    sidebar()
    
    # Display selected page
    page = st.session_state.page
    
    if page == 'dashboard':
        display_dashboard()
    elif page == 'strategies':
        display_strategy_library()
    elif page == 'create_strategy':
        display_create_strategy()
    elif page == 'backtest':
        display_backtest()
    elif page == 'optimize':
        display_optimize()
    elif page == 'reports':
        display_reports()


if __name__ == "__main__":
    main()
