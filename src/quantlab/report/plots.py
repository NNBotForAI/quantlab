"""
Lightweight plotting for reports
"""
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from ..common.logging import get_logger
from ..common.io import atomic_write

logger = get_logger(__name__)


class PlotGenerator:
    """
    Generate lightweight plots for reports using matplotlib.
    """

    def __init__(self, spec: dict):
        """
        Initialize plot generator.

        Args:
            spec: Strategy specification
        """
        self.spec = spec
        self.plots_dir = Path("plots")  # Will be saved relative to report

    def generate_all_plots(
        self,
        backtest_results: Dict[str, Any],
        optimization_results: Optional[Dict[str, Any]] = None,
        robustness_results: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """
        Generate all required plots.

        Args:
            backtest_results: Backtest results
            optimization_results: Optimization results (optional)
            robustness_results: Robustness results (optional)

        Returns:
            Dictionary mapping plot names to file paths
        """
        plots_dir = Path(self.spec.get("output_dir", "results")) / self.spec.get("run_id", "default") / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        plot_paths = {}

        # Equity curve
        equity_plot = self._generate_equity_curve(backtest_results, plots_dir)
        if equity_plot:
            plot_paths["equity_curve"] = str(equity_plot)

        # Drawdown
        drawdown_plot = self._generate_drawdown(backtest_results, plots_dir)
        if drawdown_plot:
            plot_paths["drawdown"] = str(drawdown_plot)

        # Monthly heatmap
        monthly_plot = self._generate_monthly_heatmap(backtest_results, plots_dir)
        if monthly_plot:
            plot_paths["monthly_heatmap"] = str(monthly_plot)

        # Sensitivity heatmap (if optimization results available)
        if optimization_results:
            sens_plot = self._generate_sensitivity_heatmap(optimization_results, plots_dir)
            if sens_plot:
                plot_paths["sensitivity_heatmap"] = str(sens_plot)

        # Bootstrap distribution (if robustness results available)
        if robustness_results:
            boot_plot = self._generate_bootstrap_dist(robustness_results, plots_dir)
            if boot_plot:
                plot_paths["bootstrap_dist"] = str(boot_plot)

        # Regime comparison (if robustness results available)
        if robustness_results:
            regime_plot = self._generate_regime_comparison(robustness_results, plots_dir)
            if regime_plot:
                plot_paths["regime_comparison"] = str(regime_plot)

        return plot_paths

    def _generate_equity_curve(
        self,
        backtest_results: Dict[str, Any],
        plots_dir: Path
    ) -> Optional[Path]:
        """
        Generate equity curve plot.

        Args:
            backtest_results: Backtest results
            plots_dir: Directory to save plots

        Returns:
            Path to generated plot
        """
        if "equity_curve" not in backtest_results:
            return None

        equity_df = backtest_results["equity_curve"]

        if equity_df.is_empty() or "equity" not in equity_df.columns:
            return None

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot equity curve
        if "ts_utc" in equity_df.columns:
            dates = equity_df["ts_utc"]
            equity_values = equity_df["equity"]
            ax.plot(dates, equity_values, linewidth=1.5, color='#2E86AB')
        else:
            equity_values = equity_df["equity"]
            ax.plot(equity_values, linewidth=1.5, color='#2E86AB')

        ax.set_title('Equity Curve', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Equity ($)')
        ax.grid(True, alpha=0.3)

        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        plt.tight_layout()

        # Save plot
        plot_path = plots_dir / "equity_curve.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        return plot_path

    def _generate_drawdown(
        self,
        backtest_results: Dict[str, Any],
        plots_dir: Path
    ) -> Optional[Path]:
        """
        Generate drawdown plot.

        Args:
            backtest_results: Backtest results
            plots_dir: Directory to save plots

        Returns:
            Path to generated plot
        """
        if "equity_curve" not in backtest_results:
            return None

        equity_df = backtest_results["equity_curve"]

        if equity_df.is_empty() or "equity" not in equity_df.columns:
            return None

        equity_values = equity_df["equity"].to_numpy()
        peak = np.maximum.accumulate(equity_values)
        drawdown = (equity_values - peak) / peak

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        if "ts_utc" in equity_df.columns:
            dates = equity_df["ts_utc"]
            ax.fill_between(dates, drawdown, 0, alpha=0.3, color='#A23B72')
            ax.plot(dates, drawdown, color='#F18F01', linewidth=1)
        else:
            ax.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='#A23B72')
            ax.plot(drawdown, color='#F18F01', linewidth=1)

        ax.set_title('Drawdown Analysis', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True, alpha=0.3)

        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))

        plt.tight_layout()

        # Save plot
        plot_path = plots_dir / "drawdown.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        return plot_path

    def _generate_monthly_heatmap(
        self,
        backtest_results: Dict[str, Any],
        plots_dir: Path
    ) -> Optional[Path]:
        """
        Generate monthly return heatmap.

        Args:
            backtest_results: Backtest results
            plots_dir: Directory to save plots

        Returns:
            Path to generated plot
        """
        if "equity_curve" not in backtest_results:
            return None

        equity_df = backtest_results["equity_curve"]

        if equity_df.is_empty() or "equity" not in equity_df.columns:
            return None

        equity_values = equity_df["equity"].to_numpy()

        # Calculate daily returns
        returns = np.diff(np.log(equity_values))
        dates = equity_df["ts_utc"].to_list()[1:] if "ts_utc" in equity_df.columns else list(range(1, len(returns)+1))

        # Group by year-month
        if "ts_utc" in equity_df.columns:
            # Convert to pandas for easier grouping
            import pandas as pd
            df_temp = pd.DataFrame({'date': [d.to_pydatetime() for d in dates], 'return': returns})
            df_temp['year'] = df_temp['date'].dt.year
            df_temp['month'] = df_temp['date'].dt.month
        else:
            # Fallback: create artificial year/month structure
            n_months = min(24, math.ceil(len(returns) / 21))  # Approximate months
            years = []
            months = []
            for i, ret in enumerate(returns):
                month_idx = i // 21  # Approximate 21 trading days per month
                year = 2020 + (month_idx // 12)
                month = (month_idx % 12) + 1
                years.append(year)
                months.append(month)
            df_temp = pd.DataFrame({'year': years, 'month': months, 'return': returns})

        # Calculate monthly returns
        monthly_returns = df_temp.groupby(['year', 'month'])['return'].sum().reset_index()
        monthly_returns['return_pct'] = monthly_returns['return'] * 100

        # Create pivot table
        pivot_table = monthly_returns.pivot(index='year', columns='month', values='return_pct')
        pivot_table = pivot_table.reindex(sorted(pivot_table.index), columns=range(1, 13))

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        im = ax.imshow(pivot_table.values, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=10)
        
        # Set ticks and labels
        ax.set_xticks(range(12))
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax.set_yticks(range(len(pivot_table.index)))
        ax.set_yticklabels(pivot_table.index)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Return (%)')
        
        ax.set_title('Monthly Returns Heatmap', fontsize=14, fontweight='bold')
        
        plt.tight_layout()

        # Save plot
        plot_path = plots_dir / "monthly_heatmap.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        return plot_path

    def _generate_sensitivity_heatmap(
        self,
        optimization_results: Dict[str, Any],
        plots_dir: Path
    ) -> Optional[Path]:
        """
        Generate sensitivity analysis heatmap.

        Args:
            optimization_results: Optimization results
            plots_dir: Directory to save plots

        Returns:
            Path to generated plot
        """
        if "multi_parameter" not in optimization_results:
            return None

        multi_param_results = optimization_results["multi_parameter"]

        if not multi_param_results or "metric_matrix" not in multi_param_results:
            return None

        metric_matrix = multi_param_results["metric_matrix"]
        param_names = multi_param_results.get("param_names", ["Param1", "Param2"])
        param_values = multi_param_results.get("param_values", [[], []])

        if len(metric_matrix) == 0 or len(param_names) < 2:
            return None

        # Convert to numpy array
        try:
            matrix = np.array(metric_matrix)
            if matrix.ndim == 1:
                # If 1D, we need to reshape based on parameter lengths
                if len(param_values) >= 2 and len(param_values[0]) * len(param_values[1]) == len(metric_matrix):
                    matrix = matrix.reshape(len(param_values[1]), len(param_values[0]))
                else:
                    return None
        except:
            return None

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(matrix, cmap='viridis', aspect='auto', origin='lower')
        
        # Set ticks and labels
        if len(param_values) >= 2:
            ax.set_xticks(range(len(param_values[0])))
            ax.set_xticklabels([f'{v:.2f}' for v in param_values[0]])
            ax.set_yticks(range(len(param_values[1])))
            ax.set_yticklabels([f'{v:.2f}' for v in param_values[1]])
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Performance Metric')
        
        ax.set_title(f'Sensitivity Analysis: {param_names[0]} vs {param_names[1]}', fontsize=14, fontweight='bold')
        ax.set_xlabel(param_names[0])
        ax.set_ylabel(param_names[1])
        
        plt.tight_layout()

        # Save plot
        plot_path = plots_dir / "sensitivity_heatmap.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        return plot_path

    def _generate_bootstrap_dist(
        self,
        robustness_results: Dict[str, Any],
        plots_dir: Path
    ) -> Optional[Path]:
        """
        Generate bootstrap distribution plot.

        Args:
            robustness_results: Robustness results
            plots_dir: Directory to save plots

        Returns:
            Path to generated plot
        """
        if "bootstrap_analysis" not in robustness_results:
            return None

        boot_results = robustness_results["bootstrap_analysis"]

        if "bootstrap_metrics" not in boot_results:
            return None

        bootstrap_metrics = boot_results["bootstrap_metrics"]

        if not bootstrap_metrics:
            return None

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(bootstrap_metrics, bins=50, density=True, alpha=0.7, color='#3498db', edgecolor='black')
        
        # Add vertical line for original metric
        if "original_metric" in boot_results:
            orig_metric = boot_results["original_metric"]
            ax.axvline(orig_metric, color='red', linestyle='--', linewidth=2, label=f'Original: {orig_metric:.4f}')
        
        ax.set_title('Bootstrap Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Metric Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()

        # Save plot
        plot_path = plots_dir / "bootstrap_dist.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        return plot_path

    def _generate_regime_comparison(
        self,
        robustness_results: Dict[str, Any],
        plots_dir: Path
    ) -> Optional[Path]:
        """
        Generate regime comparison plot.

        Args:
            robustness_results: Robustness results
            plots_dir: Directory to save plots

        Returns:
            Path to generated plot
        """
        if "regime_results" not in robustness_results:
            return None

        regime_results = robustness_results["regime_results"]

        if not regime_results:
            return None

        # Extract metrics for each regime
        regimes = list(regime_results.keys())
        sharpe_ratios = []
        calmar_ratios = []
        total_returns = []

        for regime in regimes:
            metrics = regime_results[regime].get("metrics", {})
            sharpe_ratios.append(metrics.get("sharpe_ratio", 0))
            calmar_ratios.append(metrics.get("calmar_ratio", 0))
            total_returns.append(metrics.get("total_return", 0))

        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Sharpe ratio bar chart
        bars1 = ax1.bar(range(len(regimes)), sharpe_ratios, color='#2ECC71')
        ax1.set_title('Sharpe Ratio by Regime')
        ax1.set_xlabel('Regime')
        ax1.set_ylabel('Sharpe Ratio')
        ax1.set_xticks(range(len(regimes)))
        ax1.set_xticklabels(regimes, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars1, sharpe_ratios):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Calmar ratio bar chart
        bars2 = ax2.bar(range(len(regimes)), calmar_ratios, color='#9B59B6')
        ax2.set_title('Calmar Ratio by Regime')
        ax2.set_xlabel('Regime')
        ax2.set_ylabel('Calmar Ratio')
        ax2.set_xticks(range(len(regimes)))
        ax2.set_xticklabels(regimes, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars2, calmar_ratios):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Total return bar chart
        bars3 = ax3.bar(range(len(regimes)), total_returns, color='#F39C12')
        ax3.set_title('Total Return by Regime')
        ax3.set_xlabel('Regime')
        ax3.set_ylabel('Total Return')
        ax3.set_xticks(range(len(regimes)))
        ax3.set_xticklabels(regimes, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars3, total_returns):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                     f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()

        # Save plot
        plot_path = plots_dir / "regime_comparison.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        return plot_path
