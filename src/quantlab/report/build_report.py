"""
Build comprehensive strategy report
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import jinja2
import polars as pl

from ..common.logging import get_logger
from ..common.io import atomic_write
from .plots import PlotGenerator
from .summarize import ReportSummarizer

logger = get_logger(__name__)


class ReportBuilder:
    """
    Build comprehensive strategy performance report.
    """

    def __init__(self, spec: dict, results_dir: Path):
        """
        Initialize report builder.

        Args:
            spec: Strategy specification
            results_dir: Results directory
        """
        self.spec = spec
        self.results_dir = results_dir
        self.run_id = spec.get("run_id", "unknown")
        
        # Initialize components
        self.plot_generator = PlotGenerator(spec)
        self.summarizer = ReportSummarizer(spec)

    def build_report(
        self,
        backtest_results: Dict[str, Any],
        optimization_results: Optional[Dict[str, Any]] = None,
        robustness_results: Optional[Dict[str, Any]] = None,
        leakage_results: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Build comprehensive report.

        Args:
            backtest_results: Backtest results
            optimization_results: Optimization results (optional)
            robustness_results: Robustness results (optional)
            leakage_results: Leakage check results (optional)

        Returns:
            HTML report as string
        """
        logger.info("report_build_start", run_id=self.run_id)

        # Generate plots
        plot_paths = self.plot_generator.generate_all_plots(
            backtest_results=backtest_results,
            optimization_results=optimization_results,
            robustness_results=robustness_results,
        )

        # Generate summary
        summary = self.summarizer.generate_summary(
            backtest_results=backtest_results,
            optimization_results=optimization_results,
            robustness_results=robustness_results,
            leakage_results=leakage_results,
        )

        # Load template
        template_path = Path(__file__).parent / "templates" / "report.html.j2"
        if not template_path.exists():
            # Create default template if not exists
            self._create_default_template(template_path)

        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()

        template = jinja2.Template(template_content)

        # Prepare report data
        report_data = {
            "spec": self.spec,
            "summary": summary,
            "plots": plot_paths,
            "backtest_results": backtest_results,
            "optimization_results": optimization_results or {},
            "robustness_results": robustness_results or {},
            "leakage_results": leakage_results or {},
            "generated_at": datetime.utcnow().isoformat(),
            "run_id": self.run_id,
        }

        # Render report
        html_report = template.render(**report_data)

        # Save report
        report_path = self.results_dir / self.run_id / "report.html"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(report_path, html_report)

        logger.info("report_build_complete", path=str(report_path))

        return html_report

    def _create_default_template(self, template_path: Path) -> None:
        """
        Create default report template.

        Args:
            template_path: Path to create template
        """
        default_template = """<!DOCTYPE html>
<html>
<head>
    <title>QuantLab Strategy Report - {{ run_id }}</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { text-align: center; border-bottom: 2px solid #ccc; padding-bottom: 20px; }
        .section { margin: 30px 0; }
        .subsection { margin: 20px 0; }
        .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }
        .metric-card { border: 1px solid #ddd; padding: 15px; text-align: center; }
        .metric-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
        .metric-name { font-size: 14px; color: #7f8c8d; }
        .plot-container { margin: 20px 0; text-align: center; }
        .plot-image { max-width: 100%; height: auto; }
        .table { width: 100%; border-collapse: collapse; margin: 15px 0; }
        .table th, .table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        .table th { background-color: #f2f2f2; }
        .warning { background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; margin: 10px 0; }
        .error { background-color: #f8d7da; border: 1px solid #f5c6cb; padding: 10px; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="header">
        <h1>QuantLab Strategy Report</h1>
        <p><strong>Run ID:</strong> {{ run_id }}</p>
        <p><strong>Generated:</strong> {{ generated_at }}</p>
        <p><strong>Strategy:</strong> {{ spec.strategy_name }}</p>
    </div>

    {% if summary %}
    <div class="section">
        <h2>Executive Summary</h2>
        <div class="summary-grid">
            {% for metric, value in summary.key_metrics.items() %}
            <div class="metric-card">
                <div class="metric-value">{{ "%.4g"|format(value) if value is number else value }}</div>
                <div class="metric-name">{{ metric.replace('_', ' ').title() }}</div>
            </div>
            {% endfor %}
        </div>
        {% if summary.risk_warnings %}
        <div class="warning">
            <h3>Risk Warnings</h3>
            <ul>
            {% for warning in summary.risk_warnings %}
            <li>{{ warning }}</li>
            {% endfor %}
            </ul>
        </div>
        {% endif %}
    </div>
    {% endif %}

    <div class="section">
        <h2>Performance Analysis</h2>
        {% if plots.equity_curve %}
        <div class="plot-container">
            <h3>Equity Curve</h3>
            <img src="{{ plots.equity_curve }}" alt="Equity Curve" class="plot-image">
        </div>
        {% endif %}

        {% if plots.drawdown %}
        <div class="plot-container">
            <h3>Drawdown Analysis</h3>
            <img src="{{ plots.drawdown }}" alt="Drawdown" class="plot-image">
        </div>
        {% endif %}

        {% if backtest_results.metrics %}
        <h3>Metrics Table</h3>
        <table class="table">
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
            </thead>
            <tbody>
                {% for metric, value in backtest_results.metrics.items() %}
                <tr>
                    <td>{{ metric.replace('_', ' ').title() }}</td>
                    <td>{{ "%.4g"|format(value) if value is number else value }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% endif %}
    </div>

    {% if optimization_results %}
    <div class="section">
        <h2>Optimization Results</h2>
        {% if plots.sensitivity_heatmap %}
        <div class="plot-container">
            <h3>Sensitivity Analysis</h3>
            <img src="{{ plots.sensitivity_heatmap }}" alt="Sensitivity Heatmap" class="plot-image">
        </div>
        {% endif %}

        {% if optimization_results.best_params %}
        <h3>Best Parameters</h3>
        <table class="table">
            <thead>
                <tr>
                    <th>Parameter</th>
                    <th>Value</th>
                </tr>
            </thead>
            <tbody>
                {% for param, value in optimization_results.best_params.items() %}
                <tr>
                    <td>{{ param.replace('_', ' ').title() }}</td>
                    <td>{{ "%.4g"|format(value) if value is number else value }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% endif %}
    </div>
    {% endif %}

    {% if robustness_results %}
    <div class="section">
        <h2>Robustness Analysis</h2>
        {% if robustness_results.summary %}
        <h3>Robustness Summary</h3>
        <table class="table">
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
            </thead>
            <tbody>
                {% for metric, value in robustness_results.summary.items() %}
                {% if value is mapping %}
                    {% for sub_metric, sub_value in value.items() %}
                    <tr>
                        <td>{{ metric.replace('_', ' ')|title }} - {{ sub_metric.replace('_', ' ')|title }}</td>
                        <td>{{ "%.4g"|format(sub_value) if sub_value is number else sub_value }}</td>
                    </tr>
                    {% endfor %}
                {% else %}
                    <tr>
                        <td>{{ metric.replace('_', ' ')|title }}</td>
                        <td>{{ "%.4g"|format(value) if value is number else value }}</td>
                    </tr>
                {% endif %}
                {% endfor %}
            </tbody>
        </table>
        {% endif %}
    </div>
    {% endif %}

    {% if leakage_results and leakage_results.summary %}
    <div class="section">
        <h2>Leakage Detection</h2>
        {% if leakage_results.summary.passed %}
        <div style="color: green;">✓ No significant leakage detected</div>
        {% else %}
        <div class="error">
            <h3>Issues Detected</h3>
            <ul>
            {% for issue in leakage_results.summary.issues %}
            <li><strong>{{ issue.check }}:</strong> {{ issue.details }}</li>
            {% endfor %}
            </ul>
        </div>
        {% endif %}
    </div>
    {% endif %}

    <div class="section">
        <h2>Configuration</h2>
        <pre>{{ spec | tojson(indent=2) }}</pre>
    </div>
</body>
</html>"""

        template_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(template_path, default_template)
