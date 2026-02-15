"""
Report summarization utilities
"""
from datetime import datetime
from typing import Dict, Any, Optional, List

import numpy as np

from ..common.logging import get_logger

logger = get_logger(__name__)


class ReportSummarizer:
    """
    Generate executive summary and key insights from results.
    """

    def __init__(self, spec: dict):
        """
        Initialize report summarizer.

        Args:
            spec: Strategy specification
        """
        self.spec = spec

    def generate_summary(
        self,
        backtest_results: Dict[str, Any],
        optimization_results: Optional[Dict[str, Any]] = None,
        robustness_results: Optional[Dict[str, Any]] = None,
        leakage_results: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate executive summary.

        Args:
            backtest_results: Backtest results
            optimization_results: Optimization results (optional)
            robustness_results: Robustness results (optional)
            leakage_results: Leakage check results (optional)

        Returns:
            Summary dictionary
        """
        summary = {
            "key_metrics": {},
            "performance_summary": "",
            "risk_warnings": [],
            "optimization_insights": "",
            "robustness_assessment": "",
            "leakage_assessment": "",
            "overall_grade": "",
            "recommendations": [],
        }

        # Add key metrics
        summary["key_metrics"] = self._extract_key_metrics(backtest_results)

        # Generate performance summary
        summary["performance_summary"] = self._generate_performance_summary(backtest_results)

        # Identify risk warnings
        summary["risk_warnings"] = self._identify_risk_warnings(backtest_results)

        # Add optimization insights
        if optimization_results:
            summary["optimization_insights"] = self._generate_optimization_insights(optimization_results)

        # Add robustness assessment
        if robustness_results:
            summary["robustness_assessment"] = self._generate_robustness_assessment(robustness_results)

        # Add leakage assessment
        if leakage_results:
            summary["leakage_assessment"] = self._generate_leakage_assessment(leakage_results)

        # Generate overall grade
        summary["overall_grade"] = self._calculate_overall_grade(
            backtest_results, robustness_results, leakage_results
        )

        # Generate recommendations
        summary["recommendations"] = self._generate_recommendations(
            backtest_results, robustness_results, leakage_results
        )

        return summary

    def _extract_key_metrics(self, backtest_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract key performance metrics.

        Args:
            backtest_results: Backtest results

        Returns:
            Dictionary of key metrics
        """
        metrics = backtest_results.get("metrics", {})

        key_metrics = {}
        for key_metric in ["sharpe_ratio", "calmar_ratio", "max_drawdown", 
                          "total_return", "win_rate", "profit_factor", "cagr"]:
            if key_metric in metrics:
                key_metrics[key_metric] = metrics[key_metric]

        return key_metrics

    def _generate_performance_summary(self, backtest_results: Dict[str, Any]) -> str:
        """
        Generate performance summary text.

        Args:
            backtest_results: Backtest results

        Returns:
            Performance summary string
        """
        metrics = backtest_results.get("metrics", {})

        sharpe = metrics.get("sharpe_ratio", 0)
        calmar = metrics.get("calmar_ratio", 0)
        total_return = metrics.get("total_return", 0)
        max_dd = metrics.get("max_drawdown", 0)

        # Determine performance level
        if sharpe > 2:
            perf_level = "excellent"
        elif sharpe > 1:
            perf_level = "good"
        elif sharpe > 0.5:
            perf_level = "fair"
        elif sharpe > 0:
            perf_level = "poor"
        else:
            perf_level = "negative"

        return (f"This strategy demonstrates {perf_level} risk-adjusted performance "
                f"with a Sharpe ratio of {sharpe:.2f} and Calmar ratio of {calmar:.2f}. "
                f"The total return is {total_return:.2%} with a maximum drawdown of {max_dd:.2%}.")

    def _identify_risk_warnings(self, backtest_results: Dict[str, Any]) -> List[str]:
        """
        Identify risk warnings from results.

        Args:
            backtest_results: Backtest results

        Returns:
            List of risk warnings
        """
        warnings = []
        metrics = backtest_results.get("metrics", {})

        # Check for high drawdown
        max_dd = metrics.get("max_drawdown", 0)
        if abs(max_dd) > 0.25:  # 25% drawdown
            warnings.append(f"High maximum drawdown: {max_dd:.2%}")

        # Check for low Sharpe ratio
        sharpe = metrics.get("sharpe_ratio", 0)
        if sharpe < 0.5:
            warnings.append(f"Low Sharpe ratio: {sharpe:.2f} (considered below threshold of 0.5)")

        # Check for high turnover
        turnover = metrics.get("turnover", 0)
        if turnover > 5:  # 500% annual turnover
            warnings.append(f"High turnover: {turnover:.2f} (may lead to high transaction costs)")

        # Check for low win rate
        win_rate = metrics.get("win_rate", 0)
        if win_rate < 0.4:  # 40% win rate
            warnings.append(f"Low win rate: {win_rate:.2%}")

        # Check for profit factor
        profit_factor = metrics.get("profit_factor", 0)
        if profit_factor < 1.5:
            warnings.append(f"Low profit factor: {profit_factor:.2f} (should be > 1.5 for robust strategies)")

        return warnings

    def _generate_optimization_insights(self, optimization_results: Dict[str, Any]) -> str:
        """
        Generate insights from optimization results.

        Args:
            optimization_results: Optimization results

        Returns:
            Optimization insights string
        """
        if not optimization_results:
            return ""

        best_params = optimization_results.get("best_params", {})
        best_score = optimization_results.get("best_score", 0)

        if not best_params:
            return "No optimization results available."

        param_desc = ", ".join([f"{k}: {v:.3f}" for k, v in list(best_params.items())[:3]])

        return (f"Optimization identified best parameters: {param_desc} "
                f"with a score of {best_score:.4f}. "
                f"{len(best_params)} parameters were optimized.")

    def _generate_robustness_assessment(self, robustness_results: Dict[str, Any]) -> str:
        """
        Generate robustness assessment.

        Args:
            robustness_results: Robustness results

        Returns:
            Robustness assessment string
        """
        if not robustness_results:
            return ""

        assessments = []

        # Check walk-forward results
        if "walk_forward_analysis" in robustness_results:
            wf_results = robustness_results["walk_forward_analysis"]
            if "aggregated" in wf_results:
                agg = wf_results["aggregated"]
                
                if "sharpe_ratio" in agg:
                    sharpe_mean = agg["sharpe_ratio"]["mean"]
                    sharpe_std = agg["sharpe_ratio"]["std"]
                    if sharpe_std > 0 and sharpe_mean != 0:
                        stability = abs(sharpe_mean) / (abs(sharpe_mean) + sharpe_std)
                        if stability > 0.8:
                            assessments.append("Strategy shows strong consistency across time periods.")
                        elif stability > 0.5:
                            assessments.append("Strategy shows moderate consistency across time periods.")
                        else:
                            assessments.append("Strategy shows poor consistency across time periods.")

        # Check sensitivity results
        if "sensitivity_analysis" in robustness_results:
            sens_results = robustness_results["sensitivity_analysis"]
            if "single_parameter" in sens_results:
                sens_data = sens_results["single_parameter"]
                
                high_sensitivity_params = []
                for param, data in sens_data.items():
                    sensitivity = data.get("sensitivity", 0)
                    if abs(sensitivity) > 0.5:  # Threshold for high sensitivity
                        high_sensitivity_params.append(param)
                
                if high_sensitivity_params:
                    assessments.append(f"Strategy is highly sensitive to: {', '.join(high_sensitivity_params)}")

        return " ".join(assessments) if assessments else "No significant robustness concerns identified."

    def _generate_leakage_assessment(self, leakage_results: Dict[str, Any]) -> str:
        """
        Generate leakage assessment.

        Args:
            leakage_results: Leakage check results

        Returns:
            Leakage assessment string
        """
        if not leakage_results or "summary" not in leakage_results:
            return "No leakage analysis performed."

        summary = leakage_results["summary"]

        if summary.get("passed", False):
            return "No significant future leakage detected. The strategy appears to be properly constructed without lookahead bias."
        else:
            issues = summary.get("issues", [])
            issue_types = [issue.get("check", "unknown") for issue in issues]
            return f"Potential future leakage detected in: {', '.join(set(issue_types))}. Review methodology carefully."

    def _calculate_overall_grade(self, 
                               backtest_results: Dict[str, Any],
                               robustness_results: Optional[Dict[str, Any]],
                               leakage_results: Optional[Dict[str, Any]]) -> str:
        """
        Calculate overall strategy grade.

        Args:
            backtest_results: Backtest results
            robustness_results: Robustness results
            leakage_results: Leakage check results

        Returns:
            Grade (A-F)
        """
        metrics = backtest_results.get("metrics", {})

        # Base grade on performance metrics
        sharpe = metrics.get("sharpe_ratio", 0)
        calmar = metrics.get("calmar_ratio", 0)
        max_dd = abs(metrics.get("max_drawdown", 0))

        # Performance score (0-100)
        perf_score = 0
        if sharpe > 2: perf_score += 25
        elif sharpe > 1: perf_score += 20
        elif sharpe > 0.5: perf_score += 15
        elif sharpe > 0: perf_score += 10

        if calmar > 2: perf_score += 25
        elif calmar > 1: perf_score += 20
        elif calmar > 0.5: perf_score += 15
        elif calmar > 0: perf_score += 10

        # Deduct for high drawdown
        if max_dd < 0.1: perf_score += 10
        elif max_dd < 0.15: perf_score += 5
        elif max_dd > 0.3: perf_score -= 10

        # Robustness adjustment
        if robustness_results:
            # If walk-forward shows consistency
            if "walk_forward_analysis" in robustness_results:
                wf_results = robustness_results["walk_forward_analysis"]
                if "aggregated" in wf_results and "consistency_score" in wf_results["aggregated"]:
                    consistency = wf_results["aggregated"]["consistency_score"]
                    perf_score += consistency * 20  # Up to +20 points for consistency

        # Leakage adjustment
        if leakage_results:
            if not leakage_results.get("summary", {}).get("passed", True):
                # Significant penalty for leakage
                perf_score -= 30

        # Convert to letter grade
        if perf_score >= 85: return "A"
        elif perf_score >= 70: return "B"
        elif perf_score >= 55: return "C"
        elif perf_score >= 40: return "D"
        else: return "F"

    def _generate_recommendations(self,
                                backtest_results: Dict[str, Any],
                                robustness_results: Optional[Dict[str, Any]],
                                leakage_results: Optional[Dict[str, Any]]) -> List[str]:
        """
        Generate recommendations based on results.

        Args:
            backtest_results: Backtest results
            robustness_results: Robustness results
            leakage_results: Leakage check results

        Returns:
            List of recommendations
        """
        recommendations = []

        metrics = backtest_results.get("metrics", {})

        # Recommendations based on performance
        if metrics.get("sharpe_ratio", 0) < 0.5:
            recommendations.append("Consider improving risk-adjusted returns by refining entry/exit rules")
        
        if abs(metrics.get("max_drawdown", 0)) > 0.25:
            recommendations.append("Implement stricter risk controls to limit maximum drawdown")
        
        if metrics.get("win_rate", 0) < 0.4:
            recommendations.append("Focus on improving trade success rate rather than just profit magnitude")
        
        if metrics.get("turnover", 0) > 5:
            recommendations.append("Reduce trading frequency to minimize transaction costs")

        # Recommendations based on robustness
        if robustness_results:
            if "walk_forward_analysis" in robustness_results:
                wf_results = robustness_results["walk_forward_analysis"]
                if "aggregated" in wf_results and "consistency_score" in wf_results["aggregated"]:
                    if wf_results["aggregated"]["consistency_score"] < 0.5:
                        recommendations.append("Strategy performance varies significantly across time periods - consider adaptive parameters")

            if "sensitivity_analysis" in robustness_results:
                sens_results = robustness_results["sensitivity_analysis"]
                if "single_parameter" in sens_results:
                    sens_data = sens_results["single_parameter"]
                    for param, data in sens_data.items():
                        if abs(data.get("sensitivity", 0)) > 1.0:  # Very sensitive
                            recommendations.append(f"Parameter '{param}' is highly sensitive - consider fixing to reduce overfitting")

        # Recommendations based on leakage
        if leakage_results:
            if not leakage_results.get("summary", {}).get("passed", True):
                recommendations.append("Address future leakage issues before deploying strategy")
                recommendations.append("Review feature engineering to ensure no future information is used")

        # General recommendation if no other issues
        if not recommendations:
            recommendations.append("Strategy appears robust. Consider additional out-of-sample testing before deployment")

        return recommendations
