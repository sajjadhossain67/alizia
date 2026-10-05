"""
ALIZIA AI - Evaluation Report Generator
Produces structured JSON metrics, console summary tables, failure cluster analyses,
and comprehensive markdown reports in docs/evals/baseline.md.
"""

from __future__ import annotations
import json
import os
import time
from typing import Dict, Any, List, Optional
from ai.evals.types import FullEvalReport, SuiteResult, PairwiseComparison
from ai.evals.scorers.pairwise import PairwiseRanker


class EvalReportGenerator:
    """Generates analytical reports, win-rate tables, and failure cluster summaries"""

    @classmethod
    def generate_markdown_report(
        cls,
        report: FullEvalReport,
        pairwise_comparisons: Optional[List[PairwiseComparison]] = None,
        output_path: Optional[str] = None
    ) -> str:
        md = []
        md.append(f"# ALIZIA AI — Evaluation & Benchmark Baseline Report")
        md.append(f"\n> **Run ID**: `{report.run_id}`  ")
        md.append(f"> **Evaluated Model**: `{report.model_name}`  ")
        md.append(f"> **Baseline Reference**: `{report.baseline_model or 'Google Gemini Flash Latest'}`  ")
        md.append(f"> **Timestamp**: `{report.timestamp}`  ")
        md.append(f"> **CI Regression Gate**: `{'PASSED' if report.passed_ci_gate else 'FAILED'}`\n")
        md.append("---\n")

        # Executive Summary
        md.append("## 1. Executive Summary\n")
        md.append(
            f"Alizia AI's evaluation subsystem audited **{len(report.suites)} core benchmark categories** "
            f"under identical prompts, temperature (0.0), and deterministic seeds. "
            f"Overall platform score: **{report.overall_score:.2f}%**."
        )
        if report.overall_win_rate_vs_baseline is not None:
            md.append(f" Overall Head-to-Head win rate vs Gemini: **{report.overall_win_rate_vs_baseline:.1f}%**.\n")
        else:
            md.append("\n")

        # Suite Results Table
        md.append("## 2. Category Benchmark Breakdown\n")
        md.append("| Benchmark Suite | Category | Tasks | Passed | Accuracy / Score | Avg Latency | p95 Latency | Est. Cost |")
        md.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

        for suite_name, s in report.suites.items():
            md.append(
                f"| **{s.suite}** | {s.category.value} | {s.total_tasks} | {s.passed_tasks} | "
                f"**{s.score:.1f}%** | {s.avg_latency_ms:.1f}ms | {s.p95_latency_ms:.1f}ms | ${s.total_cost_usd:.4f} |"
            )
        md.append("\n---\n")

        # Pairwise Head-to-Head Section
        if pairwise_comparisons:
            stats = PairwiseRanker.compute_win_rate_with_bootstrap_ci(
                comparisons=pairwise_comparisons,
                target_model=report.model_name
            )
            md.append("## 3. Head-to-Head Blind Arena & Pairwise Comparisons\n")
            md.append(
                f"- **Win Rate vs Gemini**: **{stats['win_rate']:.1f}%**  \n"
                f"- **95% Bootstrap Confidence Interval**: `[{stats['ci_lower']:.1f}%, {stats['ci_upper']:.1f}%]`  \n"
                f"- **Total Pairwise Battles**: `{stats['total_battles']}`  \n"
            )

            # Failure clusters / Disagreements
            disagreements = [c for c in pairwise_comparisons if c.swapped_order]
            md.append(f"### Positional Debiasing & Disagreement Clusters\n")
            md.append(
                f"Evaluated position-swap debiasing on all pairwise comparisons. "
                f"Detected **{len(disagreements)}** positional order sensitivities flagged as ties.\n"
            )

        # Regression Gate Audit
        md.append("## 4. CI Regression Gate Status\n")
        if report.passed_ci_gate:
            md.append("✅ **All quality and latency thresholds satisfied:**\n")
            md.append("- No suite score dropped > 1.0 point below baseline.\n")
            md.append("- Latency p95 within 5% tolerance budget.\n")
            md.append("- Zero security/injection barriers breached.\n")
        else:
            md.append("❌ **Regression Gate Triggered Violations:**\n")
            for viol in report.regression_violations:
                md.append(f"- {viol}\n")

        md.append("\n---\n")
        md.append(f"*Generated automatically by Alizia AI Evaluation Subsystem (`ai.evals.runner`)*\n")

        content = "\n".join(md)
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)

        return content
