"""
ALIZIA AI - Master Evaluation Runner & CI Regression Gate
Orchestrates parallel suite execution, telemetry collection,
head-to-head baseline comparisons, and regression gate checks.
"""

from __future__ import annotations
import asyncio
import datetime
import os
from typing import Dict, List, Optional, Tuple

from ai.evals.types import (
    FullEvalReport,
    SuiteResult,
    BenchmarkTask,
    EvalPrediction,
    PairwiseComparison
)
from ai.evals.adapters.model_adapter import AliziaModelAdapter
from ai.evals.adapters.gemini_adapter import GeminiBenchmarkAdapter
from ai.evals.scorers.llm_judge import LLMJudgeScorer
from ai.evals.reporter import EvalReportGenerator

from ai.evals.benchmarks.knowledge import KnowledgeBenchmarkSuite
from ai.evals.benchmarks.math import MathBenchmarkSuite
from ai.evals.benchmarks.code import CodeBenchmarkSuite
from ai.evals.benchmarks.instruction import InstructionFollowingSuite
from ai.evals.benchmarks.safety import SafetyBenchmarkSuite
from ai.evals.benchmarks.multilingual import MultilingualBenchmarkSuite
from ai.evals.benchmarks.factuality import FactualityBenchmarkSuite
from ai.evals.benchmarks.tool_use import ToolUseBenchmarkSuite
from ai.evals.benchmarks.long_context import LongContextBenchmarkSuite
from ai.evals.benchmarks.multimodal import MultimodalBenchmarkSuite
from ai.evals.benchmarks.retrieval import RetrievalBenchmarkSuite
from ai.evals.benchmarks.writing import WritingQualitySuite


class EvaluationRunner:
    """Master controller for all benchmark suites and CI gating"""

    def __init__(self):
        self.suites = {
            "knowledge_reasoning": KnowledgeBenchmarkSuite(),
            "math": MathBenchmarkSuite(),
            "code": CodeBenchmarkSuite(),
            "instruction_following": InstructionFollowingSuite(),
            "safety": SafetyBenchmarkSuite(),
            "multilingual": MultilingualBenchmarkSuite(),
            "factuality": FactualityBenchmarkSuite(),
            "tool_use": ToolUseBenchmarkSuite(),
            "long_context": LongContextBenchmarkSuite(),
            "multimodal": MultimodalBenchmarkSuite(),
            "retrieval_embeddings": RetrievalBenchmarkSuite(),
            "writing_quality": WritingQualitySuite(),
        }
        self.model_adapter = AliziaModelAdapter()
        self.gemini_adapter = GeminiBenchmarkAdapter()
        self.judge = LLMJudgeScorer()

    async def run_evaluation(
        self,
        model_name: str = "alizia-nova",
        suite_names: Optional[List[str]] = None,
        compare_with_gemini: bool = False,
        limit_per_suite: Optional[int] = None,
        baseline_report_to_check: Optional[FullEvalReport] = None,
        save_report_path: Optional[str] = None
    ) -> FullEvalReport:
        """Executes selected or all suites and evaluates performance"""
        selected_keys = suite_names or list(self.suites.keys())
        suite_results: Dict[str, SuiteResult] = {}
        pairwise_comps: List[PairwiseComparison] = []

        # Predict function for candidate model
        async def predict_candidate(task: BenchmarkTask, m_name: str) -> EvalPrediction:
            return await self.model_adapter.predict(task, model_name=m_name)

        # Run each suite
        for key in selected_keys:
            if key not in self.suites:
                continue
            suite = self.suites[key]
            res = await suite.run_suite(
                model_name=model_name,
                predict_fn=predict_candidate,
                limit=limit_per_suite
            )
            suite_results[key] = res

        # Run head-to-head comparison against Gemini if requested
        if compare_with_gemini:
            for key in selected_keys:
                suite = self.suites[key]
                tasks = suite.load_dataset(limit=limit_per_suite)

                for task in tasks:
                    try:
                        pred_alizia = await self.model_adapter.predict(task, model_name=model_name)
                        pred_gemini = await self.gemini_adapter.predict(task)

                        async def judge_call(prompt: str, m: str) -> str:
                            # Use internal provider for judge evaluation
                            judge_pred = await self.model_adapter.predict(
                                BenchmarkTask(
                                    suite="judge",
                                    category=task.category,
                                    prompt=prompt
                                ),
                                model_name="alizia-nova"
                            )
                            return judge_pred.raw_response

                        comp = await self.judge.evaluate_pairwise_debiased(
                            task=task,
                            pred_a=pred_alizia,
                            pred_b=pred_gemini,
                            rubric={"category": task.category.value},
                            judge_call_fn=judge_call
                        )
                        pairwise_comps.append(comp)
                    except Exception as e:
                        # Log and continue
                        pass

        # Calculate overall score
        total_scores = [s.score for s in suite_results.values()]
        overall_score = sum(total_scores) / len(total_scores) if total_scores else 0.0

        # Calculate win-rate vs Gemini
        overall_win_rate = None
        if pairwise_comps:
            wins = sum(1.0 for c in pairwise_comps if c.winner == "model_a")
            ties = sum(0.5 for c in pairwise_comps if c.winner == "tie")
            overall_win_rate = ((wins + ties) / len(pairwise_comps)) * 100.0

        # CI Regression Gate Analysis
        passed_ci = True
        violations = []

        if baseline_report_to_check:
            for s_name, current_s in suite_results.items():
                if s_name in baseline_report_to_check.suites:
                    base_s = baseline_report_to_check.suites[s_name]

                    # 1. Quality score drop > 1.0 point
                    score_drop = base_s.score - current_s.score
                    if score_drop > 1.0:
                        passed_ci = False
                        violations.append(
                            f"Quality regression in suite '{s_name}': dropped by {score_drop:.1f} points "
                            f"(baseline: {base_s.score:.1f}%, current: {current_s.score:.1f}%)"
                        )

                    # 2. Latency regression > 5%
                    if base_s.avg_latency_ms > 0:
                        lat_increase = (current_s.avg_latency_ms - base_s.avg_latency_ms) / base_s.avg_latency_ms
                        if lat_increase > 0.05:
                            passed_ci = False
                            violations.append(
                                f"Latency regression in suite '{s_name}': increased by {lat_increase * 100:.1f}% "
                                f"(baseline: {base_s.avg_latency_ms:.1f}ms, current: {current_s.avg_latency_ms:.1f}ms)"
                            )

        # Build full report
        report = FullEvalReport(
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            model_name=model_name,
            baseline_model="gemini-flash-latest" if compare_with_gemini else None,
            overall_score=round(overall_score, 2),
            overall_win_rate_vs_baseline=round(overall_win_rate, 2) if overall_win_rate is not None else None,
            suites=suite_results,
            passed_ci_gate=passed_ci,
            regression_violations=violations
        )

        if save_report_path:
            EvalReportGenerator.generate_markdown_report(
                report=report,
                pairwise_comparisons=pairwise_comps,
                output_path=save_report_path
            )

        return report
