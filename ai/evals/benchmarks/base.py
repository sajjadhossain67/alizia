"""
ALIZIA AI - Base Benchmark Suite Specification
Abstract framework for dataset loading, parallel task execution, scoring,
and performance metric collection (latency, tokens, cost).
"""

from __future__ import annotations
import abc
import time
from typing import List, Dict, Any, Optional
from ai.evals.types import BenchmarkTask, EvalPrediction, TaskScore, SuiteResult, BenchmarkCategory, EvaluationMetric


class BaseBenchmarkSuite(abc.ABC):
    """Abstract Base Class for every benchmark suite in Alizia AI"""

    def __init__(self, name: str, category: BenchmarkCategory):
        self.name = name
        self.category = category
        self.tasks: List[BenchmarkTask] = []

    @abc.abstractmethod
    def load_dataset(self, limit: Optional[int] = None) -> List[BenchmarkTask]:
        """Loads and returns benchmark tasks"""
        pass

    @abc.abstractmethod
    async def score_prediction(self, task: BenchmarkTask, pred: EvalPrediction) -> TaskScore:
        """Scores a single model prediction against the task"""
        pass

    async def run_suite(
        self,
        model_name: str,
        predict_fn,
        limit: Optional[int] = None
    ) -> SuiteResult:
        """Runs the entire benchmark suite against a model and returns aggregate metrics"""
        tasks = self.load_dataset(limit=limit)
        task_scores: List[TaskScore] = []
        latencies: List[float] = []
        total_cost = 0.0

        for task in tasks:
            start_t = time.time()
            try:
                # Call model through predict_fn
                pred = await predict_fn(task, model_name)
                latency_ms = (time.time() - start_t) * 1000.0
                pred.latency_ms = latency_ms
                latencies.append(latency_ms)
                total_cost += pred.estimated_cost_usd

                # Score prediction
                score = await self.score_prediction(task, pred)
                task_scores.append(score)
            except Exception as e:
                # Graceful handling of individual task failure
                latency_ms = (time.time() - start_t) * 1000.0
                latencies.append(latency_ms)
                task_scores.append(
                    TaskScore(
                        task_id=task.id,
                        suite=self.name,
                        metric=task_scores[0].metric if task_scores else EvaluationMetric.ACCURACY,
                        score=0.0,
                        passed=False,
                        details={"error": str(e)}
                    )
                )

        total_tasks = len(task_scores)
        passed_tasks = sum(1 for s in task_scores if s.passed)
        avg_score = (sum(s.score for s in task_scores) / total_tasks * 100.0) if total_tasks > 0 else 0.0

        latencies.sort()
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        p95_idx = int(0.95 * len(latencies))
        p95_latency = latencies[min(p95_idx, len(latencies) - 1)] if latencies else 0.0

        return SuiteResult(
            suite=self.name,
            category=self.category,
            total_tasks=total_tasks,
            passed_tasks=passed_tasks,
            score=round(avg_score, 2),
            avg_latency_ms=round(avg_latency, 2),
            p95_latency_ms=round(p95_latency, 2),
            total_cost_usd=round(total_cost, 5),
            scores=task_scores
        )
