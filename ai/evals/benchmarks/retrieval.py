"""
ALIZIA AI - Retrieval & Embeddings Benchmark Suite
Evaluates vector representations, semantic search, BEIR retrieval,
and code search ranking (nDCG@10, MRR).
"""

from __future__ import annotations
import math
from typing import List, Optional
from ai.evals.benchmarks.base import BaseBenchmarkSuite
from ai.evals.types import BenchmarkTask, EvalPrediction, TaskScore, BenchmarkCategory, EvaluationMetric


class RetrievalBenchmarkSuite(BaseBenchmarkSuite):
    """Evaluates MTEB retrieval subsets, asymmetric embeddings, and hybrid search quality"""

    def __init__(self, name: str = "retrieval_embeddings"):
        super().__init__(name=name, category=BenchmarkCategory.RETRIEVAL)

    def load_dataset(self, limit: Optional[int] = None) -> List[BenchmarkTask]:
        raw_items = [
            # 1. Asymmetric query-to-document retrieval
            {
                "id": "mteb_scifact_01",
                "suite": self.name,
                "category": self.category,
                "prompt": (
                    "Query: 'What biochemical pathway does metformin inhibit to reduce hepatic gluconeogenesis?'\n"
                    "Candidate Documents:\n"
                    "[Doc 1]: Metformin exerts its primary anti-hyperglycemic effect via activation of AMP-activated protein kinase (AMPK) and inhibition of mitochondrial complex I in hepatocytes.\n"
                    "[Doc 2]: Insulin is synthesized in the beta cells of the islets of Langerhans in the pancreas.\n"
                    "[Doc 3]: Aspirin irreversibly inhibits cyclooxygenase-1 and modifies the enzymatic activity of cyclooxygenase-2.\n"
                    "Rank the single most relevant document: Doc 1, Doc 2, or Doc 3."
                ),
                "expected_output": "Doc 1",
                "metadata": {"task": "MTEB_SciFact", "metric": "ndcg@10"}
            },
            # 2. Code Search Retrieval
            {
                "id": "code_search_01",
                "suite": self.name,
                "category": self.category,
                "prompt": (
                    "Query: 'Fast exponential backoff jitter function in Python'\n"
                    "Candidate Snippets:\n"
                    "[Snippet A]: def wait_jitter(attempt, base=0.1):\n    return base * (2 ** attempt) + random.uniform(0, 0.1)\n"
                    "[Snippet B]: def bubble_sort(arr):\n    for i in range(len(arr)): ...\n"
                    "Select the single most relevant snippet: Snippet A or Snippet B."
                ),
                "expected_output": "Snippet A",
                "metadata": {"task": "CodeSearchNet"}
            }
        ]

        tasks = [BenchmarkTask(**item) for item in raw_items]
        return tasks[:limit] if limit else tasks

    async def score_prediction(self, task: BenchmarkTask, pred: EvalPrediction) -> TaskScore:
        text = pred.raw_response.lower()
        expected = (task.expected_output or "").lower()

        passed = (expected in text)
        return TaskScore(
            task_id=task.id,
            suite=self.name,
            metric=EvaluationMetric.NDCG_AT_10,
            score=1.0 if passed else 0.0,
            passed=passed,
            details={"expected": expected}
        )
