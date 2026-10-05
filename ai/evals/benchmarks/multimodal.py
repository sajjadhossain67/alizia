"""
ALIZIA AI - Multimodal Benchmark Suite
Evaluates visual understanding across ChartQA, DocVQA, UI-to-Code,
diagram reasoning, and spatial geometry.
"""

from __future__ import annotations
from typing import List, Optional
from ai.evals.benchmarks.base import BaseBenchmarkSuite
from ai.evals.types import BenchmarkTask, EvalPrediction, TaskScore, BenchmarkCategory, EvaluationMetric


class MultimodalBenchmarkSuite(BaseBenchmarkSuite):
    """Evaluates Chart comprehension, OCR document understanding, and UI reverse engineering"""

    def __init__(self, name: str = "multimodal"):
        super().__init__(name=name, category=BenchmarkCategory.MULTIMODAL)

    def load_dataset(self, limit: Optional[int] = None) -> List[BenchmarkTask]:
        raw_items = [
            # 1. ChartQA: Numerical data extraction from chart description/matrix
            {
                "id": "chartqa_rev_01",
                "suite": self.name,
                "category": self.category,
                "prompt": (
                    "[IMAGE: Sales bar chart showing Q1=$4.2M, Q2=$5.1M, Q3=$6.8M, Q4=$8.4M]\n"
                    "Question: By what percentage did revenue grow from Q1 to Q4? Calculate and state the percentage."
                ),
                "expected_output": "100%",
                "metadata": {"task": "ChartQA", "modality": "chart"}
            },
            # 2. DocVQA: Document OCR & form extraction
            {
                "id": "docvqa_invoice_01",
                "suite": self.name,
                "category": self.category,
                "prompt": (
                    "[IMAGE: Invoice #9021 containing Subtotal=$1,200, Tax (10%)=$120, Total Due=$1,320]\n"
                    "Question: What is the total due on Invoice #9021?"
                ),
                "expected_output": "$1,320",
                "metadata": {"task": "DocVQA", "modality": "document"}
            },
            # 3. UI-to-Code: Generating accessible HTML/CSS component
            {
                "id": "ui_to_code_01",
                "suite": self.name,
                "category": self.category,
                "prompt": (
                    "[IMAGE: Clean floating capsule pill button with an upload icon and label 'Submit Evidence']\n"
                    "Generate responsive, semantic HTML/CSS for this button with accessible aria attributes. "
                    "Include ```html ... ```"
                ),
                "expected_output": "<button",
                "metadata": {"task": "UI-to-Code", "modality": "ui"}
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
            metric=EvaluationMetric.ACCURACY,
            score=1.0 if passed else 0.0,
            passed=passed,
            details={"task": task.metadata.get("task"), "expected": expected}
        )
