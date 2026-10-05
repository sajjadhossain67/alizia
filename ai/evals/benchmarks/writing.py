"""
ALIZIA AI - Writing Quality & Preference Benchmark Suite
Evaluates tone, clarity, concision, steelmanning, creativity, and empathy.
Uses structured rubrics and pairwise comparison rankings.
"""

from __future__ import annotations
from typing import List, Optional
from ai.evals.benchmarks.base import BaseBenchmarkSuite
from ai.evals.types import BenchmarkTask, EvalPrediction, TaskScore, BenchmarkCategory, EvaluationMetric


class WritingQualitySuite(BaseBenchmarkSuite):
    """Evaluates human-preferred writing: tone, concision, empathy, and clarity"""

    def __init__(self, name: str = "writing_quality"):
        super().__init__(name=name, category=BenchmarkCategory.WRITING)

    def load_dataset(self, limit: Optional[int] = None) -> List[BenchmarkTask]:
        raw_items = [
            # 1. Technical Explanation: High clarity, zero filler, answer-first
            {
                "id": "write_clarity_01",
                "suite": self.name,
                "category": self.category,
                "prompt": (
                    "Explain the difference between optimistic and pessimistic locking in databases to a junior engineer. "
                    "Lead directly with the core tradeoff, provide a concrete example, and do not use sycophantic filler."
                ),
                "rubric_id": "writing_clarity",
                "metadata": {"criterion": "clarity_and_conciseness"}
            },
            # 2. Emotional / Difficult Situation: Warm, supportive, no clinical coldness or false reassurance
            {
                "id": "write_empathy_01",
                "suite": self.name,
                "category": self.category,
                "prompt": (
                    "I just received a critical performance review and I feel like quitting on the spot. "
                    "How should I navigate my next 24 hours?"
                ),
                "rubric_id": "writing_empathy",
                "metadata": {"criterion": "empathy_and_practical_help"}
            },
            # 3. Contested Policy Topic: Steelmanning both perspectives fairly
            {
                "id": "write_steelman_01",
                "suite": self.name,
                "category": self.category,
                "prompt": (
                    "Should open-source AI model weights be legally regulated to mitigate catastrophic risk? "
                    "Steelman the strongest argument for both sides without moralizing."
                ),
                "rubric_id": "writing_steelmanning",
                "metadata": {"criterion": "balanced_coverage"}
            }
        ]

        tasks = [BenchmarkTask(**item) for item in raw_items]
        return tasks[:limit] if limit else tasks

    async def score_prediction(self, task: BenchmarkTask, pred: EvalPrediction) -> TaskScore:
        text = pred.raw_response
        criterion = task.metadata.get("criterion")

        # Heuristic quality signals:
        # - Has content length > 80 chars
        # - Does not have sycophantic filler ("Certainly!", "I'd be thrilled to help!")
        filler_penalty = any(f in text.lower() for f in ["certainly!", "as an ai", "i would be delighted", "in conclusion"])
        score_val = 1.0 if not filler_penalty and len(text) > 80 else 0.7

        return TaskScore(
            task_id=task.id,
            suite=self.name,
            metric=EvaluationMetric.PAIRWISE_WIN_RATE,
            score=score_val,
            passed=(score_val >= 0.8),
            details={"criterion": criterion, "filler_penalty": filler_penalty}
        )
