"""
ALIZIA AI - Mathematics Benchmark Suite
Implements GSM8K, MATH-500, AIME competition problems, and formal proof verification.
"""

from __future__ import annotations
from typing import List, Optional
from ai.evals.benchmarks.base import BaseBenchmarkSuite
from ai.evals.types import BenchmarkTask, EvalPrediction, TaskScore, BenchmarkCategory
from ai.evals.scorers.deterministic import MathToleranceScorer


class MathBenchmarkSuite(BaseBenchmarkSuite):
    """Evaluates multi-step mathematical derivation, competition algebra, and discrete math"""

    def __init__(self, name: str = "math"):
        super().__init__(name=name, category=BenchmarkCategory.MATH)

    def load_dataset(self, limit: Optional[int] = None) -> List[BenchmarkTask]:
        raw_items = [
            # GSM8K: Multi-step arithmetic reasoning
            {
                "id": "gsm8k_01",
                "suite": self.name,
                "category": self.category,
                "prompt": (
                    "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins "
                    "for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per egg. "
                    "How much in dollars does she make every day at the farmers' market?\n"
                    "Provide the reasoning and state the final answer in \\boxed{}."
                ),
                "expected_output": "18",
                "metadata": {"source": "GSM8K", "difficulty": "Grade School"}
            },
            # MATH-500: High-school competition algebra
            {
                "id": "math_500_algebra_01",
                "suite": self.name,
                "category": self.category,
                "prompt": (
                    "Find the positive value of x such that 2^(2x) - 10 * 2^x + 16 = 0.\n"
                    "Put your final answer in \\boxed{}."
                ),
                "expected_output": "3",
                "metadata": {"source": "MATH-500", "topic": "Algebra"}
            },
            # AIME: Competition Discrete Math / Number Theory
            {
                "id": "aime_2024_01",
                "suite": self.name,
                "category": self.category,
                "prompt": (
                    "Find the number of positive integers n <= 1000 such that gcd(n, 36) = 1 and gcd(n, 25) = 1.\n"
                    "Calculate step by step and present your final integer answer in \\boxed{}."
                ),
                "expected_output": "267",
                "metadata": {"source": "AIME", "topic": "Number Theory / Inclusion-Exclusion"}
            },
            # Probability / Combinatorics
            {
                "id": "math_combinatorics_01",
                "suite": self.name,
                "category": self.category,
                "prompt": (
                    "A fair 6-sided die is rolled 4 times. What is the probability that the numbers rolled are strictly increasing? "
                    "Express your answer as a simplified fraction or decimal. Put the final numerical result in \\boxed{}."
                ),
                "expected_output": "0.011574",
                "metadata": {"source": "MATH-500", "topic": "Combinatorics"}
            }
        ]

        tasks = [BenchmarkTask(**item) for item in raw_items]
        return tasks[:limit] if limit else tasks

    async def score_prediction(self, task: BenchmarkTask, pred: EvalPrediction) -> TaskScore:
        return MathToleranceScorer.score(task, pred)
