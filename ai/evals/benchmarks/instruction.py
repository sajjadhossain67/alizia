"""
ALIZIA AI - Instruction Following Benchmark Suite (IFEval)
Verifies multi-constraint formatting, exact word bounds, forbidden tokens,
and system prompt adherence.
"""

from __future__ import annotations
from typing import List, Optional
from ai.evals.benchmarks.base import BaseBenchmarkSuite
from ai.evals.types import BenchmarkTask, EvalPrediction, TaskScore, BenchmarkCategory
from ai.evals.scorers.deterministic import IFEvalScorer


class InstructionFollowingSuite(BaseBenchmarkSuite):
    """Evaluates strict multi-constraint adherence and formatting compliance"""

    def __init__(self, name: str = "instruction_following"):
        super().__init__(name=name, category=BenchmarkCategory.INSTRUCTION)

    def load_dataset(self, limit: Optional[int] = None) -> List[BenchmarkTask]:
        raw_items = [
            # Strict JSON with exactly 3 keys and no markdown
            {
                "id": "ifeval_json_01",
                "suite": self.name,
                "category": self.category,
                "prompt": (
                    "Provide a summary of Redis replication. "
                    "Your response MUST be a valid JSON object containing exactly three keys: "
                    "'concept', 'primary_role', and 'replica_role'. Do not include any explanation outside the JSON."
                ),
                "metadata": {
                    "constraints": {
                        "require_valid_json": True
                    }
                }
            },
            # Length bounds + forbidden words
            {
                "id": "ifeval_bounds_01",
                "suite": self.name,
                "category": self.category,
                "prompt": (
                    "Explain what an operating system kernel is. "
                    "Constraint 1: The response must be between 30 and 60 words in total. "
                    "Constraint 2: Do NOT use the words 'hardware', 'software', or 'bridge'."
                ),
                "metadata": {
                    "constraints": {
                        "min_words": 30,
                        "max_words": 60,
                        "forbidden_words": ["hardware", "software", "bridge"]
                    }
                }
            },
            # Required starting phrase and section headers
            {
                "id": "ifeval_sections_01",
                "suite": self.name,
                "category": self.category,
                "prompt": (
                    "Write an architectural note on microservices communication. "
                    "Constraint 1: Must begin with exactly: 'Architecture Note: Synchronous vs Asynchronous'. "
                    "Constraint 2: Must contain the two sections: 'Synchronous REST/gRPC' and 'Event-Driven Messaging'."
                ),
                "metadata": {
                    "constraints": {
                        "start_with": "Architecture Note: Synchronous vs Asynchronous",
                        "required_sections": ["Synchronous REST/gRPC", "Event-Driven Messaging"]
                    }
                }
            }
        ]

        tasks = [BenchmarkTask(**item) for item in raw_items]
        return tasks[:limit] if limit else tasks

    async def score_prediction(self, task: BenchmarkTask, pred: EvalPrediction) -> TaskScore:
        return IFEvalScorer.score(task, pred)
