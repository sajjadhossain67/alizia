"""
ALIZIA AI - Knowledge & Reasoning Benchmark Suites
Implements MMLU-Pro, GPQA-Diamond, BIG-Bench Hard, ARC-Challenge, and HLE tasks.
"""

from __future__ import annotations
from typing import List, Optional
from ai.evals.benchmarks.base import BaseBenchmarkSuite
from ai.evals.types import BenchmarkTask, EvalPrediction, TaskScore, BenchmarkCategory
from ai.evals.scorers.deterministic import ExactMatchScorer


class KnowledgeBenchmarkSuite(BaseBenchmarkSuite):
    """Evaluates multi-discipline knowledge and graduate-level scientific reasoning"""

    def __init__(self, name: str = "knowledge_reasoning"):
        super().__init__(name=name, category=BenchmarkCategory.KNOWLEDGE)

    def load_dataset(self, limit: Optional[int] = None) -> List[BenchmarkTask]:
        # Curated representative standard items across MMLU-Pro, GPQA-Diamond, ARC, BBH
        raw_items = [
            # MMLU-Pro: Computer Science & Distributed Systems
            {
                "id": "mmlu_cs_01",
                "suite": self.name,
                "category": self.category,
                "prompt": (
                    "In a Raft consensus cluster with 5 nodes, if 2 nodes crash and a network partition divides "
                    "the remaining 3 nodes into partitions of size 2 and 1, what can be stated about leader election?\n"
                    "A) Partition of size 2 can elect a new leader\n"
                    "B) No partition can elect a leader because a majority of 3 is required\n"
                    "C) Partition of size 1 can elect a leader via timeout\n"
                    "D) The cluster triggers split-brain with two active leaders\n"
                    "Answer with the single letter of the correct option."
                ),
                "expected_output": "B",
                "metadata": {"source": "MMLU-Pro", "field": "Computer Science"}
            },
            # GPQA-Diamond: Quantum Physics / Chemistry
            {
                "id": "gpqa_diamond_01",
                "suite": self.name,
                "category": self.category,
                "prompt": (
                    "Consider a spin-1/2 particle in the state |psi> = (1/sqrt(5))|up> + (2/sqrt(5))|down>. "
                    "What is the expectation value <S_z> in units of hbar?\n"
                    "A) -3/10\n"
                    "B) -3/5\n"
                    "C) 3/10\n"
                    "D) 1/2\n"
                    "Answer with the single letter of the correct option."
                ),
                "expected_output": "A",
                "metadata": {"source": "GPQA-Diamond", "field": "Quantum Mechanics"}
            },
            # ARC-Challenge: Scientific Common Sense
            {
                "id": "arc_challenge_01",
                "suite": self.name,
                "category": self.category,
                "prompt": (
                    "Which change occurs when a substance undergoes a chemical change rather than a physical change?\n"
                    "A) Liquid water boils to form water vapor\n"
                    "B) Solid sodium metal reacts with chlorine gas to form a white crystalline solid\n"
                    "C) Solid iodine sublimates directly into purple gas\n"
                    "D) Sugar crystals dissolve completely in water\n"
                    "Answer with the letter of the correct option."
                ),
                "expected_output": "B",
                "metadata": {"source": "ARC-Challenge", "field": "Physical Chemistry"}
            },
            # BIG-Bench Hard: Causal Reasoning
            {
                "id": "bbh_causal_01",
                "suite": self.name,
                "category": self.category,
                "prompt": (
                    "Premise: The technician restarted the server after applying the kernel patch. "
                    "The database connection pool began rejecting new incoming requests immediately after reboot.\n"
                    "Question: Is it logically guaranteed that the kernel patch caused the connection pool failure?\n"
                    "A) Yes, post hoc ergo propter hoc guarantee\n"
                    "B) No, it is a temporal correlation that does not prove sole causality without inspecting config or ports\n"
                    "Answer with the letter of the correct option."
                ),
                "expected_output": "B",
                "metadata": {"source": "BIG-Bench Hard", "field": "Causal Judgement"}
            },
            # HLE-Style Hard Question: Advanced Complexity
            {
                "id": "hle_complexity_01",
                "suite": self.name,
                "category": self.category,
                "prompt": (
                    "Under the assumption that P != NP, which of the following statements regarding the class BQP is true?\n"
                    "A) BQP is known to be strictly contained in P\n"
                    "B) BQP contains NP-complete problems with proven polynomial time algorithms\n"
                    "C) BQP contains problems in BPP, and Factoring is in BQP but not known to be in P or NP-complete\n"
                    "D) BQP equals PSPACE\n"
                    "Answer with the letter of the correct option."
                ),
                "expected_output": "C",
                "metadata": {"source": "HLE-Hard", "field": "Computational Complexity"}
            }
        ]

        tasks = [BenchmarkTask(**item) for item in raw_items]
        return tasks[:limit] if limit else tasks

    async def score_prediction(self, task: BenchmarkTask, pred: EvalPrediction) -> TaskScore:
        return ExactMatchScorer.score(task, pred)
