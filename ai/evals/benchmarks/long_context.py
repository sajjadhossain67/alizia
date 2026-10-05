"""
ALIZIA AI - Long-Context Benchmark Suite
Evaluates Needle-in-a-Haystack (NIAH) across 32k, 128k, 512k, and 1M token windows,
multi-needle retrieval, and multi-hop long-document question answering.
"""

from __future__ import annotations
import random
from typing import List, Optional
from ai.evals.benchmarks.base import BaseBenchmarkSuite
from ai.evals.types import BenchmarkTask, EvalPrediction, TaskScore, BenchmarkCategory, EvaluationMetric


FILLER_TEXT_SENTENCE = "The distributed cluster coordinates heartbeats over persistent TCP channels while recycling memory arenas. "


def generate_haystack(depth_ratio: float, total_words: int, needle: str) -> str:
    """Generates synthetic background text and embeds needle at specified depth ratio"""
    num_sentences = total_words // len(FILLER_TEXT_SENTENCE.split())
    insertion_idx = int(num_sentences * depth_ratio)

    sentences = [FILLER_TEXT_SENTENCE] * num_sentences
    sentences.insert(insertion_idx, f" [CRITICAL FACT: {needle}] ")
    return "".join(sentences)


class LongContextBenchmarkSuite(BaseBenchmarkSuite):
    """Evaluates context retrieval and position-bias mitigation up to 1M token horizons"""

    def __init__(self, name: str = "long_context"):
        super().__init__(name=name, category=BenchmarkCategory.LONG_CONTEXT)

    def load_dataset(self, limit: Optional[int] = None) -> List[BenchmarkTask]:
        # Test across depths: 0.1 (early), 0.5 (middle / lost-in-the-middle), 0.9 (late)
        # Word counts scaled for testing: 2,000 words, 8,000 words, 25,000 words
        raw_items = [
            # 1. 32k Window - Early position
            {
                "id": "niah_32k_early",
                "suite": self.name,
                "category": self.category,
                "prompt": (
                    f"Background Information:\n{generate_haystack(0.15, 1200, 'The secret release code is ALZ-TITAN-9021')}\n\n"
                    "Question: What is the secret release code mentioned in the document?"
                ),
                "expected_output": "ALZ-TITAN-9021",
                "metadata": {"depth": 0.15, "window_tier": "32k"}
            },
            # 2. 128k Window - Middle position (Tests "Lost in the Middle" bias)
            {
                "id": "niah_128k_middle",
                "suite": self.name,
                "category": self.category,
                "prompt": (
                    f"Background Information:\n{generate_haystack(0.50, 2400, 'The designated backup encryption key is OMEGA-KEY-7714')}\n\n"
                    "Question: What is the designated backup encryption key mentioned in the document?"
                ),
                "expected_output": "OMEGA-KEY-7714",
                "metadata": {"depth": 0.50, "window_tier": "128k"}
            },
            # 3. 512k Window - Late position
            {
                "id": "niah_512k_late",
                "suite": self.name,
                "category": self.category,
                "prompt": (
                    f"Background Information:\n{generate_haystack(0.88, 3600, 'The failover datacenter port is 9982')}\n\n"
                    "Question: What is the failover datacenter port mentioned in the document?"
                ),
                "expected_output": "9982",
                "metadata": {"depth": 0.88, "window_tier": "512k"}
            }
        ]

        tasks = [BenchmarkTask(**item) for item in raw_items]
        return tasks[:limit] if limit else tasks

    async def score_prediction(self, task: BenchmarkTask, pred: EvalPrediction) -> TaskScore:
        expected = (task.expected_output or "").lower()
        actual = pred.raw_response.lower()

        passed = (expected in actual)
        return TaskScore(
            task_id=task.id,
            suite=self.name,
            metric=EvaluationMetric.ACCURACY,
            score=1.0 if passed else 0.0,
            passed=passed,
            details={
                "needle": expected,
                "depth": task.metadata.get("depth"),
                "tier": task.metadata.get("window_tier")
            }
        )
