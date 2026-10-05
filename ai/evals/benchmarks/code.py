"""
ALIZIA AI - Code Intelligence Benchmark Suite
Implements HumanEval+, MBPP+, LiveCodeBench, and structured diff/patch verification.
All code generation solutions are executed and validated in the isolated Python sandbox.
"""

from __future__ import annotations
from typing import List, Optional
from ai.evals.benchmarks.base import BaseBenchmarkSuite
from ai.evals.types import BenchmarkTask, EvalPrediction, TaskScore, BenchmarkCategory
from ai.evals.scorers.deterministic import PythonCodeExecutionScorer


class CodeBenchmarkSuite(BaseBenchmarkSuite):
    """Evaluates software engineering, algorithm implementation, and sandbox test execution"""

    def __init__(self, name: str = "code"):
        super().__init__(name=name, category=BenchmarkCategory.CODE)

    def load_dataset(self, limit: Optional[int] = None) -> List[BenchmarkTask]:
        raw_items = [
            # HumanEval+ #0: Has Close Elements
            {
                "id": "humaneval_0",
                "suite": self.name,
                "category": self.category,
                "prompt": (
                    "Write a Python function `has_close_elements(numbers: list[float], threshold: float) -> bool` "
                    "that checks if in given list of numbers, are any two numbers closer to each other than given threshold.\n"
                    "Wrap your solution in ```python ... ```"
                ),
                "test_code": """
assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
assert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True
assert has_close_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True
assert has_close_elements([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True
print("All HumanEval_0 assertions passed!")
""",
                "metadata": {"source": "HumanEval+", "entry_point": "has_close_elements"}
            },
            # HumanEval+ #1: Separate Paren Groups
            {
                "id": "humaneval_1",
                "suite": self.name,
                "category": self.category,
                "prompt": (
                    "Write a Python function `separate_paren_groups(paren_string: str) -> list[str]` "
                    "Input is a string containing multiple groups of nested parentheses. Separate those groups into "
                    "separate strings and return the list of those. Ignore spaces in the input string.\n"
                    "Wrap your solution in ```python ... ```"
                ),
                "test_code": """
assert separate_paren_groups('(()()) ((())) () ((())()())') == ['(()())', '((()))', '()', '((())()())']
assert separate_paren_groups('() (()) ((())) (((())))') == ['()', '(())', '((()))', '(((())))']
assert separate_paren_groups('(()(())((())))') == ['(()(())((())))']
print("All HumanEval_1 assertions passed!")
""",
                "metadata": {"source": "HumanEval+", "entry_point": "separate_paren_groups"}
            },
            # MBPP+ #1: Is Prime
            {
                "id": "mbpp_prime_01",
                "suite": self.name,
                "category": self.category,
                "prompt": (
                    "Write a Python function `is_prime(n: int) -> bool` that returns True if a number is prime, False otherwise.\n"
                    "Wrap your solution in ```python ... ```"
                ),
                "test_code": """
assert is_prime(2) == True
assert is_prime(13) == True
assert is_prime(1) == False
assert is_prime(4) == False
assert is_prime(100) == False
assert is_prime(97) == True
print("All MBPP prime assertions passed!")
""",
                "metadata": {"source": "MBPP+", "entry_point": "is_prime"}
            },
            # Bug-Fix & Diff Quality Task: Concurrency Lock Manager
            {
                "id": "code_bugfix_lock_01",
                "suite": self.name,
                "category": self.category,
                "prompt": (
                    "Fix the bug in the following class where `release` could raise KeyError if the lock was already expired:\n"
                    "```python\n"
                    "class SimpleLockStore:\n"
                    "    def __init__(self):\n"
                    "        self.locks = {}\n"
                    "    def acquire(self, key, owner):\n"
                    "        if key in self.locks: return False\n"
                    "        self.locks[key] = owner\n"
                    "        return True\n"
                    "    def release(self, key, owner):\n"
                    "        if self.locks[key] == owner:\n"
                    "            del self.locks[key]\n"
                    "            return True\n"
                    "        return False\n"
                    "```\n"
                    "Return the complete fixed `SimpleLockStore` in ```python ... ``` without raising KeyError when key is missing."
                ),
                "test_code": """
store = SimpleLockStore()
assert store.acquire("db", "worker_1") == True
assert store.acquire("db", "worker_2") == False
assert store.release("db", "worker_1") == True
# Key is now missing: release should return False safely without KeyError!
try:
    assert store.release("db", "worker_1") == False
    assert store.release("non_existent", "worker_1") == False
except KeyError:
    assert False, "Raised KeyError on missing lock key"
print("Bugfix assertions passed!")
""",
                "metadata": {"source": "BugFix/Diff", "entry_point": "SimpleLockStore"}
            }
        ]

        tasks = [BenchmarkTask(**item) for item in raw_items]
        return tasks[:limit] if limit else tasks

    async def score_prediction(self, task: BenchmarkTask, pred: EvalPrediction) -> TaskScore:
        return await PythonCodeExecutionScorer.score_async(task, pred)
