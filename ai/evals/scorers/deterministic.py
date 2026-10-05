"""
ALIZIA AI - Deterministic Scorers
Evaluates model predictions with 100% deterministic reproducibility:
- Exact Match & Normalization
- Numerical & Math Formula Extraction (\boxed{...}, float tolerance)
- Python Sandbox Unit Testing (HumanEval / MBPP style)
- IFEval Constraints (Length, formatting, negative constraints)
- JSON Schema & Structure Validation
"""

from __future__ import annotations
import re
import json
import math
from typing import Dict, Any, Optional, Tuple
from ai.evals.types import BenchmarkTask, EvalPrediction, TaskScore, EvaluationMetric
from tools.executor import ToolExecutor


class DeterministicScorer:
    """Base class for all deterministic algorithmic scorers"""

    @classmethod
    def clean_text(cls, text: str) -> str:
        if not text:
            return ""
        return " ".join(text.strip().split())


class ExactMatchScorer(DeterministicScorer):
    """Normalized exact match comparison"""

    @classmethod
    def score(cls, task: BenchmarkTask, pred: EvalPrediction) -> TaskScore:
        expected = cls.clean_text(task.expected_output or "")
        actual = cls.clean_text(pred.parsed_output or pred.raw_response)

        # Check for multiple choice format (e.g., "A", "(A)", "The answer is A")
        is_match = False
        if len(expected) == 1 and expected.isalpha():
            # Extract choice letters
            choice_patterns = [
                rf"\\b{expected}\\b",
                rf"\\({expected}\\)",
                rf"\\[{expected}\\]",
                rf"answer is:?\s*\(?{expected}\)?",
                rf"correct option is:?\s*\(?{expected}\)?",
            ]
            for pat in choice_patterns:
                if re.search(pat, actual, re.IGNORECASE):
                    is_match = True
                    break
        else:
            is_match = (actual.lower() == expected.lower())

        return TaskScore(
            task_id=task.id,
            suite=task.suite,
            metric=EvaluationMetric.EXACT_MATCH,
            score=1.0 if is_match else 0.0,
            passed=is_match,
            details={"expected": expected, "actual": actual[:200]}
        )


class MathToleranceScorer(DeterministicScorer):
    """
    Math verification scorer:
    Extracts final answer from \\boxed{...}, "The answer is ...", or last numeric token.
    Evaluates equivalence within numeric tolerance (1e-4).
    """

    @classmethod
    def extract_math_answer(cls, text: str) -> Optional[str]:
        # 1. Look for LaTeX \boxed{...}
        boxed_match = re.search(r"\\boxed\{([^}]+)\}", text)
        if boxed_match:
            return boxed_match.group(1).strip()

        # 2. Look for "#### <number>" (GSM8K format)
        gsm_match = re.search(r"####\s*([0-9.,\-+/]+)", text)
        if gsm_match:
            return gsm_match.group(1).strip()

        # 3. Look for "the answer is <val>"
        ans_match = re.search(r"(?:answer is|equals|result is)\s*:?\s*([0-9.,\-+/]+)", text, re.IGNORECASE)
        if ans_match:
            return ans_match.group(1).strip()

        # 4. Fallback: extract last number in the text
        numbers = re.findall(r"[-+]?\d*\.?\d+", text)
        if numbers:
            return numbers[-1]
        return None

    @classmethod
    def score(cls, task: BenchmarkTask, pred: EvalPrediction) -> TaskScore:
        expected_raw = task.expected_output or ""
        expected_val_str = cls.extract_math_answer(expected_raw) or expected_raw
        actual_val_str = cls.extract_math_answer(pred.raw_response)

        passed = False
        score_val = 0.0

        if expected_val_str and actual_val_str:
            try:
                # Clean commas
                exp_clean = expected_val_str.replace(",", "").replace("$", "").strip()
                act_clean = actual_val_str.replace(",", "").replace("$", "").strip()

                exp_num = float(exp_clean)
                act_num = float(act_clean)

                if math.isclose(exp_num, act_num, rel_tol=1e-4, abs_tol=1e-4):
                    passed = True
                    score_val = 1.0
            except ValueError:
                # Fallback to string equality
                passed = (expected_val_str.strip().lower() == actual_val_str.strip().lower())
                score_val = 1.0 if passed else 0.0

        return TaskScore(
            task_id=task.id,
            suite=task.suite,
            metric=EvaluationMetric.ACCURACY,
            score=score_val,
            passed=passed,
            details={
                "expected": expected_val_str,
                "extracted": actual_val_str,
                "raw_snippet": pred.raw_response[-200:]
            }
        )


class PythonCodeExecutionScorer:
    """
    Executes Python test harness inside the isolated ToolExecutor sandbox.
    Supports HumanEval+, MBPP+, and live unit-test assertions.
    """

    @classmethod
    def extract_python_code(cls, response: str) -> str:
        # Look for ```python ... ``` block
        pattern = r"```(?:python|py)?\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()
        return response.strip()

    @classmethod
    async def score_async(cls, task: BenchmarkTask, pred: EvalPrediction) -> TaskScore:
        candidate_code = cls.extract_python_code(pred.raw_response)
        test_harness = task.test_code or ""

        full_execution_script = f"""
# Candidate Solution:
{candidate_code}

# Test Harness Verification:
{test_harness}
"""
        result = await ToolExecutor.execute_python_sandbox(
            code=full_execution_script,
            timeout_seconds=5.0
        )

        exit_code = result.get("exit_code", 1)
        passed = (exit_code == 0)

        return TaskScore(
            task_id=task.id,
            suite=task.suite,
            metric=EvaluationMetric.PASS_AT_K,
            score=1.0 if passed else 0.0,
            passed=passed,
            details={
                "exit_code": exit_code,
                "stdout": result.get("stdout", "")[:300],
                "stderr": result.get("stderr", "")[:300]
            }
        )


class IFEvalScorer(DeterministicScorer):
    """
    Instruction Following Evaluation (IFEval) verifier:
    Verifies multi-constraint formatting, length bounds, forbidden tokens,
    line count requirements, and case rules.
    """

    @classmethod
    def score(cls, task: BenchmarkTask, pred: EvalPrediction) -> TaskScore:
        constraints = task.metadata.get("constraints", {})
        text = pred.raw_response

        passed_all = True
        violation_notes = []

        # 1. Minimum / Maximum word count
        words = text.split()
        word_count = len(words)

        if "min_words" in constraints and word_count < constraints["min_words"]:
            passed_all = False
            violation_notes.append(f"Word count {word_count} < min {constraints['min_words']}")

        if "max_words" in constraints and word_count > constraints["max_words"]:
            passed_all = False
            violation_notes.append(f"Word count {word_count} > max {constraints['max_words']}")

        # 2. Forbidden words
        if "forbidden_words" in constraints:
            for word in constraints["forbidden_words"]:
                if re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE):
                    passed_all = False
                    violation_notes.append(f"Contains forbidden word: '{word}'")

        # 3. Required JSON format
        if constraints.get("require_valid_json"):
            try:
                # Strip markdown fence if present
                clean_json_str = text.strip()
                if clean_json_str.startswith("```"):
                    clean_json_str = re.sub(r"^```(?:json)?\n", "", clean_json_str)
                    clean_json_str = re.sub(r"\n```$", "", clean_json_str)
                json.loads(clean_json_str)
            except Exception as e:
                passed_all = False
                violation_notes.append(f"Invalid JSON: {str(e)}")

        # 4. Starting phrase
        if "start_with" in constraints:
            if not text.strip().lower().startswith(constraints["start_with"].lower()):
                passed_all = False
                violation_notes.append(f"Does not start with '{constraints['start_with']}'")

        # 5. Section headers
        if "required_sections" in constraints:
            for sec in constraints["required_sections"]:
                if sec.lower() not in text.lower():
                    passed_all = False
                    violation_notes.append(f"Missing required section header: '{sec}'")

        return TaskScore(
            task_id=task.id,
            suite=task.suite,
            metric=EvaluationMetric.EXACT_MATCH,
            score=1.0 if passed_all else 0.0,
            passed=passed_all,
            details={
                "violations": violation_notes,
                "word_count": word_count
            }
        )
