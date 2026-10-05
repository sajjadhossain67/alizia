"""
ALIZIA AI - LLM-as-a-Judge with Position-Swap Debiasing & Calibration
Implements rubric-based subjective scoring, pairwise head-to-head ranking,
position-swap bias cancellation (evaluates (A, B) and (B, A)), and judge disagreement detection.
"""

from __future__ import annotations
import json
import re
from typing import Dict, Any, List, Optional, Tuple
from ai.evals.types import BenchmarkTask, EvalPrediction, TaskScore, PairwiseComparison, EvaluationMetric


DEFAULT_JUDGE_SYSTEM_PROMPT = """You are a rigorous, unbiased Senior AI Evaluation Scientist.
Your task is to evaluate the quality of one or two AI model outputs based strictly on the provided rubric.
You must be completely impartial:
- Do not favor responses simply because they are longer or sound more verbose.
- Prioritize factual accuracy, instruction adherence, sound reasoning, and conciseness.
- Grade strictly according to the rubric criteria.
You must respond with valid JSON containing 'winner' (either 'model_a', 'model_b', or 'tie'), 'confidence' (0.5 to 1.0), and 'rationale'.
"""


class LLMJudgeScorer:
    """
    Evaluates responses using calibrated LLM-as-a-judge with position-swap debiasing.
    """

    def __init__(self, judge_model_name: str = "gemini-flash-latest"):
        self.judge_model_name = judge_model_name

    def build_pairwise_prompt(
        self,
        task_prompt: str,
        response_a: str,
        response_b: str,
        rubric: Dict[str, Any]
    ) -> str:
        rubric_str = json.dumps(rubric, indent=2)
        return f"""[USER PROMPT]
{task_prompt}

[RESPONSE CANDIDATE A]
{response_a}

[RESPONSE CANDIDATE B]
{response_b}

[EVALUATION RUBRIC]
{rubric_str}

Evaluate both responses according to the rubric.
Determine whether Candidate A or Candidate B is superior, or if they are equal (tie).
Return JSON matching:
{{
  "winner": "model_a" | "model_b" | "tie",
  "confidence": 0.5 - 1.0,
  "scores": {{
    "candidate_a": 1-10,
    "candidate_b": 1-10
  }},
  "rationale": "Clear, concise technical justification of the decision."
}}
"""

    def parse_judge_decision(self, judge_output: str) -> Dict[str, Any]:
        """Safely parses JSON decision from judge output"""
        try:
            # Strip markdown json block if present
            clean = judge_output.strip()
            if "```" in clean:
                clean = re.sub(r"^```(?:json)?\n", "", clean)
                clean = re.sub(r"\n```$", "", clean)
            return json.loads(clean)
        except Exception:
            # Fallback regex extraction
            winner_match = re.search(r'"winner":\s*"(model_a|model_b|tie)"', judge_output, re.IGNORECASE)
            winner = winner_match.group(1).lower() if winner_match else "tie"
            return {
                "winner": winner,
                "confidence": 0.5,
                "scores": {"candidate_a": 5, "candidate_b": 5},
                "rationale": "Regex extracted decision due to non-strict JSON formatting."
            }

    async def evaluate_pairwise_debiased(
        self,
        task: BenchmarkTask,
        pred_a: EvalPrediction,
        pred_b: EvalPrediction,
        rubric: Dict[str, Any],
        judge_call_fn
    ) -> PairwiseComparison:
        """
        Executes position-swap debiasing:
        Call 1: (A, B) -> judge evaluates
        Call 2: (B, A) -> judge evaluates with reversed positions
        If decisions agree when mapped back, decision is accepted.
        If decisions disagree (positional bias detected), flag as tie or resolve by confidence.
        """
        # Call 1: A first, B second
        prompt_forward = self.build_pairwise_prompt(
            task_prompt=task.prompt,
            response_a=pred_a.raw_response,
            response_b=pred_b.raw_response,
            rubric=rubric
        )
        res_forward_raw = await judge_call_fn(prompt_forward, self.judge_model_name)
        decision_forward = self.parse_judge_decision(res_forward_raw)

        # Call 2: B first, A second (Swapped order)
        prompt_reversed = self.build_pairwise_prompt(
            task_prompt=task.prompt,
            response_a=pred_b.raw_response,
            response_b=pred_a.raw_response,
            rubric=rubric
        )
        res_reversed_raw = await judge_call_fn(prompt_reversed, self.judge_model_name)
        decision_reversed = self.parse_judge_decision(res_reversed_raw)

        # Map reversed decision back to original models
        # In reversed call: "model_a" means pred_b, "model_b" means pred_a
        winner_forward = decision_forward.get("winner", "tie")
        winner_reversed_raw = decision_reversed.get("winner", "tie")

        winner_reversed_mapped = "tie"
        if winner_reversed_raw == "model_a":
            winner_reversed_mapped = "model_b"
        elif winner_reversed_raw == "model_b":
            winner_reversed_mapped = "model_a"

        # Consensus resolution
        final_winner = "tie"
        disagreement_detected = False

        if winner_forward == winner_reversed_mapped:
            final_winner = winner_forward
        else:
            disagreement_detected = True
            # Positional bias detected: assign tie
            final_winner = "tie"

        combined_rationale = (
            f"Forward evaluation: {winner_forward} (conf={decision_forward.get('confidence', 0.5):.2f}). "
            f"Reversed evaluation: {winner_reversed_mapped} (conf={decision_reversed.get('confidence', 0.5):.2f}). "
            f"{'Positional bias detected: resolving to tie.' if disagreement_detected else decision_forward.get('rationale', '')}"
        )

        return PairwiseComparison(
            task_id=task.id,
            suite=task.suite,
            model_a=pred_a.model,
            model_b=pred_b.model,
            winner=final_winner,
            confidence=decision_forward.get("confidence", 0.7),
            criteria={"position_swapped": "true", "disagreement_flag": str(disagreement_detected)},
            rationale=combined_rationale,
            swapped_order=disagreement_detected
        )
