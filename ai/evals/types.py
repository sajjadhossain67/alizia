"""
ALIZIA AI - Evaluation Subsystem Type Definitions
Implements core schema models for evaluation tasks, predictions, scoring,
head-to-head Arena, Bradley-Terry ratings, and CI regression gating.
"""

from __future__ import annotations
import uuid
import time
from enum import Enum
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field


class BenchmarkCategory(str, Enum):
    KNOWLEDGE = "knowledge"
    MATH = "math"
    CODE = "code"
    INSTRUCTION = "instruction"
    LONG_CONTEXT = "long_context"
    MULTIMODAL = "multimodal"
    RETRIEVAL = "retrieval"
    TOOL_USE = "tool_use"
    FACTUALITY = "factuality"
    SAFETY = "safety"
    MULTILINGUAL = "multilingual"
    WRITING = "writing"


class EvaluationMetric(str, Enum):
    ACCURACY = "accuracy"
    EXACT_MATCH = "exact_match"
    PASS_AT_K = "pass@k"
    ROUGE_L = "rouge_l"
    BLEU = "bleu"
    NDCG_AT_10 = "ndcg@10"
    MRR = "mrr"
    PAIRWISE_WIN_RATE = "pairwise_win_rate"
    REFUSAL_PRECISION = "refusal_precision"
    OVER_REFUSAL_RATE = "over_refusal_rate"
    ATTACK_SUCCESS_RATE = "attack_success_rate"
    LATENCY_P95_MS = "latency_p95_ms"
    COST_PER_CORRECT_USD = "cost_per_correct_usd"


class BenchmarkTask(BaseModel):
    """Atomic evaluation item across any benchmark suite"""
    id: str = Field(default_factory=lambda: f"task_{uuid.uuid4().hex[:12]}")
    suite: str
    category: BenchmarkCategory
    prompt: str
    system_prompt: Optional[str] = None
    expected_output: Optional[str] = None
    test_code: Optional[str] = None
    rubric_id: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    canary: Optional[str] = "alizia_eval_canary_v1_guid_26b5c67b"


class EvalPrediction(BaseModel):
    """Raw prediction returned by a candidate model"""
    task_id: str
    model: str
    raw_response: str
    thinking_process: Optional[str] = None
    parsed_output: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    latency_ms: float = 0.0
    ttft_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0
    timestamp: float = Field(default_factory=time.time)


class TaskScore(BaseModel):
    """Scored evaluation result for a task"""
    task_id: str
    suite: str
    metric: EvaluationMetric
    score: float  # Normalized 0.0 to 1.0 (or 0 to 100)
    passed: bool
    details: Dict[str, Any] = Field(default_factory=dict)
    rationale: Optional[str] = None
    judge_name: Optional[str] = None


class PairwiseComparison(BaseModel):
    """Head-to-head comparison between Candidate (Alizia) and Baseline (Gemini)"""
    id: str = Field(default_factory=lambda: f"cmp_{uuid.uuid4().hex[:12]}")
    task_id: str
    suite: str
    model_a: str
    model_b: str
    winner: str  # "model_a" | "model_b" | "tie"
    confidence: float = 1.0
    criteria: Dict[str, str] = Field(default_factory=dict)
    rationale: str
    rater_type: str = "llm_judge"  # "llm_judge" | "human_arena"
    swapped_order: bool = False


class SuiteResult(BaseModel):
    """Aggregated result for a benchmark suite"""
    suite: str
    category: BenchmarkCategory
    total_tasks: int
    passed_tasks: int
    score: float  # Percentage 0.0 - 100.0
    avg_latency_ms: float
    p95_latency_ms: float
    total_cost_usd: float
    win_rate_vs_baseline: Optional[float] = None
    scores: List[TaskScore] = Field(default_factory=list)


class FullEvalReport(BaseModel):
    """Full platform evaluation run report"""
    run_id: str = Field(default_factory=lambda: f"run_{uuid.uuid4().hex[:12]}")
    timestamp: str
    model_name: str
    baseline_model: Optional[str] = "gemini-flash-latest"
    overall_score: float
    overall_win_rate_vs_baseline: Optional[float] = None
    suites: Dict[str, SuiteResult] = Field(default_factory=dict)
    git_commit: Optional[str] = None
    passed_ci_gate: bool = True
    regression_violations: List[str] = Field(default_factory=list)
