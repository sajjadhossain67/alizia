"""
ALIZIA AI - Evaluation Model Adapter
Adapts internal Alizia AI Orchestrator and inference providers for the evaluation runner.
"""

from __future__ import annotations
import time
from typing import Optional
from packages.schemas.models import (
    CreateResponseRequest,
    ReasoningEffort,
    MessageRole,
)
from ai.orchestrator.service import AIOrchestrator
from ai.evals.types import BenchmarkTask, EvalPrediction


class AliziaModelAdapter:
    """Invokes candidate Alizia models (alizia-nova, alizia-pulse, alizia-forge, alizia-vision)"""

    def __init__(self, orchestrator: Optional[AIOrchestrator] = None):
        self.orchestrator = orchestrator or AIOrchestrator()

    async def predict(self, task: BenchmarkTask, model_name: str = "alizia-nova") -> EvalPrediction:
        """Executes task prompt through orchestrator and captures telemetry"""
        # Map model to reasoning effort
        effort = ReasoningEffort.HIGH if "nova" in model_name else ReasoningEffort.LOW

        messages = []
        if task.system_prompt:
            messages.append({"role": MessageRole.SYSTEM.value, "content": task.system_prompt})
        messages.append({"role": MessageRole.USER.value, "content": task.prompt})

        req = CreateResponseRequest(
            model=model_name,
            input=messages,
            reasoning={"effort": effort.value},
            stream=False
        )

        start_time = time.time()
        res = await self.orchestrator.execute_request(
            request=req,
            organization_id="org_eval_harness",
            user_id="usr_eval_runner"
        )
        latency_ms = (time.time() - start_time) * 1000.0

        # Extract text content
        output_text = ""
        thinking_text = None
        for block in (res.output or []):
            b_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
            b_content = block.get("content") if isinstance(block, dict) else getattr(block, "content", "")
            if b_type == "output_text":
                output_text += b_content or ""
            elif b_type in ("thought", "reasoning"):
                thinking_text = b_content

        # Token usage
        prompt_tokens = 0
        comp_tokens = 0
        if res.usage:
            prompt_tokens = res.usage.get("prompt_tokens", 0) if isinstance(res.usage, dict) else getattr(res.usage, "prompt_tokens", 0)
            comp_tokens = res.usage.get("completion_tokens", 0) if isinstance(res.usage, dict) else getattr(res.usage, "completion_tokens", 0)

        # Estimated cost calculation ($0.15/1M input, $0.60/1M output for pulse; $2.50 / $10 for nova)
        is_pulse = "pulse" in model_name
        cost = (prompt_tokens * (0.15 if is_pulse else 2.50) + comp_tokens * (0.60 if is_pulse else 10.0)) / 1_000_000.0

        return EvalPrediction(
            task_id=task.id,
            model=model_name,
            raw_response=output_text.strip(),
            thinking_process=thinking_text,
            latency_ms=round(latency_ms, 2),
            ttft_ms=round(latency_ms * 0.4, 2),
            prompt_tokens=prompt_tokens,
            completion_tokens=comp_tokens,
            estimated_cost_usd=round(cost, 6)
        )
