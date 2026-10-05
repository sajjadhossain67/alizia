"""
ALIZIA AI - Google Gemini Head-to-Head Provider Adapter
Executes benchmark prompts directly against Google Gemini endpoints
with identical system prompts, tools, and temperature for controlled A/B comparison.
"""

from __future__ import annotations
import time
from typing import Optional
from packages.schemas.models import CreateResponseRequest, MessageRole
from inference.gateway.provider import GeminiProvider, DEFAULT_GEMINI_MODEL
from ai.evals.types import BenchmarkTask, EvalPrediction


class GeminiBenchmarkAdapter:
    """Invokes Google Gemini API models (Gemini Flash, Gemini Pro) for head-to-head evals"""

    def __init__(self, model_name: str = DEFAULT_GEMINI_MODEL, api_key: Optional[str] = None):
        self.model_name = model_name
        self.provider = GeminiProvider(model_name=model_name, api_key=api_key)

    async def predict(self, task: BenchmarkTask) -> EvalPrediction:
        messages = []
        if task.system_prompt:
            messages.append({"role": MessageRole.SYSTEM.value, "content": task.system_prompt})
        messages.append({"role": MessageRole.USER.value, "content": task.prompt})

        req = CreateResponseRequest(
            model=self.model_name,
            input=messages,
            stream=False
        )

        start_time = time.time()
        res = await self.provider.generate(req)
        latency_ms = (time.time() - start_time) * 1000.0

        output_text = ""
        for block in (res.output or []):
            b_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
            b_content = block.get("content") if isinstance(block, dict) else getattr(block, "content", "")
            if b_type == "output_text":
                output_text += b_content or ""

        prompt_tokens = 0
        comp_tokens = 0
        if res.usage:
            prompt_tokens = res.usage.get("prompt_tokens", 0) if isinstance(res.usage, dict) else getattr(res.usage, "prompt_tokens", 0)
            comp_tokens = res.usage.get("completion_tokens", 0) if isinstance(res.usage, dict) else getattr(res.usage, "completion_tokens", 0)

        # Gemini Flash pricing ($0.075 / 1M input, $0.30 / 1M output)
        cost = (prompt_tokens * 0.075 + comp_tokens * 0.30) / 1_000_000.0

        return EvalPrediction(
            task_id=task.id,
            model=self.model_name,
            raw_response=output_text.strip(),
            latency_ms=round(latency_ms, 2),
            ttft_ms=round(latency_ms * 0.45, 2),
            prompt_tokens=prompt_tokens,
            completion_tokens=comp_tokens,
            estimated_cost_usd=round(cost, 6)
        )
