"""
ALIZIA AI - Semantic Model Router & Adaptive Intelligence
Implements PRD Sections 6, 17, 18, 192.
Dynamically routes incoming requests to the optimal model based on:
task complexity, modalities, coding requirements, latency targets, and budget.
"""

from __future__ import annotations
import re
from typing import List, Optional, Tuple
from packages.schemas.models import (
    CreateResponseRequest,
    ReasoningEffort,
    ContentType,
    ToolDefinition,
)


class ModelRouter:
    """
    Semantic Model Router.
    Routes queries to Alizia Pulse, Forge, Nova, or Vision.
    """

    CODING_KEYWORDS = {
        "code", "function", "class", "refactor", "debug", "bug", "python", "typescript",
        "javascript", "rust", "go", "sql", "git", "patch", "compile", "test", "ast",
        "syntax", "repo", "repository", "dockerfile", "migration"
    }

    COMPLEX_REASONING_KEYWORDS = {
        "architect", "proof", "mathematics", "theorem", "distributed system",
        "tradeoff", "analysis", "compare deep", "plan strategy", "formal verification",
        "consensus", "concurrency", "deadlock"
    }

    MODALITY_KEYWORDS = {
        "image", "screenshot", "diagram", "chart", "vision", "ocr",
        "audio", "speech", "voice", "microphone",
        "video", "screen", "ui", "interface"
    }

    TOOL_REQUIREMENTS = {
        "web.search", "knowledge.search", "memory.search",
        "code.execute", "python.execute", "file.read", "file.write",
        "database.query", "browser.open", "browser.click"
    }

    @classmethod
    def determine_route(
        cls,
        request: CreateResponseRequest,
        cost_ceiling: Optional[float] = None,
        context_window_needed: Optional[int] = None,
    ) -> str:
        """
        Determines the appropriate model (Section 17 & 18).
        Returns model_id as string.
        """
        model_id, _ = cls.determine_route_with_metadata(
            request=request,
            cost_ceiling=cost_ceiling,
            context_window_needed=context_window_needed,
        )
        return model_id

    @classmethod
    def determine_route_with_metadata(
        cls,
        request: CreateResponseRequest,
        cost_ceiling: Optional[float] = None,
        context_window_needed: Optional[int] = None,
    ) -> Tuple[str, dict]:
        """
        Determines the appropriate model and routing metadata (Section 17 & 18).
        Respects explicit model requests unless 'auto' is specified or routing optimization is enabled.
        Returns tuple of (model_id, routing_metadata).
        """
        metadata = {}

        # Respect explicit model request
        if request.model and request.model not in ["auto", "default"]:
            model_id = request.model
            # Verify the model exists in the known set
            known_models = ["alizia-nova", "alizia-pulse", "alizia-forge", "alizia-vision", "alizia-edge"]
            if model_id in known_models:
                return model_id, metadata
            # If unknown, fall through to auto-routing

        # Extract text content and modalities
        text_corpus = []
        has_images = False
        has_audio = False
        has_video = False
        tool_requirements: set = set()

        for msg in request.input:
            if isinstance(msg.content, str):
                text_corpus.append(msg.content)
            elif isinstance(msg.content, list):
                for part in msg.content:
                    part_type = getattr(part, "type", None)
                    if part_type == "image":
                        has_images = True
                    elif part_type == "audio":
                        has_audio = True
                    elif part_type == "video":
                        has_video = True
                    elif getattr(part, "text", None):
                        text_corpus.append(part.text)
                    # Check for tool references in the text
                    for tool_name in cls.TOOL_REQUIREMENTS:
                        if tool_name in text_corpus[-1] if text_corpus else "":
                            tool_requirements.add(tool_name)

        full_text = " ".join(text_corpus).lower()
        words = set(re.findall(r"\b\w+\b", full_text))

        # 1. Check for Multimodal / Vision intent (PRD 5.4, 5.5)
        if has_images or "screenshot" in words or "diagram" in words or "chart" in words or has_audio or has_video:
            model_id = "alizia-vision"
            metadata["modality"] = "multimodal"
            metadata["has_images"] = has_images
            metadata["has_audio"] = has_audio
            metadata["has_video"] = has_video
            # Check context window
            if context_window_needed and 8192 < context_window_needed:
                model_id = "alizia-nova"  # Nova has 1M token context
                metadata["reason"] = "context_window_too_large_for_vision"
            return model_id, metadata

        # 2. Check for Software Engineering / Coding intent (PRD 5.3)
        if len(words.intersection(cls.CODING_KEYWORDS)) >= 2 or "```" in full_text or tool_requirements.intersection({"code.execute", "python.execute", "file.read", "file.write"}):
            model_id = "alizia-forge"
            metadata["modality"] = "coding"
            metadata["tool_requirements"] = list(tool_requirements)
            # Check cost ceiling - Forge is more expensive
            if cost_ceiling and cost_ceiling < 0.0015:
                model_id = "alizia-pulse"
                metadata["reason"] = "cost_ceiling_too_low_for_forge"
            return model_id, metadata

        # 3. Check for Deep Reasoning / Complex Architecture (PRD 5.1)
        if len(words.intersection(cls.COMPLEX_REASONING_KEYWORDS)) >= 2 or len(full_text) > 4000:
            model_id = "alizia-nova"
            metadata["modality"] = "reasoning"
            metadata["reasoning_effort"] = "high"
            # Check context window
            if context_window_needed and context_window_needed > 1000000:
                # Nova supports 1M tokens
                metadata["context_window_ok"] = True
            # Cost check
            if cost_ceiling and cost_ceiling < 0.001:
                model_id = "alizia-pulse"
                metadata["reason"] = "cost_ceiling_too_low_for_nova"
            return model_id, metadata

        # 4. Check for agent workflows
        if any(kw in full_text for kw in ["agent", "workflow", "task", "objective", "plan", "execute"]):
            # Agent workflows benefit from Nova's reasoning capabilities
            model_id = "alizia-nova"
            metadata["modality"] = "agent"
            metadata["reasoning_effort"] = "high"
            if cost_ceiling and cost_ceiling < 0.001:
                model_id = "alizia-pulse"
                metadata["reason"] = "cost_ceiling_too_low_for_agent_nova"
            return model_id, metadata

        # 5. Check for efficient/low-cost requests (PRD 5.5 Edge)
        if cost_ceiling and cost_ceiling < 0.0005:
            model_id = "alizia-edge"
            metadata["modality"] = "efficient"
            metadata["reason"] = "cost_optimized_request"
            return model_id, metadata

        # 6. Default fast general-purpose model for chat / lightweight queries (PRD 5.2)
        model_id = "alizia-pulse"
        metadata["modality"] = "general"
        return model_id, metadata

    @classmethod
    def resolve_reasoning_effort(cls, request: CreateResponseRequest, selected_model: str) -> ReasoningEffort:
        """
        Section 6 & 192: Dynamic reasoning budget allocation.
        """
        if request.reasoning and request.reasoning.effort != ReasoningEffort.AUTO:
            return request.reasoning.effort

        # Adaptive budget based on model
        if selected_model == "alizia-pulse":
            return ReasoningEffort.LOW
        elif selected_model == "alizia-forge":
            return ReasoningEffort.MEDIUM
        elif selected_model == "alizia-nova":
            return ReasoningEffort.HIGH
        elif selected_model == "alizia-vision":
            return ReasoningEffort.MEDIUM
        elif selected_model == "alizia-edge":
            return ReasoningEffort.LOW

        return ReasoningEffort.MEDIUM

    @classmethod
    def estimate_token_cost(
        cls,
        model_id: str,
        input_text: str,
        output_tokens: int,
        pricing: Optional[dict] = None,
        request: Optional[CreateResponseRequest] = None,
    ) -> dict:
        """
        Estimate the cost of a request in tokens and currency.
        """
        if pricing is None:
            pricing = {
                "alizia-nova": {"input": 0.001, "output": 0.002, "reasoning": 0.003},
                "alizia-pulse": {"input": 0.0002, "output": 0.0005, "reasoning": 0.001},
                "alizia-forge": {"input": 0.0015, "output": 0.003, "reasoning": 0.005},
                "alizia-vision": {"input": 0.002, "output": 0.004, "reasoning": 0.006},
                "alizia-edge": {"input": 0.0001, "output": 0.0002, "reasoning": 0.0005},
            }

        p = pricing.get(model_id, pricing["alizia-pulse"])
        input_cost = len(input_text) // 4 * p["input"] / 1000  # rough heuristic: 4 chars per token
        output_cost = output_tokens * p["output"] / 1000
        reasoning_cost = 0

        if request and request.reasoning and request.reasoning.effort:
            effort = request.reasoning.effort
            if effort == ReasoningEffort.LOW:
                reasoning_cost = 120
            elif effort == ReasoningEffort.MEDIUM:
                reasoning_cost = 380
            elif effort == ReasoningEffort.HIGH:
                reasoning_cost = 1150
            elif effort == ReasoningEffort.EXTREME:
                reasoning_cost = 2400
            else:
                reasoning_cost = 350

        total = input_cost + output_cost + reasoning_cost

        return {
            "input_tokens": len(input_text) // 4,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_cost,
            "input_cost_usd": round(input_cost, 4),
            "output_cost_usd": round(output_cost, 4),
            "reasoning_cost_usd": round(reasoning_cost, 4),
            "total_cost_usd": round(total, 4),
        }
