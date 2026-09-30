"""
ALIZIA AI - Semantic Model Router & Adaptive Intelligence
Implements PRD Sections 6, 17, 18, 192.
Dynamically routes incoming requests to the optimal model based on:
task complexity, modalities, coding requirements, latency targets, and budget.
"""

from __future__ import annotations
import re
from typing import List, Optional
from packages.schemas.models import (
    CreateResponseRequest,
    ReasoningEffort,
    ContentType,
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

    @classmethod
    def determine_route(cls, request: CreateResponseRequest) -> str:
        """
        Determines the appropriate model (Section 17 & 18).
        Respects explicit model requests unless 'auto' is specified or routing optimization is enabled.
        """
        if request.model and request.model not in ["auto", "default"]:
            return request.model

        # Extract text content and modalities
        text_corpus = []
        has_images = False

        for msg in request.input:
            if isinstance(msg.content, str):
                text_corpus.append(msg.content)
            elif isinstance(msg.content, list):
                for part in msg.content:
                    if getattr(part, "type", None) == "image":
                        has_images = True
                    if getattr(part, "text", None):
                        text_corpus.append(part.text)

        full_text = " ".join(text_corpus).lower()
        words = set(re.findall(r"\b\w+\b", full_text))

        # 1. Check for Multimodal / Vision intent
        if has_images or "screenshot" in words or "diagram" in words or "chart" in words:
            return "alizia-vision"

        # 2. Check for Software Engineering / Coding intent
        if len(words.intersection(cls.CODING_KEYWORDS)) >= 2 or "```" in full_text:
            return "alizia-forge"

        # 3. Check for Deep Reasoning / Complex Architecture
        if len(words.intersection(cls.COMPLEX_REASONING_KEYWORDS)) >= 2 or len(full_text) > 4000:
            return "alizia-nova"

        # 4. Default fast general-purpose model for chat / lightweight queries
        return "alizia-pulse"

    @classmethod
    def resolve_reasoning_effort(cls, request: CreateResponseRequest, selected_model: str) -> ReasoningEffort:
        """
        Section 6 & 192: Dynamic reasoning budget allocation.
        """
        if request.reasoning and request.reasoning.effort != ReasoningEffort.AUTO:
            return request.reasoning.effort

        # Adaptive budget
        if selected_model == "alizia-pulse":
            return ReasoningEffort.LOW
        elif selected_model == "alizia-forge":
            return ReasoningEffort.MEDIUM
        elif selected_model == "alizia-nova":
            return ReasoningEffort.HIGH

        return ReasoningEffort.MEDIUM
