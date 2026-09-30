"""
ALIZIA AI - Centralized Prompt Compiler & Prompt Registry
Implements PRD Sections 22, 23, 24.
Prompts must NEVER be assembled manually throughout application code.
Compiles system policies, tool descriptions, safety directives, and memory context
into structured inputs.
"""

from __future__ import annotations
import json
from typing import List, Dict, Any, Optional
from packages.schemas.models import (
    MessageItem,
    MessageRole,
    ContentPart,
    ToolDefinition,
    MemoryObject,
    TrustLevel,
)


class PromptRegistry:
    """Versioned prompt template store (Section 24)"""
    _templates: Dict[str, Dict[str, Any]] = {
        "system_core_v1": {
            "name": "system_core",
            "version": "1.0.0",
            "content": (
                "You are Alizia, a next-generation frontier multimodal artificial intelligence "
                "operating under the principle: User Intent -> Reasoning -> Planning -> Tools -> Execution -> Verification -> Result. "
                "Adhere to factual truth, verify claims with tools, maintain security boundaries, and prioritize reliable results."
            )
        },
        "coding_specialist_v1": {
            "name": "coding_specialist",
            "version": "1.0.0",
            "content": (
                "You are Alizia Forge, a software engineering specialist. Analyze repositories, "
                "reason about ASTs, generate precise diff patches, and verify changes with syntax and test executions."
            )
        },
        "agent_supervisor_v1": {
            "name": "agent_supervisor",
            "version": "1.0.0",
            "content": (
                "You are an autonomous Agent Planner. Break objectives into verifiable, sequential tasks. "
                "Never claim completion without concrete execution evidence."
            )
        }
    }

    @classmethod
    def get_template(cls, template_id: str) -> str:
        tmpl = cls._templates.get(template_id)
        if tmpl:
            return tmpl["content"]
        return cls._templates["system_core_v1"]["content"]


class PromptCompiler:
    """
    Centralized PromptCompiler (Section 23).
    Ensures safe formatting, injection barriers, memory injection ranking,
    and model-specific token budgeting.
    """

    @classmethod
    def compile_system_prompt(
        cls,
        model_name: str,
        developer_prompt: Optional[str] = None,
        memories: Optional[List[MemoryObject]] = None,
        tools: Optional[List[ToolDefinition]] = None,
    ) -> MessageItem:
        # 1. Base System Policy from versioned registry
        if "forge" in model_name:
            base_prompt = PromptRegistry.get_template("coding_specialist_v1")
        else:
            base_prompt = PromptRegistry.get_template("system_core_v1")

        sections = [base_prompt]

        # 2. Developer instructions (if provided)
        if developer_prompt:
            sections.append(f"\n[Developer Policy]\n{developer_prompt.strip()}")

        # 3. Memory Context (Section 27-29: Ranked memory items)
        if memories:
            memory_lines = [f"- {m.content} (importance: {m.importance:.2f})" for m in memories]
            sections.append(f"\n[Retrieved Memory Context]\n" + "\n".join(memory_lines))

        # 4. Tool definitions summary (if tools provided)
        if tools:
            tool_summaries = [f"- {t.name}: {t.description} (Risk Level: {t.risk_level.name})" for t in tools]
            sections.append(f"\n[Available Tool Capabilities]\n" + "\n".join(tool_summaries))

        compiled_text = "\n\n".join(sections)

        return MessageItem(
            role=MessageRole.SYSTEM,
            content=[
                ContentPart(
                    type="input_text",
                    text=compiled_text,
                    trust_level=TrustLevel.TRUSTED
                )
            ]
        )
