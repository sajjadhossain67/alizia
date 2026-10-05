"""
ALIZIA AI - Core AI Request Orchestrator
Implements PRD Section 199 (Core Request Lifecycle, steps 1-27).
Coordinates:
- Safety validation (input & injection)
- Model routing & dynamic reasoning effort
- Memory retrieval
- Prompt compilation
- Provider inference (native & fallback)
- Tool execution loops & verification
- Token accounting & telemetry emission
"""

from __future__ import annotations
import time
import uuid
from typing import AsyncIterator, Dict, Any, Optional
from packages.schemas.models import (
    CreateResponseRequest,
    ResponseObject,
    StreamEvent,
    SSEEventType,
    TokenUsage,
    ReasoningEffort,
    APIErrorResponse,
    APIErrorDetails,
)
from ai.safety_engine.safety import SafetyEngine
from ai.model_router.router import ModelRouter
from ai.memory_engine.memory import MemoryEngine
from ai.prompt_engine.compiler import PromptCompiler
from agents.tool_manager.manager import ToolManager
from inference.gateway.provider import AliziaNativeProvider, FallbackProvider, ModelProvider


class AIOrchestrator:
    def __init__(
        self,
        memory_engine: Optional[MemoryEngine] = None,
        tool_manager: Optional[ToolManager] = None,
        primary_provider: Optional[ModelProvider] = None,
        fallback_provider: Optional[ModelProvider] = None
    ):
        self.memory_engine = memory_engine or MemoryEngine()
        self.tool_manager = tool_manager or ToolManager()
        self.primary_provider = primary_provider or AliziaNativeProvider()
        self.fallback_provider = fallback_provider or FallbackProvider()

    async def execute_request(
        self,
        request: CreateResponseRequest,
        organization_id: str = "org_default",
        user_id: str = "usr_default"
    ) -> ResponseObject:
        """
        Executes non-streaming request lifecycle (Section 199).
        """
        req_id = f"req_{uuid.uuid4().hex[:16]}"

        # Step 6: Input Safety Classification
        prompt_text = " ".join([
            m.content if isinstance(m.content, str) else " ".join(p.text or "" for p in m.content)
            for m in request.input
        ])
        is_safe, safety_err = SafetyEngine.validate_input_safety(prompt_text)
        if not is_safe:
            raise ValueError(safety_err)

        # Step 7 & 13: Detect task complexity & Select model
        selected_model = ModelRouter.determine_route(request)
        request.model = selected_model

        # Step 14: Allocate reasoning budget
        effort = ModelRouter.resolve_reasoning_effort(request, selected_model)
        if request.reasoning:
            request.reasoning.effort = effort

        # Step 10: Retrieve relevant memory (Section 29)
        memories = self.memory_engine.retrieve_relevant_memories(
            owner_id=user_id,
            query=prompt_text,
            top_k=3
        )

        # Step 12: Assemble context via PromptCompiler (Section 22, 23)
        system_msg = PromptCompiler.compile_system_prompt(
            model_name=selected_model,
            memories=memories,
            tools=request.tools
        )
        # Prepend compiled system prompt
        prepared_inputs = [system_msg] + request.input
        request.input = prepared_inputs

        # Step 15: Execute inference via Model Gateway
        try:
            resp = await self.primary_provider.generate(request)
        except Exception as e:
            # Step 128: Fallback Strategy
            resp = await self.fallback_provider.generate(request)

        # Step 16-19: Handle tool requests if status is requires_action
        if resp.status == "requires_action" and resp.output:
            for item in resp.output:
                if item.get("type") == "tool_call":
                    tool_res = await self.tool_manager.execute_tool_protocol(
                        tool_name=item["name"],
                        arguments=item["arguments"]
                    )
                    # Step 20: Append tool result
                    resp.output.append({
                        "type": "tool_result",
                        "execution_id": tool_res.execution_id,
                        "status": tool_res.status,
                        "output": tool_res.output
                    })
            resp.status = "completed"

        # Step 22: Output secrets scan & redaction (Section 77)
        for out in resp.output:
            if out.get("type") == "output_text" and "text" in out:
                redacted, _ = SafetyEngine.scan_for_secrets(out["text"])
                out["text"] = redacted

        # Step 23: Proof Mode Verification (Sprint 2 - Phase 1)
        if request.proof:
            from ai.proof_engine import ProofEngine
            resp.proof = await ProofEngine.verify_response(resp, request)

        return resp

    async def stream_request(
        self,
        request: CreateResponseRequest,
        organization_id: str = "org_default",
        user_id: str = "usr_default"
    ) -> AsyncIterator[StreamEvent]:
        """
        Executes streaming request lifecycle (Section 16 & 199).
        """
        # Input safety check
        prompt_text = " ".join([
            m.content if isinstance(m.content, str) else " ".join(p.text or "" for p in m.content)
            for m in request.input
        ])
        is_safe, safety_err = SafetyEngine.validate_input_safety(prompt_text)
        if not is_safe:
            yield StreamEvent(
                event=SSEEventType.RESPONSE_FAILED,
                data={"error": safety_err}
            )
            return

        selected_model = ModelRouter.determine_route(request)
        request.model = selected_model

        # Step 10: Retrieve relevant memory (Section 29)
        memories = self.memory_engine.retrieve_relevant_memories(
            owner_id=user_id,
            query=prompt_text,
            top_k=3
        )

        # Step 12: Assemble context via PromptCompiler (Section 22, 23)
        system_msg = PromptCompiler.compile_system_prompt(
            model_name=selected_model,
            memories=memories,
            tools=request.tools
        )
        prepared_inputs = [system_msg] + request.input
        request.input = prepared_inputs

        accumulated_text = ""
        last_resp_id = None

        async for event in self.primary_provider.stream(request):
            if event.event == SSEEventType.OUTPUT_TEXT_DELTA:
                accumulated_text += event.data.get("delta", "")
            elif event.event == SSEEventType.RESPONSE_CREATED:
                last_resp_id = event.data.get("id")

            if event.event == SSEEventType.RESPONSE_COMPLETED and request.proof:
                try:
                    from ai.proof_engine import ProofEngine
                    temp_resp = ResponseObject(
                        id=last_resp_id or "resp_stream",
                        model=request.model,
                        output=[{"type": "output_text", "text": accumulated_text}]
                    )
                    proof_obj = await ProofEngine.verify_response(temp_resp, request)
                    yield StreamEvent(
                        event=SSEEventType.RESPONSE_PROOF,
                        data=proof_obj.model_dump()
                    )
                except Exception as proof_err:
                    pass

            yield event
