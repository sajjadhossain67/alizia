"""
ALIZIA AI - Core Protocol Schemas & Data Transfer Objects
Implements PRD Sections 6, 14-16, 26, 28, 40-49, 74-76, 111-115, 134, 161, 166.
"""

from __future__ import annotations
import uuid
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Literal
from pydantic import BaseModel, Field, ConfigDict


# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------
class ReasoningEffort(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"
    AUTO = "auto"


class MessageRole(str, Enum):
    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ContentType(str, Enum):
    INPUT_TEXT = "input_text"
    OUTPUT_TEXT = "output_text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    FILE = "file"
    CODE = "code"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    CITATION = "citation"
    ARTIFACT = "artifact"


class TrustLevel(str, Enum):
    """Section 76: Prompt Injection Defense trust categorization"""
    TRUSTED = "trusted"              # System and developer policies
    AUTHORIZED = "authorized"        # Authenticated user direct prompt
    UNTRUSTED_DATA = "untrusted_data" # Connectors, web fetch, scraped pages, uploaded files


class RiskLevel(int, Enum):
    """Section 74: Action Risk Engine"""
    R0_INFORMATIONAL = 0  # Web search, calculations, read-only
    R1_REVERSIBLE = 1     # Local temporary scratch file
    R2_EXTERNALLY_VISIBLE = 2 # Send public comment, draft email
    R3_SENSITIVE = 3      # Modify production settings/records
    R4_DESTRUCTIVE = 4    # Delete tables, purge databases, drop resources


class AgentState(str, Enum):
    """Section 41: Explicit Agent States"""
    QUEUED = "queued"
    PLANNING = "planning"
    RUNNING = "running"
    WAITING_FOR_TOOL = "waiting_for_tool"
    WAITING_FOR_USER = "waiting_for_user"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


# -----------------------------------------------------------------------------
# Message and Content Part Schemas (Sections 14, 26, 76)
# -----------------------------------------------------------------------------
class ContentPart(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str
    text: Optional[str] = None
    media_url: Optional[str] = None
    media_mime_type: Optional[str] = None
    trust_level: TrustLevel = Field(
        default=TrustLevel.AUTHORIZED,
        description="Section 76 Trust metadata to neutralize indirect prompt injection"
    )


class MessageItem(BaseModel):
    id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:16]}")
    role: MessageRole
    content: Union[str, List[ContentPart]]
    parent_message_id: Optional[str] = None
    created_at: float = Field(default_factory=time.time)


# -----------------------------------------------------------------------------
# Tool & Function Calling Schemas (Sections 45-47, 65, 66)
# -----------------------------------------------------------------------------
class ToolParameterSchema(BaseModel):
    type: str = "object"
    properties: Dict[str, Any] = Field(default_factory=dict)
    required: Optional[List[str]] = None


class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: ToolParameterSchema
    permission_scope: str = "general"
    risk_level: RiskLevel = RiskLevel.R0_INFORMATIONAL


class ToolCall(BaseModel):
    id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:12]}")
    type: Literal["function"] = "function"
    name: str
    arguments: Dict[str, Any]


class ToolResult(BaseModel):
    """Section 49: Cryptographically unique tool execution result"""
    execution_id: str
    tool_name: str
    status: Literal["success", "failed", "requires_confirmation", "denied"]
    output: Any
    started_at: float
    completed_at: float
    error: Optional[str] = None
    confirmation_token: Optional[str] = None


# -----------------------------------------------------------------------------
# Unified AI API (POST /v1/responses) (Section 14 & 15)
# -----------------------------------------------------------------------------
class ReasoningConfig(BaseModel):
    effort: ReasoningEffort = ReasoningEffort.AUTO


class ResponseFormat(BaseModel):
    type: Literal["text", "json_object", "json_schema"] = "text"
    schema_definition: Optional[Dict[str, Any]] = Field(default=None, alias="schema")


class CreateResponseRequest(BaseModel):
    model: str = Field(default="alizia-nova", description="alizia-nova, alizia-pulse, alizia-forge, alizia-vision, or fallback")
    input: List[MessageItem]
    reasoning: Optional[ReasoningConfig] = Field(default_factory=ReasoningConfig)
    tools: Optional[List[ToolDefinition]] = Field(default_factory=list)
    response_format: Optional[ResponseFormat] = None
    stream: bool = False
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 4096
    workspace_id: Optional[str] = None
    conversation_id: Optional[str] = None


class TokenUsage(BaseModel):
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0


class ResponseObject(BaseModel):
    id: str = Field(default_factory=lambda: f"resp_{uuid.uuid4().hex[:18]}")
    object: str = "response"
    created_at: int = Field(default_factory=lambda: int(time.time()))
    model: str
    status: Literal["completed", "in_progress", "failed", "requires_action"] = "completed"
    output: List[Dict[str, Any]] = Field(default_factory=list)
    reasoning_summary: Optional[str] = None
    usage: TokenUsage = Field(default_factory=TokenUsage)
    citations: Optional[List[Dict[str, Any]]] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Streaming Protocol Events (Section 16)
# -----------------------------------------------------------------------------
class SSEEventType(str, Enum):
    RESPONSE_CREATED = "response.created"
    OUTPUT_TEXT_DELTA = "response.output_text.delta"
    REASONING_SUMMARY_DELTA = "response.reasoning_summary.delta"
    TOOL_CALL_CREATED = "response.tool_call.created"
    TOOL_CALL_ARGUMENTS_DELTA = "response.tool_call.arguments.delta"
    TOOL_CALL_COMPLETED = "response.tool_call.completed"
    OUTPUT_ITEM_COMPLETED = "response.output_item.completed"
    RESPONSE_COMPLETED = "response.completed"
    RESPONSE_FAILED = "response.failed"


class StreamEvent(BaseModel):
    event: SSEEventType
    data: Dict[str, Any]


# -----------------------------------------------------------------------------
# Agent System & Verification Engine (Sections 40, 42, 114, 130, 191)
# -----------------------------------------------------------------------------
class PlanStep(BaseModel):
    step_number: int
    description: str
    status: Literal["pending", "executing", "verified", "failed", "skipped"] = "pending"
    evidence: Optional[Dict[str, Any]] = None


class AgentPlan(BaseModel):
    objective: str
    steps: List[PlanStep] = Field(default_factory=list)


class CreateAgentRunRequest(BaseModel):
    goal: str
    workspace_id: Optional[str] = None
    max_steps: int = 100
    token_budget: int = 1000000
    cost_budget: float = 10.00
    allowed_tools: Optional[List[str]] = None


class AgentRunObject(BaseModel):
    id: str = Field(default_factory=lambda: f"agent_run_{uuid.uuid4().hex[:16]}")
    goal: str
    status: AgentState = AgentState.QUEUED
    current_step: int = 0
    max_steps: int = 100
    workspace_id: Optional[str] = None
    plan: List[PlanStep] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    completed_at: Optional[float] = None
    error: Optional[str] = None


class VerificationResult(BaseModel):
    """Section 130 & 191: Verifiable AI Evidence Result"""
    passed: bool
    evidence_type: str # test_run, compiler_check, linter, citation_check
    details: Dict[str, Any]
    verifier_notes: Optional[str] = None


# -----------------------------------------------------------------------------
# Memory Platform (Sections 27-30)
# -----------------------------------------------------------------------------
class MemoryObject(BaseModel):
    id: str = Field(default_factory=lambda: f"mem_{uuid.uuid4().hex[:12]}")
    owner_id: str
    scope: Literal["session", "user", "workspace", "agent"]
    content: str
    embedding: Optional[List[float]] = None
    importance: float = 0.5
    confidence: float = 0.9
    created_at: float = Field(default_factory=time.time)
    last_used_at: float = Field(default_factory=time.time)


# -----------------------------------------------------------------------------
# Embeddings & Error Format (Sections 161, 166)
# -----------------------------------------------------------------------------
class CreateEmbeddingRequest(BaseModel):
    model: str = "alizia-embed-v1"
    input: Union[str, List[str]]


class EmbeddingItem(BaseModel):
    object: str = "embedding"
    index: int
    embedding: List[float]


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingItem]
    model: str
    usage: Dict[str, int]


class APIErrorDetails(BaseModel):
    type: str
    code: str
    message: str
    request_id: str


class APIErrorResponse(BaseModel):
    error: APIErrorDetails
