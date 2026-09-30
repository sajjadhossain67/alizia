"""
ALIZIA AI - Tool Manager & Confirmation Engine
Implements PRD Sections 47, 48, 74, 75.
Coordinates:
- Tool Registry
- Permission Scopes
- Risk Evaluation (R0 to R4)
- Confirmation Token generation for high-impact actions (Section 75)
"""

from __future__ import annotations
import secrets
import time
from typing import Dict, Any, Optional, Tuple
from packages.schemas.models import (
    ToolDefinition,
    ToolParameterSchema,
    ToolResult,
    RiskLevel
)
from ai.safety_engine.safety import SafetyEngine
from tools.executor import ToolExecutor


class ToolManager:
    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._pending_confirmations: Dict[str, Dict[str, Any]] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """Registers Section 46 First-Party Tools"""
        self.register_tool(ToolDefinition(
            name="web.search",
            description="Search the web for up-to-date factual information and citations.",
            parameters=ToolParameterSchema(
                type="object",
                properties={"query": {"type": "string", "description": "The search keywords"}},
                required=["query"]
            ),
            permission_scope="internet.search",
            risk_level=RiskLevel.R0_INFORMATIONAL
        ))

        self.register_tool(ToolDefinition(
            name="code.execute",
            description="Execute code in an isolated ephemeral sandbox environment.",
            parameters=ToolParameterSchema(
                type="object",
                properties={"code": {"type": "string", "description": "Python source code"}},
                required=["code"]
            ),
            permission_scope="code.execute",
            risk_level=RiskLevel.R1_REVERSIBLE
        ))

        self.register_tool(ToolDefinition(
            name="file.read",
            description="Read content from a workspace file.",
            parameters=ToolParameterSchema(
                type="object",
                properties={"path": {"type": "string", "description": "Workspace relative path"}},
                required=["path"]
            ),
            permission_scope="files.read",
            risk_level=RiskLevel.R0_INFORMATIONAL
        ))

        self.register_tool(ToolDefinition(
            name="database.query",
            description="Execute a query against a connected database.",
            parameters=ToolParameterSchema(
                type="object",
                properties={"query": {"type": "string", "description": "SQL statement"}},
                required=["query"]
            ),
            permission_scope="database.write",
            risk_level=RiskLevel.R3_SENSITIVE
        ))

    def register_tool(self, tool_def: ToolDefinition):
        self._tools[tool_def.name] = tool_def

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def list_tools(self) -> Dict[str, ToolDefinition]:
        return self._tools

    async def execute_tool_protocol(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        user_scopes: Optional[list] = None,
        confirmation_token: Optional[str] = None
    ) -> ToolResult:
        """
        Section 48: Tool Execution Protocol
        Model requests tool -> Schema validator -> Permission engine -> Policy engine ->
        User confirmation if needed -> Sandbox execution -> Result sanitizer -> Model receives result
        """
        tool = self.get_tool(tool_name)
        if not tool:
            return ToolResult(
                execution_id=f"tool_exec_err_{secrets.token_hex(4)}",
                tool_name=tool_name,
                status="failed",
                output=None,
                started_at=time.time(),
                completed_at=time.time(),
                error=f"Tool '{tool_name}' is not registered."
            )

        # 1. Permission Scope Check (Section 12 & 47)
        if user_scopes is not None and tool.permission_scope not in user_scopes:
            return ToolResult(
                execution_id=f"tool_exec_denied_{secrets.token_hex(4)}",
                tool_name=tool_name,
                status="denied",
                output=None,
                started_at=time.time(),
                completed_at=time.time(),
                error=f"Unauthorized: missing required scope '{tool.permission_scope}'."
            )

        # 2. Risk Engine Evaluation (Section 74)
        risk = SafetyEngine.evaluate_tool_risk(tool_name, arguments)

        # 3. Confirmation Engine Check (Section 75)
        # Sensitive (R3) or Destructive (R4) actions require explicit user confirmation
        if risk in [RiskLevel.R3_SENSITIVE, RiskLevel.R4_DESTRUCTIVE]:
            if not confirmation_token:
                # Issue confirmation token bound to this exact action
                token = f"alz_confirm_{secrets.token_hex(16)}"
                self._pending_confirmations[token] = {
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "risk": risk.value,
                    "issued_at": time.time()
                }
                return ToolResult(
                    execution_id=f"tool_exec_hold_{secrets.token_hex(4)}",
                    tool_name=tool_name,
                    status="requires_confirmation",
                    output={"message": f"Action requires user confirmation (Risk Level: {risk.name})."},
                    started_at=time.time(),
                    completed_at=time.time(),
                    confirmation_token=token
                )

            # Validate submitted token
            pending = self._pending_confirmations.get(confirmation_token)
            if not pending or pending["tool_name"] != tool_name or pending["arguments"] != arguments:
                return ToolResult(
                    execution_id=f"tool_exec_invalid_{secrets.token_hex(4)}",
                    tool_name=tool_name,
                    status="denied",
                    output=None,
                    started_at=time.time(),
                    completed_at=time.time(),
                    error="Invalid or expired confirmation token."
                )
            # Remove used token
            del self._pending_confirmations[confirmation_token]

        # 4. Sandbox Execution
        result = await ToolExecutor.execute(tool_name, arguments)
        return result
