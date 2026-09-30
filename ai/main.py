"""
ALIZIA AI - Core API & AI Orchestration Service
Implements PRD Sections 14-16, 40, 69, 117, 161, 162, 166, 176.
"""

from __future__ import annotations
import json
import time
import uuid
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, Request, Response, HTTPException, status, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from packages.schemas.models import (
    CreateResponseRequest,
    ResponseObject,
    CreateEmbeddingRequest,
    EmbeddingResponse,
    CreateAgentRunRequest,
    AgentRunObject,
    AgentState,
    MemoryObject,
    APIErrorResponse,
    APIErrorDetails,
)
from ai.orchestrator.service import AIOrchestrator
from ai.memory_engine.memory import MemoryEngine
from ai.rag_engine.rag import RAGEngine
from agents.tool_manager.manager import ToolManager
from agents.runtime.state_machine import AgentStateMachine


app = FastAPI(
    title="Alizia AI Platform Backend",
    version="1.0.0",
    description="Frontier multimodal AI and agent platform architecture API"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Core singletons
memory_engine = MemoryEngine()
rag_engine = RAGEngine()
tool_manager = ToolManager()
orchestrator = AIOrchestrator(
    memory_engine=memory_engine,
    tool_manager=tool_manager
)

# In-memory storage for active agent runs
agent_runs_db: Dict[str, AgentRunObject] = {}


# -----------------------------------------------------------------------------
# Middleware: Request IDs & Rate-Limit Headers (Sections 162 & 176)
# -----------------------------------------------------------------------------
@app.middleware("http")
async def request_lifecycle_middleware(request: Request, call_next):
    req_id = request.headers.get("x-request-id", f"req_{uuid.uuid4().hex[:16]}")
    request.state.request_id = req_id

    # Simulated tenant context from auth header
    auth_header = request.headers.get("authorization", "")
    request.state.org_id = "org_production_01" if "alz_live_" in auth_header else "org_default"
    request.state.user_id = "usr_01"

    start_time = time.time()
    response: Response = await call_next(request)

    # Section 162: Request ID propagation
    response.headers["x-request-id"] = req_id

    # Section 176: Rate-Limit Headers
    response.headers["x-ratelimit-limit-requests"] = "10000"
    response.headers["x-ratelimit-remaining-requests"] = "9985"
    response.headers["x-ratelimit-reset-requests"] = "60"
    response.headers["x-ratelimit-limit-tokens"] = "2000000"
    response.headers["x-process-time-ms"] = str(round((time.time() - start_time) * 1000, 2))

    return response


# -----------------------------------------------------------------------------
# Section 14, 15, 16: Unified AI Responses API (POST /v1/responses)
# -----------------------------------------------------------------------------
@app.post(
    "/v1/responses",
    response_model=None,
    summary="Unified AI Responses API (Section 14)",
    responses={
        200: {"description": "Standard response or SSE stream"},
        400: {"model": APIErrorResponse},
        429: {"model": APIErrorResponse}
    }
)
async def create_response(req_body: CreateResponseRequest, request: Request):
    req_id = request.state.request_id
    org_id = request.state.org_id
    user_id = request.state.user_id

    try:
        if req_body.stream:
            # Section 16: SSE Streaming Protocol
            async def event_generator():
                async for evt in orchestrator.stream_request(req_body, org_id, user_id):
                    yield {
                        "event": evt.event.value,
                        "data": json.dumps(evt.data)
                    }

            return EventSourceResponse(event_generator())
        else:
            # Synchronous / Direct execution
            resp = await orchestrator.execute_request(req_body, org_id, user_id)
            return resp
    except ValueError as val_err:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=APIErrorResponse(
                error=APIErrorDetails(
                    type="invalid_request_error",
                    code="safety_or_validation_failure",
                    message=str(val_err),
                    request_id=req_id
                )
            ).model_dump()
        )
    except Exception as exc:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=APIErrorResponse(
                error=APIErrorDetails(
                    type="internal_server_error",
                    code="orchestration_error",
                    message=str(exc),
                    request_id=req_id
                )
            ).model_dump()
        )


# -----------------------------------------------------------------------------
# Section 166: Embeddings API
# -----------------------------------------------------------------------------
@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(req_body: CreateEmbeddingRequest):
    return await orchestrator.primary_provider.embed(req_body)


# -----------------------------------------------------------------------------
# Sections 40, 69: Async Agent Runs API
# -----------------------------------------------------------------------------
@app.post("/v1/agents/runs", response_model=AgentRunObject)
async def create_agent_run(req_body: CreateAgentRunRequest, request: Request):
    run = AgentRunObject(
        goal=req_body.goal,
        max_steps=req_body.max_steps,
        workspace_id=req_body.workspace_id,
        status=AgentState.QUEUED
    )
    agent_runs_db[run.id] = run

    # Execute state machine asynchronously or immediately for testing
    sm = AgentStateMachine(run, tool_manager)
    await sm.run_until_completion(max_ticks=25)
    agent_runs_db[run.id] = sm.run

    return sm.run


@app.get("/v1/agents/runs/{run_id}", response_model=AgentRunObject)
async def get_agent_run(run_id: str, request: Request):
    run = agent_runs_db.get(run_id)
    if not run:
        raise HTTPException(
            status_code=404,
            detail=APIErrorResponse(
                error=APIErrorDetails(
                    type="not_found_error",
                    code="agent_run_not_found",
                    message=f"Agent run {run_id} was not found",
                    request_id=request.state.request_id
                )
            ).model_dump()
        )
    return run


# -----------------------------------------------------------------------------
# Sections 31-33, 124: RAG & Hybrid Search API
# -----------------------------------------------------------------------------
@app.post("/v1/rag/ingest")
async def ingest_document(payload: Dict[str, Any], request: Request):
    org_id = request.state.org_id
    raw_text = payload.get("text", "")
    filename = payload.get("filename", "document.txt")
    file_id = payload.get("file_id", f"file_{uuid.uuid4().hex[:12]}")

    chunks = rag_engine.ingest_document(
        organization_id=org_id,
        file_id=file_id,
        raw_text=raw_text,
        filename=filename
    )
    return {"status": "success", "chunks_created": len(chunks), "file_id": file_id}


@app.post("/v1/rag/search")
async def search_rag(payload: Dict[str, Any], request: Request):
    org_id = request.state.org_id
    query = payload.get("query", "")
    top_k = payload.get("top_k", 5)

    results = rag_engine.hybrid_search(
        organization_id=org_id,
        query=query,
        top_k=top_k
    )
    return {"results": results, "query": query}


# -----------------------------------------------------------------------------
# Section 27-30: Memory Management API
# -----------------------------------------------------------------------------
@app.post("/v1/memory")
async def add_memory(payload: Dict[str, Any], request: Request):
    user_id = request.state.user_id
    content = payload.get("content", "")
    scope = payload.get("scope", "user")
    importance = float(payload.get("importance", 0.5))

    mem = memory_engine.add_memory(
        owner_id=user_id,
        scope=scope,
        content=content,
        importance=importance
    )
    return mem


@app.get("/v1/memory")
async def get_memories(request: Request):
    user_id = request.state.user_id
    return memory_engine.export_memories(owner_id=user_id)


# -----------------------------------------------------------------------------
# Section 5, 117: Models Registry API
# -----------------------------------------------------------------------------
@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "alizia-nova",
                "object": "model",
                "description": "Default flagship reasoning model (Section 5.1)",
                "capabilities": ["reasoning", "coding", "research", "agent_workflows"],
                "context_window": 1000000
            },
            {
                "id": "alizia-pulse",
                "object": "model",
                "description": "Fast general-purpose model, low latency (Section 5.2)",
                "capabilities": ["chat", "classification", "summarization", "low_latency"],
                "context_window": 256000
            },
            {
                "id": "alizia-forge",
                "object": "model",
                "description": "Software engineering specialist (Section 5.3)",
                "capabilities": ["repo_reasoning", "debugging", "code_gen", "diff_patching"],
                "context_window": 512000
            },
            {
                "id": "alizia-vision",
                "object": "model",
                "description": "Multimodal visual reasoning specialist (Section 5.4)",
                "capabilities": ["screenshots", "diagrams", "charts", "ocr"],
                "context_window": 256000
            },
            {
                "id": "alizia-embed-v1",
                "object": "model",
                "description": "Embedding model family (Section 5.6)",
                "capabilities": ["semantic_search", "clustering", "rag"],
                "dimensions": 1536
            }
        ]
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "alizia-ai-orchestrator",
        "timestamp": time.time(),
        "version": "1.0.0"
    }
