# ALIZIA AI - Production Backend Platform

> **Frontier Multimodal AI and Agent Platform Backend**  
> *Production Architecture Specification v1.0 (Sections 1-200)*

---

## 1. System Overview

Alizia is an AI operating system designed around the core philosophy:
$$\text{User Intent} \longrightarrow \text{Reasoning} \longrightarrow \text{Planning} \longrightarrow \text{Tools} \longrightarrow \text{Execution} \longrightarrow \text{Verification} \longrightarrow \text{Result}$$

### Service Layout (PRD Section 158)

```
alizia-backend/
├── apps/
│   ├── api-gateway/            # Global API Gateway (Fastify / TypeScript) - WAF, Auth, Rate Limits, Reverse Proxy
│   ├── account-service/        # Organization, User, Workspace, API Key system (alz_live_*, alz_test_*)
│   └── billing-service/        # Usage ledger, quotas, and pricing engine
├── ai/
│   ├── main.py                 # FastAPI production server (POST /v1/responses, POST /v1/embeddings, /v1/agents)
│   ├── orchestrator/           # Core Request Lifecycle Orchestrator (Section 199, steps 1-27)
│   ├── model_router/           # Semantic Router & Adaptive Intelligence (Section 18, 192)
│   ├── prompt_engine/          # Centralized PromptCompiler & Versioned PromptRegistry (Section 22-24)
│   ├── memory_engine/          # 4-tier Memory Platform: Session, User, Workspace, Agent (Section 27-30)
│   ├── rag_engine/             # Hybrid Search: BM25 + Vector + Reranking with tenant isolation (Section 31-33, 124)
│   └── safety_engine/          # Multi-layer safety, secrets redaction, and prompt injection defense (Section 71-77)
├── agents/
│   ├── runtime/                # Agent State Machine: QUEUED -> PLANNING -> RUNNING -> VERIFYING -> COMPLETED (Section 41-44)
│   ├── planner/                # Objective decomposition and dynamic plan revision (Section 42)
│   ├── verifier/               # Verification Engine: syntax parsing, execution checks, Verifiable AI (Section 130, 191)
│   └── tool_manager/           # Tool execution protocol, scopes, R0-R4 risk evaluation, confirmation tokens (Section 48, 74-75)
├── tools/
│   └── executor.py             # First-party tools (web.search, code.execute in isolated sandbox, file I/O)
├── inference/
│   └── gateway/                # Unified ModelProvider abstraction (AliziaNativeProvider, FallbackProvider)
├── platform/
│   └── db/
│       └── schema.sql          # PostgreSQL DDL with pgvector, multi-tenant isolation, and indexes
├── packages/
│   └── schemas/                # Shared Pydantic models & TypeScript type definitions
└── tests/
    └── test_alizia_platform.py # Automated test suite validating all core PRD requirements
```

---

## 2. Alizia Model Family (Section 5)

| Model Identifier | Specialization | Capabilities | Context Window |
|---|---|---|---|
| `alizia-nova` | Default Flagship Reasoning | Complex reasoning, architecture, math, agent workflows | 1,000,000 tokens |
| `alizia-pulse` | High-Speed General Purpose | Low latency chat, summarization, extraction (TTFT < 400ms) | 256,000 tokens |
| `alizia-forge` | Software Engineering Specialist| Repository reasoning, AST analysis, debugging, structured diffs | 512,000 tokens |
| `alizia-vision` | Multimodal Specialist | Screenshots, UI analysis, charts, diagram reasoning | 256,000 tokens |
| `alizia-embed-v1`| Vector Embeddings | Semantic search, RAG, clustering, memory ranking (1536-dim) | 8,192 tokens |

---

## 3. Quickstart & Local Execution

### Prerequisites
- Python 3.12+
- Node.js 22+ & npm 10+
- Docker & Docker Compose (optional for full container stack)

### 1. Python AI Orchestrator & Engine
```bash
# Activate virtual environment
.\.venv\Scripts\activate

# Run tests
pytest -v tests/

# Start FastAPI server on port 8000
uvicorn ai.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. TypeScript Services
```bash
# Install workspace dependencies
npm install

# Run Global API Gateway (port 3000)
npm run dev:gateway

# Run Account Service (port 3001)
npm run dev:accounts

# Run Billing Service (port 3002)
npm run dev:billing
```

### 3. Docker Compose (Full Stack)
```bash
docker compose up -d
```

---

## 4. API Reference

### Unified Responses API (`POST /v1/responses`)
```bash
curl -X POST http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer alz_live_secret1234567890abcdef" \
  -d '{
    "model": "alizia-nova",
    "input": [
      {
        "role": "user",
        "content": "Analyze the concurrency bottlenecks in this system."
      }
    ],
    "reasoning": { "effort": "high" },
    "stream": false
  }'
```

### Proof Mode Unified Responses (`POST /v1/responses` with `"proof": true`)
```bash
curl -X POST http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "alizia-nova",
    "input": [
      { "role": "user", "content": "Calculate 15 * 24 and explain the architecture of Alizia Nova." }
    ],
    "proof": true,
    "stream": false
  }'
```

**Additive Proof Response Envelope**:
```json
{
  "id": "resp_098f6bcd4621d373ca",
  "object": "response",
  "model": "alizia-nova",
  "status": "completed",
  "output": [
    { "type": "output_text", "text": "15 * 24 = 360. Alizia Nova provides 1,000,000 context reasoning." }
  ],
  "proof": {
    "id": "prf_8a7c2e1b",
    "confidence": 0.98,
    "verifier_verdict": "PASS",
    "unsupported_claims_count": 0,
    "verification_latency_ms": 38.5,
    "claims": [
      {
        "id": "clm_1",
        "claim_text": "15 * 24 = 360",
        "status": "verified",
        "confidence": 1.0,
        "tests": ["tst_1"],
        "reasoning": "Algorithmically verified via Python Sandbox test."
      }
    ],
    "sources": [
      {
        "id": "src_1",
        "title": "Alizia Architecture Specification",
        "url": "https://docs.alizia.ai/models/nova",
        "snippet": "alizia-nova — Flagship reasoning engine with 1,000,000 token context window.",
        "relevance_score": 0.98
      }
    ],
    "tests": [
      {
        "id": "tst_1",
        "test_type": "python_sandbox",
        "code_or_assertion": "assert abs((15 * 24) - (360)) < 1e-6",
        "passed": true,
        "output": "Assertion passed cleanly.",
        "execution_time_ms": 1.6
      }
    ]
  }
}
```

### Server-Sent Events (SSE) Streaming
```bash
curl -N -X POST http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "alizia-pulse",
    "input": [{ "role": "user", "content": "Tell me about Alizia AI." }],
    "proof": true,
    "stream": true
  }'
```
*Emits `event: response.output_text.delta` chunks followed by `event: response.proof` and `event: response.completed`.*

### Verifiable Agent Execution (`POST /v1/agents/runs`)
```bash
curl -X POST http://localhost:8000/v1/agents/runs \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "Fix failing authentication unit tests and verify execution proof",
    "max_steps": 50,
    "token_budget": 500000
  }'
```

---

## 5. Evaluation Harness & Continuous Benchmarks (`ai/evals/`)

The platform contains a dedicated production evaluation harness benchmarking Alizia against Google Gemini:

- **12 Core Evaluation Suites**:
  1. `knowledge` (MMLU-Pro / GPQA style)
  2. `math` (GSM8k & Olympiad proof verification)
  3. `code` (HumanEval & AST parsing)
  4. `instruction` (IFEval strict formatting constraints)
  5. `safety` (HarmBench / Refusal robustness)
  6. `multilingual` (Translation and cultural nuance)
  7. `factuality` (Hallucination detection and citation recall)
  8. `tool_use` (BFCL multi-turn tool calling)
  9. `long_context` (Needle-in-a-Haystack at 128k - 1M tokens)
  10. `multimodal` (Chart, UI, and visual reasoning)
  11. `retrieval` (Dense & hybrid passage ranking)
  12. `writing` (Stylistic voice and rubric synthesis)
- **Scorers & Debiasing**:
  - Deterministic exact-match and math tolerance checks.
  - LLM Judge with position-swap debiasing `(A, B)` and `(B, A)` to eliminate positional bias.
  - Pairwise Elo arena with 95% bootstrap confidence intervals.
  - Dataset contamination detection (canary GUIDs & 8-gram overlap).
- **Execution & CI Gate**:
  ```bash
  # Run evaluations via CLI
  python -m ai.evals.cli run --model alizia-nova --limit 5
  
  # Run regression gate with baseline comparison
  python -m ai.evals.cli gate --baseline docs/evals/baseline.md --candidate docs/evals/candidate.md
  ```

---

## 6. Security & Verification Guarantees

1. **Proof Mode (Section 191 & Sprint 2)**: Decomposes answers into atomic claims, grounded across cited source spans and verified via ephemeral Python sandbox assertion executions.
2. **Multi-Layer Safety & Secrets Redaction (Section 71-77)**: Automatically sanitizes sensitive keys (`alz_live_*`, `sk-*`, JWTs, private keys) and enforces trust levels (`trusted`, `authorized`, `untrusted_data`) to prevent indirect prompt injection.
3. **Action Risk Engine (Section 74-75)**: Actions are ranked from R0 (informational) to R4 (destructive). Sensitive actions require cryptographic confirmation tokens before execution.
4. **Strict Tenant Isolation (Section 78, 124)**: Database queries and vector retrieval enforce organization boundaries (`organization_id`) before vector similarity computation.

