"""
Alizia AI Platform - Sprint 2 Phase 1: Proof Mode Test Suite
Tests atomic claim extraction, source grounding, sandbox assertion verification,
and backwards compatibility for /v1/responses.
"""

import pytest
import asyncio
from packages.schemas.models import (
    CreateResponseRequest,
    ResponseObject,
    MessageItem,
    MessageRole,
    ProofSource,
    ClaimVerificationStatus,
    ProofObject,
)
from ai.proof_engine.engine import ProofEngine
from ai.orchestrator.service import AIOrchestrator


@pytest.mark.asyncio
async def test_extract_atomic_claims():
    """Verify atomic claim extraction separates discrete assertions and removes conversational filler."""
    text = (
        "Here is the verified data for your report:\n"
        "Alizia Nova was launched with 128k context window.\n"
        "The system achieves a 99.4% accuracy rate on deterministic arithmetic tests.\n"
        "In conclusion, let me know if you need anything else."
    )
    claims = ProofEngine.extract_atomic_claims(text)
    assert len(claims) >= 2
    # Verify conversational filler was dropped
    assert not any("here is the verified data" in c.lower() for c in claims)
    assert not any("in conclusion" in c.lower() for c in claims)
    # Verify core claims were captured
    assert any("128k context" in c for c in claims)
    assert any("99.4%" in c for c in claims)


@pytest.mark.asyncio
async def test_source_grounding_positive_and_negative():
    """Verify claim grounding matches valid sources and flags unsupported assertions."""
    sources = [
        ProofSource(
            id="src_1",
            title="Alizia Release Notes",
            url="https://alizia.ai/releases/nova",
            snippet="Alizia Nova was launched with 128k context window and dual-engine architecture.",
            domain="alizia.ai",
            relevance_score=0.98
        )
    ]

    # Grounded claim
    valid_claim = "Alizia Nova was launched with 128k context window."
    status, conf, spans, reason = ProofEngine.match_source_grounding(valid_claim, sources)
    assert status == ClaimVerificationStatus.VERIFIED
    assert conf >= 0.85
    assert len(spans) > 0
    assert spans[0]["source_id"] == "src_1"

    # Ungrounded claim (different number)
    unsupported_claim = "Alizia Nova was launched with 1024k context window."
    status_bad, conf_bad, spans_bad, _ = ProofEngine.match_source_grounding(unsupported_claim, sources)
    assert status_bad == ClaimVerificationStatus.UNSUPPORTED
    assert conf_bad <= 0.50


@pytest.mark.asyncio
async def test_sandbox_assertion_execution():
    """Verify the proof engine generates and executes sandbox assertions for arithmetic / logic."""
    math_text = (
        "The optimal cluster size is calculated as follows:\n"
        "15 * 24 = 360 total compute units.\n"
        "Therefore, 360 units provide sufficient throughput."
    )
    claims = ProofEngine.extract_atomic_claims(math_text)
    tests = await ProofEngine.run_sandbox_assertions(math_text, claims)

    assert len(tests) >= 1
    # Check the executed assertion
    assert any("assert abs((15 * 24) - (360)) < 1e-6" in t.code_or_assertion for t in tests)
    # Check it passed in the sandbox
    assert tests[0].passed is True
    assert tests[0].execution_time_ms > 0


@pytest.mark.asyncio
async def test_end_to_end_proof_verification():
    """Verify ProofEngine.verify_response produces a valid ProofObject."""
    req = CreateResponseRequest(
        model="alizia-nova",
        input=[MessageItem(role=MessageRole.USER, content="Calculate 12 * 12 and state the context size.")],
        proof=True
    )

    resp = ResponseObject(
        model="alizia-nova",
        output=[{
            "type": "output_text",
            "text": "12 * 12 = 144. Alizia platform provides robust inference."
        }],
        citations=[{
            "title": "Alizia Architecture",
            "url": "https://alizia.ai/docs",
            "snippet": "Alizia platform provides robust inference across multiple models.",
            "relevance": 0.95
        }]
    )

    proof = await ProofEngine.verify_response(resp, req)
    assert isinstance(proof, ProofObject)
    assert proof.id.startswith("prf_")
    assert len(proof.claims) >= 1
    assert proof.verifier_verdict in ("PASS", "PARTIAL")
    assert proof.confidence >= 0.70
    assert proof.verification_latency_ms > 0
    assert len(proof.sources) >= 1


@pytest.mark.asyncio
async def test_orchestrator_proof_mode_backwards_compatibility():
    """Verify that when proof=False, proof is None (frozen contract preserved). When proof=True, proof object is returned."""
    orchestrator = AIOrchestrator()

    # 1. Standard request (Proof Mode omitted / default False)
    req_std = CreateResponseRequest(
        model="alizia-nova",
        input=[MessageItem(role=MessageRole.USER, content="Hello Alizia!")],
        proof=False
    )
    resp_std = await orchestrator.execute_request(req_std)
    assert resp_std.proof is None
    assert resp_std.status == "completed"

    # 2. Proof Mode request (proof=True)
    req_proof = CreateResponseRequest(
        model="alizia-nova",
        input=[MessageItem(role=MessageRole.USER, content="Calculate 5 * 20 = 100.")],
        proof=True
    )
    resp_proof = await orchestrator.execute_request(req_proof)
    assert resp_proof.proof is not None
    assert isinstance(resp_proof.proof, ProofObject)
    assert resp_proof.proof.verifier_verdict in ("PASS", "PARTIAL")
    assert len(resp_proof.proof.claims) >= 1
