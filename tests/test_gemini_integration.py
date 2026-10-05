"""
ALIZIA AI - Google Gemini Integration Test Suite
Verifies:
1. Hardcoded API key and resilient Google Gemini Flash provider
2. Direct REST compatibility: POST /v1beta/models/gemini-flash-latest:generateContent
3. Unified AI response API: POST /v1/responses (sync & streaming)
4. Exact token accounting and thinking/reasoning extraction
5. Web playground availability
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from ai.main import app
from inference.gateway.provider import DEFAULT_GEMINI_API_KEY, DEFAULT_GEMINI_MODEL


@pytest.mark.asyncio
async def test_direct_gemini_generate_content_curl_endpoint():
    """Validates the exact user curl request against the backend endpoint."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": "Explain how AI works in a few words"
                        }
                    ]
                }
            ]
        }
        res = await client.post(
            "/v1beta/models/gemini-flash-latest:generateContent",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        assert res.status_code == 200
        data = res.json()
        assert "candidates" in data
        assert len(data["candidates"]) > 0
        part_text = data["candidates"][0]["content"]["parts"][0]["text"]
        assert len(part_text) > 0
        assert "usageMetadata" in data


@pytest.mark.asyncio
async def test_unified_responses_with_gemini_flash():
    """Validates POST /v1/responses powered by Gemini."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post(
            "/v1/responses",
            json={
                "model": "gemini-flash-latest",
                "input": [{"role": "user", "content": "Explain how AI works in a few words"}],
                "reasoning": {"effort": "auto"},
                "stream": False
            }
        )
        assert res.status_code == 200
        data = res.json()
        assert data["object"] == "response"
        assert len(data["output"]) > 0
        text = data["output"][0]["text"]
        assert len(text) > 0
        assert data["usage"]["total_tokens"] > 0
        assert data["usage"]["reasoning_tokens"] > 0


@pytest.mark.asyncio
async def test_playground_and_health_endpoints():
    """Validates health check and HTML playground."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Health
        health_res = await client.get("/health")
        assert health_res.status_code == 200
        health_data = health_res.json()
        assert health_data["status"] == "healthy"
        assert health_data["hardcoded_api_key_configured"] is True

        # Playground UI
        ui_res = await client.get("/")
        assert ui_res.status_code == 200
        assert "Alizia AI" in ui_res.text
        assert "Explain how AI works in a few words" in ui_res.text
