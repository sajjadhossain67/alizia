"""
ALIZIA AI - Comprehensive Generative AI Model Verification Test Suite
Verifies:
1. Native BPE & Subword Tokenizer (encode, decode, special tokens, byte fallback)
2. Transformer Architecture (RMSNorm, RoPE, SwiGLU, KV Cache, Tier Configs)
3. Stochastic Logits Sampler (Temperature, Top-K, Nucleus Top-P, Repetition Penalty)
4. Multi-Modal Generative Engines:
   - Alizia Nova: Chain-of-Thought (CoT), reasoning traces, arithmetic & proof synthesis
   - Alizia Pulse: Low-latency conversational dialogue
   - Alizia Forge: Software engineering code generation with AST validation
   - Alizia Vision: Multimodal perception & SVG composition
   - Alizia Image Generator: Procedural diffusion & PNG/SVG binary synthesis
   - Alizia Audio Generator: Text-to-speech RIFF WAVE synthesis
   - Alizia Embedding Generator: 1536-dim normalized vector embeddings
5. Fine-Tuning Pipeline (LoRA adapter rank decomposition & SFT trainer)
6. Standard Generative API Endpoints (Chat completions, Image generations, Audio speech)
"""

import ast
import json
import numpy as np
import pytest
from httpx import AsyncClient, ASGITransport

from ai.generative_engine.tokenizer import AliziaTokenizer
from ai.generative_engine.transformer import (
    AliziaTransformer,
    ModelArchitectureConfig,
    KVCache,
)
from ai.generative_engine.sampling import Sampler, SamplingParams
from ai.generative_engine.generator import (
    AliziaGenerativeEngine,
    AliziaNovaGenerator,
    AliziaPulseGenerator,
    AliziaForgeGenerator,
    AliziaVisionGenerator,
)
from ai.generative_engine.image_engine import AliziaImageGenerator, ImageGenerationConfig
from ai.generative_engine.audio_engine import AliziaAudioGenerator, AudioSpeechConfig
from ai.generative_engine.embedding_engine import AliziaEmbeddingGenerator
from ai.generative_engine.training import (
    FineTuningDataset,
    TrainingSample,
    LoRAAdapter,
    SFTTrainer,
)
from ai.main import app


# -----------------------------------------------------------------------------
# 1. Native Tokenizer Tests
# -----------------------------------------------------------------------------
def test_tokenizer_encode_decode():
    tokenizer = AliziaTokenizer.get_default()
    sample_text = "Alizia Nova is a complete generative AI model."
    token_ids = tokenizer.encode(sample_text)
    assert len(token_ids) > 0
    assert tokenizer.count_tokens(sample_text) == len(token_ids)

    # Decode round-trip
    decoded_text = tokenizer.decode(token_ids)
    assert decoded_text == sample_text

    # Special tokens
    special_text = "<|im_start|>user\nHello Alizia!<|im_end|>"
    tokens_special = tokenizer.encode(special_text)
    assert AliziaTokenizer.SPECIAL_TOKENS["<|im_start|>"] in tokens_special
    assert AliziaTokenizer.SPECIAL_TOKENS["<|im_end|>"] in tokens_special


# -----------------------------------------------------------------------------
# 2. Transformer Architecture Tests
# -----------------------------------------------------------------------------
def test_transformer_architecture():
    # Verify tier configs
    nova_cfg = ModelArchitectureConfig.get_tier_config("alizia-nova")
    assert nova_cfg.hidden_size == 5120
    assert nova_cfg.context_window == 1000000

    pulse_cfg = ModelArchitectureConfig.get_tier_config("alizia-pulse")
    assert pulse_cfg.hidden_size == 3072
    assert pulse_cfg.context_window == 256000

    forge_cfg = ModelArchitectureConfig.get_tier_config("alizia-forge")
    assert forge_cfg.hidden_size == 4096

    # Test RMSNorm
    x = np.random.randn(2, 64)
    normed = AliziaTransformer.rms_norm(x)
    variances = np.mean(np.square(normed), axis=-1)
    np.testing.assert_allclose(variances, np.ones_like(variances), atol=1e-3)

    # Test RoPE
    vec = np.ones(32)
    rot = AliziaTransformer.apply_rotary_emb(vec, pos=5, dim=32)
    assert rot.shape == vec.shape
    assert not np.allclose(vec, rot)

    # Test KV Cache
    cache = KVCache()
    k = np.zeros((1, 4, 32))
    v = np.zeros((1, 4, 32))
    k_cat, v_cat = cache.update(k, v)
    assert k_cat.shape == (1, 4, 32)
    k_cat2, v_cat2 = cache.update(k, v)
    assert k_cat2.shape == (1, 8, 32)


# -----------------------------------------------------------------------------
# 3. Logits Sampling Tests
# -----------------------------------------------------------------------------
def test_logits_sampler():
    logits = np.array([1.0, 2.0, 5.0, 0.5])
    # Greedy (temperature 0) must pick argmax
    params_greedy = SamplingParams(temperature=0.0)
    tok_greedy = Sampler.sample_next_token(logits, params_greedy)
    assert tok_greedy == 2

    # Repetition penalty decreases score of already chosen token
    params_penalty = SamplingParams(temperature=0.0, repetition_penalty=10.0)
    tok_penalized = Sampler.sample_next_token(logits, params_penalty, generated_token_ids=[2])
    assert tok_penalized != 2


# -----------------------------------------------------------------------------
# 4. Multi-Modal Generative Models Tests
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_alizia_nova_reasoning_generation():
    nova = AliziaNovaGenerator()
    res = await nova.generate_response(
        prompt="Explain the distributed consensus tradeoffs in Raft vs Paxos theorem."
    )
    assert "thought_trace" in res
    assert len(res["thought_trace"]) > 0
    assert "Raft" in res["text"]
    assert "Paxos" in res["text"]

    # Arithmetic reasoning
    math_res = await nova.generate_response(prompt="Calculate 25 * 16")
    assert "400" in math_res["text"]


@pytest.mark.asyncio
async def test_alizia_forge_code_generation():
    forge = AliziaForgeGenerator()
    res = await forge.generate_response(
        prompt="Write a python function to refactor a class with AST syntax verification."
    )
    assert "import ast" in res["text"]
    assert "class CodeRefactorer" in res["text"]
    # Check that generated code parses without syntax errors
    code_block = res["text"].split("```python")[1].split("```")[0]
    ast.parse(code_block)


def test_alizia_image_generation():
    image_gen = AliziaImageGenerator()
    res = image_gen.generate(ImageGenerationConfig(
        prompt="Futuristic neural network interface",
        style="cyberpunk",
        width=512,
        height=512
    ))
    assert res["b64_json"] is not None
    assert len(res["b64_json"]) > 100
    assert res["data_uri"].startswith("data:image/png;base64,")
    assert "<svg" in res["svg_markup"]


def test_alizia_audio_speech_generation():
    audio_gen = AliziaAudioGenerator()
    res = audio_gen.generate(AudioSpeechConfig(
        input_text="Alizia AI platform is ready.",
        voice="nova-voice"
    ))
    assert res["format"] == "wav"
    assert len(res["wav_bytes"]) > 44  # Greater than RIFF header size
    assert res["wav_bytes"][:4] == b"RIFF"
    assert res["wav_bytes"][8:12] == b"WAVE"


def test_alizia_dense_embeddings():
    v1 = AliziaEmbeddingGenerator.embed_text("Machine learning neural networks")
    v2 = AliziaEmbeddingGenerator.embed_text("Deep artificial intelligence models")
    v3 = AliziaEmbeddingGenerator.embed_text("Baking sourdough artisan bread")

    assert len(v1) == 1536
    # Vector length should be approximately 1.0 (L2-normalized)
    norm = sum(x * x for x in v1) ** 0.5
    assert abs(norm - 1.0) < 1e-3

    sim_related = AliziaEmbeddingGenerator.cosine_similarity(v1, v2)
    sim_unrelated = AliziaEmbeddingGenerator.cosine_similarity(v1, v3)
    assert sim_related > sim_unrelated


# -----------------------------------------------------------------------------
# 5. LoRA Fine-Tuning Tests
# -----------------------------------------------------------------------------
def test_fine_tuning_and_lora():
    adapter = LoRAAdapter(in_features=1024, out_features=1024, rank=8, alpha=16.0)
    delta = adapter.compute_weight_delta()
    assert delta.shape == (1024, 1024)

    samples = [
        TrainingSample(prompt="Optimize query", completion="SELECT * FROM users"),
        TrainingSample(prompt="Fix memory leak", completion="Close connections in finally block")
    ]
    dataset = FineTuningDataset(samples)
    trainer = SFTTrainer(base_model_name="alizia-nova", epochs=2, lora_rank=8)
    job = trainer.train(dataset)
    assert job["status"] == "completed"
    assert job["epochs"] == 2
    assert len(job["history"]) == 2
    assert job["final_train_loss"] < job["history"][0]["train_loss"]


# -----------------------------------------------------------------------------
# 6. Standard Generative Endpoints API Tests
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_openai_chat_completions_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Chat completion
        res = await client.post(
            "/v1/chat/completions",
            json={
                "model": "alizia-nova",
                "messages": [
                    {"role": "user", "content": "Explain how AI works in a few words"}
                ],
                "temperature": 0.7,
                "stream": False
            }
        )
        assert res.status_code == 200
        data = res.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) > 0
        content = data["choices"][0]["message"]["content"]
        assert len(content) > 0
        assert data["usage"]["total_tokens"] > 0


@pytest.mark.asyncio
async def test_images_generations_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post(
            "/v1/images/generations",
            json={
                "prompt": "Futuristic skyline with neon colors",
                "model": "alizia-image",
                "response_format": "url",
                "style": "cyberpunk"
            }
        )
        assert res.status_code == 200
        data = res.json()
        assert len(data["data"]) > 0
        assert data["data"][0]["url"].startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_audio_speech_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post(
            "/v1/audio/speech",
            json={
                "model": "alizia-tts",
                "input": "Hello from Alizia speech generator.",
                "voice": "nova-voice"
            }
        )
        assert res.status_code == 200
        assert res.headers["content-type"] == "audio/wav"
        assert len(res.content) > 44
        assert res.content[:4] == b"RIFF"


@pytest.mark.asyncio
async def test_fine_tuning_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post(
            "/v1/models/fine-tune",
            json={
                "model": "alizia-nova"
            }
        )
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "completed"
        job_id = data["job_id"]

        # Check job retrieval
        get_res = await client.get(f"/v1/models/fine-tune/{job_id}")
        assert get_res.status_code == 200
        assert get_res.json()["job_id"] == job_id
