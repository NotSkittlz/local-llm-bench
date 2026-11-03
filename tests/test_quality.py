from __future__ import annotations

import hashlib
import json

import httpx
import pytest

from llmbench.backends.mock import MockBackend
from llmbench.backends.openai_compat import OpenAICompatBackend
from llmbench.config import BackendConfig, BenchConfig, QualityConfig
from llmbench.quality import QUALITY_RETRY_MAX_TOKENS, run_quality_probe


@pytest.mark.asyncio
async def test_quality_probe_passes_on_mock_paris() -> None:
    cfg = BenchConfig(backend=BackendConfig(type="mock", mock_ttft_ns=0, mock_inter_token_ns=0))
    backend = MockBackend(cfg.backend)
    probe = await run_quality_probe(backend, cfg)
    assert probe.enabled is True
    assert probe.ok is True
    assert probe.contains_hit is True
    assert probe.output_text is not None
    assert probe.output_text.startswith("Paris")
    assert probe.output_sha256 == hashlib.sha256(probe.output_text.encode()).hexdigest()
    await backend.aclose()


@pytest.mark.asyncio
async def test_quality_probe_hash_mismatch() -> None:
    cfg = BenchConfig(
        backend=BackendConfig(type="mock", mock_ttft_ns=0, mock_inter_token_ns=0),
        quality=QualityConfig(expected_sha256="deadbeef" * 8, expected_contains=None),
    )
    backend = MockBackend(cfg.backend)
    probe = await run_quality_probe(backend, cfg)
    assert probe.hash_match is False
    assert probe.ok is False
    await backend.aclose()


@pytest.mark.asyncio
async def test_disabled_probe() -> None:
    cfg = BenchConfig(quality=QualityConfig(enabled=False))
    backend = MockBackend(cfg.backend)
    probe = await run_quality_probe(backend, cfg)
    assert probe.enabled is False
    assert probe.ok is None
    await backend.aclose()


@pytest.mark.asyncio
async def test_quality_probe_sends_thinking_off_and_scores_content() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        seen.append(payload)
        kwargs = payload.get("chat_template_kwargs") or {}
        if kwargs.get("enable_thinking") is False:
            body = (
                b'data: {"choices":[{"delta":{"content":"Paris"},"finish_reason":"stop"}]}\n\n'
                b"data: [DONE]\n\n"
            )
        else:
            body = (
                b'data: {"choices":[{"delta":{"reasoning_content":"plan"}}]}\n\n'
                b'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\n'
                b"data: [DONE]\n\n"
            )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    backend = OpenAICompatBackend(
        BackendConfig(type="llamacpp", base_url="http://127.0.0.1:8080", model="m"),
        flavor="llamacpp",
        transport=httpx.MockTransport(handler),
    )
    cfg = BenchConfig(
        backend=BackendConfig(type="llamacpp", base_url="http://127.0.0.1:8080", model="m"),
        quality=QualityConfig(max_tokens=16),
    )
    probe = await run_quality_probe(backend, cfg)
    await backend.aclose()
    assert seen
    assert seen[0]["chat_template_kwargs"]["enable_thinking"] is False
    assert probe.ok is True
    assert probe.output_text == "Paris"
    assert probe.contains_hit is True
    assert "thinking=off" in " ".join(probe.notes)


@pytest.mark.asyncio
async def test_quality_probe_retries_with_larger_budget_when_thinking_eats_it() -> None:
    seen_max: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        seen_max.append(int(payload["max_tokens"]))
        if payload["max_tokens"] >= QUALITY_RETRY_MAX_TOKENS:
            body = (
                b'data: {"choices":[{"delta":{"content":"Paris"},"finish_reason":"stop"}]}\n\n'
                b"data: [DONE]\n\n"
            )
        else:
            # thinking=off ignored: 16-token budget is all reasoning.
            body = (
                b'data: {"choices":[{"delta":{"reasoning_content":"plan"}}]}\n\n'
                b'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\n'
                b"data: [DONE]\n\n"
            )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    backend = OpenAICompatBackend(
        BackendConfig(type="llamacpp", base_url="http://127.0.0.1:8080", model="m"),
        flavor="llamacpp",
        transport=httpx.MockTransport(handler),
    )
    cfg = BenchConfig(
        backend=BackendConfig(type="llamacpp", base_url="http://127.0.0.1:8080", model="m"),
        quality=QualityConfig(max_tokens=16),
    )
    probe = await run_quality_probe(backend, cfg)
    await backend.aclose()
    assert seen_max == [16, QUALITY_RETRY_MAX_TOKENS]
    assert probe.ok is True
    assert probe.output_text == "Paris"
    assert any("retrying" in n for n in probe.notes)
