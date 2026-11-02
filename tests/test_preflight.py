from __future__ import annotations

import json

import httpx
import pytest

from llmbench.backends.base import CompletionRequest
from llmbench.backends.mock import MockBackend
from llmbench.backends.openai_compat import OpenAICompatBackend
from llmbench.config import BackendConfig, BenchConfig, MemoryConfig, QualityConfig
from llmbench.http_errors import error_message_from_payload, format_http_error
from llmbench.preflight import context_overflow_message
from llmbench.runner import run_benchmark
from llmbench.schema import BackendInfo, SamplingParams


def test_extracts_llamacpp_context_exceeded_message() -> None:
    payload = {"code": 500, "message": "Context size has been exceeded.", "type": "server_error"}
    assert error_message_from_payload(payload) == "Context size has been exceeded."
    body = json.dumps({"error": payload}).encode()
    assert format_http_error(500, body) == "HTTP 500: Context size has been exceeded."


def test_skips_when_concurrency_times_prompt_exceeds_n_ctx() -> None:
    cfg = BenchConfig(
        suites=["long_context"],
        concurrency=[4],
        long_context_chars=16_000,
        sampling=SamplingParams(max_tokens=32),
        backend=BackendConfig(type="llamacpp", base_url="http://127.0.0.1:8080", model="m"),
        quality=QualityConfig(enabled=False),
    )
    info = BackendInfo(
        type="llamacpp",
        extra={
            "props": {
                "total_slots": 4,
                "default_generation_settings": {"n_ctx": 8192},
            }
        },
    )
    msg = context_overflow_message(cfg, info, suite="long_context", concurrency=4)
    assert msg is not None
    assert "skipping long_context concurrency=4" in msg
    assert "8192" in msg
    assert context_overflow_message(cfg, info, suite="long_context", concurrency=1) is None


@pytest.mark.asyncio
async def test_sse_error_object_uses_server_message() -> None:
    body = (
        b'data: {"error":{"code":500,"message":"Context size has been exceeded.","type":"server_error"}}\n\n'
        b"data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    backend = OpenAICompatBackend(
        BackendConfig(type="llamacpp", base_url="http://127.0.0.1:8080", model="m"),
        flavor="llamacpp",
        transport=httpx.MockTransport(handler),
    )
    result = await backend.complete(
        CompletionRequest(prompt="hi", sampling=SamplingParams(max_tokens=8, seed=1))
    )
    await backend.aclose()
    assert result.error == "Context size has been exceeded."
    assert result.error_kind == "protocol"


class _PropsBackend(MockBackend):
    async def info(self) -> BackendInfo:
        return BackendInfo(
            type="llamacpp",
            extra={
                "props": {
                    "total_slots": 4,
                    "default_generation_settings": {"n_ctx": 8192},
                }
            },
        )


@pytest.mark.asyncio
async def test_runner_skips_overflowing_long_context_wave() -> None:
    cfg = BenchConfig(
        warmup=0,
        iterations=2,
        concurrency=[4],
        suites=["long_context"],
        long_context_chars=16_000,
        backend=BackendConfig(
            type="llamacpp",
            base_url="http://127.0.0.1:8080",
            model="m",
            mock_output_tokens=2,
            mock_ttft_ns=0,
            mock_inter_token_ns=0,
        ),
        quality=QualityConfig(enabled=False),
        memory=MemoryConfig(enabled=False),
    )
    result = await run_benchmark(cfg, backend=_PropsBackend())
    assert result.trials == []
    assert result.summaries[0].n_ok == 0
    assert any("exceeds server context 8192" in w for w in result.warnings)
    assert any("Context" in n or "exceeds" in n for n in result.summaries[0].notes)
