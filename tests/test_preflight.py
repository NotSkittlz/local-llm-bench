from __future__ import annotations

import json

import httpx
import pytest

from llmbench.backends.base import CompletionRequest
from llmbench.backends.mock import MockBackend
from llmbench.backends.openai_compat import OpenAICompatBackend
from llmbench.config import BackendConfig, BenchConfig, MemoryConfig, QualityConfig
from llmbench.http_errors import error_message_from_payload, format_http_error
from llmbench.preflight import context_overflow_message, context_queueing_note
from llmbench.report import render_markdown
from llmbench.runner import run_benchmark
from llmbench.schema import BackendInfo, SamplingParams


def _llamacpp_info(
    *, n_ctx: int = 8192, slots: int = 4, version: str | None = "b11193-4e7481175"
) -> BackendInfo:
    return BackendInfo(
        type="llamacpp",
        version=version,
        extra={
            "props": {
                "total_slots": slots,
                "default_generation_settings": {"n_ctx": n_ctx},
            }
        },
    )


def test_extracts_llamacpp_context_exceeded_message() -> None:
    payload = {"code": 500, "message": "Context size has been exceeded.", "type": "server_error"}
    assert error_message_from_payload(payload) == "Context size has been exceeded."
    body = json.dumps({"error": payload}).encode()
    assert format_http_error(500, body) == "HTTP 500: Context size has been exceeded."


def test_parallel_fits_per_slot_n_ctx() -> None:
    cfg = BenchConfig(
        suites=["long_context"],
        concurrency=[4],
        long_context_chars=16_000,
        sampling=SamplingParams(max_tokens=32),
        backend=BackendConfig(type="llamacpp", base_url="http://127.0.0.1:8080", model="m"),
        quality=QualityConfig(enabled=False),
    )
    info = _llamacpp_info(n_ctx=8192, slots=4)
    assert context_overflow_message(cfg, info, suite="long_context", concurrency=4) is None
    assert context_overflow_message(cfg, info, suite="long_context", concurrency=1) is None
    assert context_queueing_note(cfg, info, suite="long_context", concurrency=4) is None


def test_skips_when_one_request_exceeds_per_slot_n_ctx() -> None:
    cfg = BenchConfig(
        suites=["long_context"],
        concurrency=[1],
        long_context_chars=50_000,
        sampling=SamplingParams(max_tokens=32),
        backend=BackendConfig(type="llamacpp", base_url="http://127.0.0.1:8080", model="m"),
        quality=QualityConfig(enabled=False),
    )
    info = _llamacpp_info(n_ctx=8192, slots=4)
    msg = context_overflow_message(cfg, info, suite="long_context", concurrency=1)
    assert msg is not None
    assert "skipping long_context" in msg
    assert "per-slot server context 8192" in msg
    assert "4 x" not in msg


def test_queueing_note_when_concurrency_exceeds_slots() -> None:
    cfg = BenchConfig(
        suites=["short_chat"],
        concurrency=[8],
        backend=BackendConfig(type="llamacpp", base_url="http://127.0.0.1:8080", model="m"),
        quality=QualityConfig(enabled=False),
    )
    info = _llamacpp_info(n_ctx=8192, slots=4)
    assert context_overflow_message(cfg, info, suite="short_chat", concurrency=8) is None
    note = context_queueing_note(cfg, info, suite="short_chat", concurrency=8)
    assert note is not None
    assert "exceeds parallel slots=4" in note
    assert "queue" in note


def test_ollama_num_ctx_is_per_request_not_times_concurrency() -> None:
    cfg = BenchConfig(
        suites=["long_context"],
        concurrency=[4],
        long_context_chars=16_000,
        sampling=SamplingParams(max_tokens=32),
        backend=BackendConfig(type="ollama", base_url="http://127.0.0.1:11434", model="m"),
        model={"context_length": 8192},
        quality=QualityConfig(enabled=False),
    )
    info = BackendInfo(type="ollama", extra={"num_parallel": 2})
    assert context_overflow_message(cfg, info, suite="long_context", concurrency=4) is None
    note = context_queueing_note(cfg, info, suite="long_context", concurrency=4)
    assert note is not None
    assert "slots=2" in note


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
    def __init__(self, n_ctx: int = 8192, slots: int = 4) -> None:
        super().__init__()
        self._n_ctx = n_ctx
        self._slots = slots

    async def info(self) -> BackendInfo:
        return _llamacpp_info(n_ctx=self._n_ctx, slots=self._slots)


@pytest.mark.asyncio
async def test_runner_runs_parallel_long_context_within_slot() -> None:
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
    assert len([t for t in result.trials if not t.warmup]) == 2
    assert result.summaries[0].n_ok == 2
    assert not any("skipping" in w for w in result.warnings)
    assert result.model.context_length == 8192
    assert result.model.context_length_source == "server"
    md = render_markdown([result])
    assert "b11193-4e7481175" in md
    assert "llamacpp None" not in md


@pytest.mark.asyncio
async def test_runner_fills_ollama_context_length_from_show() -> None:
    class _ShowBackend(MockBackend):
        async def info(self) -> BackendInfo:
            return BackendInfo(type="ollama", extra={"num_ctx": 8192, "context_length": 32768})

    cfg = BenchConfig(
        warmup=0,
        iterations=1,
        concurrency=[1],
        suites=["short_chat"],
        backend=BackendConfig(
            type="ollama",
            base_url="http://127.0.0.1:11434",
            model="m",
            mock_output_tokens=2,
            mock_ttft_ns=0,
            mock_inter_token_ns=0,
        ),
        quality=QualityConfig(enabled=False),
        memory=MemoryConfig(enabled=False),
    )
    result = await run_benchmark(cfg, backend=_ShowBackend())
    assert result.model.context_length == 8192
    assert result.model.context_length_source == "server"


@pytest.mark.asyncio
async def test_runner_notes_queueing_when_concurrency_exceeds_slots() -> None:
    cfg = BenchConfig(
        warmup=0,
        iterations=2,
        concurrency=[8],
        suites=["short_chat"],
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
    result = await run_benchmark(cfg, backend=_PropsBackend(slots=4))
    assert len([t for t in result.trials if not t.warmup]) == 2
    assert result.summaries[0].n_ok == 2
    assert any("exceeds parallel slots=4" in w for w in result.warnings)
    assert any("queue" in n for n in result.summaries[0].notes)


@pytest.mark.asyncio
async def test_runner_skips_when_one_prompt_exceeds_slot_ctx() -> None:
    cfg = BenchConfig(
        warmup=0,
        iterations=2,
        concurrency=[4],
        suites=["long_context"],
        long_context_chars=50_000,
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
    assert any("per-slot server context 8192" in w for w in result.warnings)
