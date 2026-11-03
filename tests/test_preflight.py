from __future__ import annotations

import json
from io import StringIO

import httpx
import pytest
from rich.console import Console

import llmbench.cli as cli_mod
from llmbench.backends.base import CompletionRequest
from llmbench.backends.mock import MockBackend
from llmbench.backends.openai_compat import OpenAICompatBackend
from llmbench.cli import _print_summary
from llmbench.config import BackendConfig, BenchConfig, MemoryConfig, QualityConfig
from llmbench.http_errors import error_message_from_payload, format_http_error
from llmbench.preflight import (
    context_overflow_message,
    context_queueing_note,
    server_n_ctx,
    slot_prefill_contention_note,
    unknown_ollama_context_warning,
)
from llmbench.report import render_html, render_markdown
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
            return BackendInfo(type="ollama", extra={"num_ctx": 8192, "context_length_max": 32768})

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
    assert result.model.context_length_source == "ollama_num_ctx"
    assert result.model.context_length_max == 32768


@pytest.mark.asyncio
async def test_runner_records_ollama_ps_runtime_and_model_max() -> None:
    class _PsBackend(MockBackend):
        async def info(self) -> BackendInfo:
            return BackendInfo(
                type="ollama",
                extra={"ps_context_length": 32768, "context_length_max": 262144},
            )

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
    result = await run_benchmark(cfg, backend=_PsBackend())
    assert result.model.context_length == 32768
    assert result.model.context_length_source == "ollama_ps"
    assert result.model.context_length_max == 262144
    md = render_markdown([result])
    assert "/api/ps" in md
    assert "model max 262144" in md
    assert any("context_length_max=262144" in n for n in result.notes)


def test_ollama_uses_ps_runtime_not_architecture_max() -> None:
    cfg = BenchConfig(
        suites=["long_context"],
        concurrency=[1],
        long_context_chars=16_000,
        sampling=SamplingParams(max_tokens=32),
        backend=BackendConfig(type="ollama", base_url="http://127.0.0.1:11434", model="m"),
        quality=QualityConfig(enabled=False),
    )
    info = BackendInfo(
        type="ollama",
        extra={"ps_context_length": 32768, "context_length_max": 262144},
    )
    n_ctx, _slots, source = server_n_ctx(info, cfg)
    assert n_ctx == 32768
    assert source == "ollama_ps"
    assert context_overflow_message(cfg, info, suite="long_context", concurrency=1) is None


def test_ollama_preflight_skips_against_runtime_not_model_max() -> None:
    cfg = BenchConfig(
        suites=["long_context"],
        concurrency=[1],
        long_context_chars=200_000,
        sampling=SamplingParams(max_tokens=32),
        backend=BackendConfig(type="ollama", base_url="http://127.0.0.1:11434", model="m"),
        quality=QualityConfig(enabled=False),
    )
    info = BackendInfo(
        type="ollama",
        extra={"ps_context_length": 32768, "context_length_max": 262144},
    )
    msg = context_overflow_message(cfg, info, suite="long_context", concurrency=1)
    assert msg is not None
    assert "32768" in msg
    assert "262144" not in msg
    assert "api/ps" in msg


def test_ollama_falls_back_to_env_not_assumed_default() -> None:
    cfg = BenchConfig(
        suites=["long_context"],
        concurrency=[1],
        long_context_chars=50_000,
        sampling=SamplingParams(max_tokens=32),
        backend=BackendConfig(type="ollama", base_url="http://127.0.0.1:11434", model="m"),
        quality=QualityConfig(enabled=False),
    )
    env_info = BackendInfo(
        type="ollama", extra={"ollama_context_length_env": 32768, "context_length_max": 262144}
    )
    n_ctx, _slots, source = server_n_ctx(env_info, cfg)
    assert n_ctx == 32768
    assert source == "ollama_env"
    empty = BackendInfo(type="ollama", extra={"context_length_max": 262144})
    n_ctx, _slots, source = server_n_ctx(empty, cfg)
    assert n_ctx is None
    assert unknown_ollama_context_warning(empty, cfg) is not None
    assert context_overflow_message(cfg, empty, suite="long_context", concurrency=1) is None


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


def test_contention_note_only_for_llamacpp_parallel_long_context() -> None:
    llama = BenchConfig(
        backend=BackendConfig(type="llamacpp", base_url="http://127.0.0.1:8080", model="m"),
        quality=QualityConfig(enabled=False),
    )
    note = slot_prefill_contention_note(llama, suite="long_context", concurrency=4)
    assert note is not None
    assert "share the GPU" in note
    assert "measurement error" in note
    assert slot_prefill_contention_note(llama, suite="long_context", concurrency=1) is None
    assert slot_prefill_contention_note(llama, suite="short_chat", concurrency=4) is None
    ollama = BenchConfig(
        backend=BackendConfig(type="ollama", base_url="http://127.0.0.1:11434", model="m"),
        quality=QualityConfig(enabled=False),
    )
    assert slot_prefill_contention_note(ollama, suite="long_context", concurrency=4) is None


@pytest.mark.asyncio
async def test_runner_notes_llamacpp_parallel_long_context_contention() -> None:
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
    assert any("share the GPU" in n for n in result.summaries[0].notes)
    html = render_html([result])
    assert "share the GPU" in html
    assert "not a measurement error" in html
    md = render_markdown([result])
    assert "share the GPU" in md

    buf = StringIO()
    previous = cli_mod.console
    cli_mod.console = Console(file=buf, width=140, force_terminal=False, no_color=True)
    try:
        _print_summary(result)
    finally:
        cli_mod.console = previous
    assert "share the GPU" in buf.getvalue()

    cfg.concurrency = [1]
    solo = await run_benchmark(cfg, backend=_PropsBackend())
    assert not any("share the GPU" in n for s in solo.summaries for n in s.notes)


class _ColdThenHotOllama(MockBackend):
    """info() reports /api/ps only after complete() has loaded the model."""

    async def info(self) -> BackendInfo:
        extra: dict = {"context_length_max": 131072}
        if self.calls > 0:
            extra["ps_context_length"] = 32768
        return BackendInfo(type="ollama", extra=extra)


@pytest.mark.asyncio
async def test_runner_rereads_ollama_ps_after_quality_probe() -> None:
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
        quality=QualityConfig(enabled=True, max_tokens=8),
        memory=MemoryConfig(enabled=False),
    )
    result = await run_benchmark(cfg, backend=_ColdThenHotOllama())
    assert result.model.context_length == 32768
    assert result.model.context_length_source == "ollama_ps"
    assert not any("4096" in w for w in result.warnings)
    assert not any("unknown" in w.lower() for w in result.warnings)


@pytest.mark.asyncio
async def test_runner_does_not_skip_long_context_on_unknown_ollama_ctx() -> None:
    class _NeverPs(MockBackend):
        async def info(self) -> BackendInfo:
            return BackendInfo(type="ollama", extra={"context_length_max": 131072})

    cfg = BenchConfig(
        warmup=0,
        iterations=1,
        concurrency=[1],
        suites=["long_context"],
        long_context_chars=50_000,
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
    result = await run_benchmark(cfg, backend=_NeverPs())
    assert result.summaries[0].n_ok == 1
    assert result.model.context_length is None
    assert any("runtime context is unknown" in w for w in result.warnings)
    assert not any(w.startswith("skipping") for w in result.warnings)
