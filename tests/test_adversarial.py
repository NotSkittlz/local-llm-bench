"""Fault injection: the harness should fail loudly, not invent numbers."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from llmbench.backends.base import CompletionRequest
from llmbench.backends.mock import MockBackend, MockFault
from llmbench.backends.openai_compat import OpenAICompatBackend
from llmbench.config import BackendConfig, BenchConfig, QualityConfig
from llmbench.exceptions import (
    BackendConnectionError,
    BackendTimeoutError,
    ConfigError,
    ReportError,
)
from llmbench.export import read_json
from llmbench.runner import run_benchmark
from llmbench.schema import SamplingParams

pytestmark = pytest.mark.adversarial


def _req(**kwargs: object) -> CompletionRequest:
    return CompletionRequest(
        prompt="hello",
        sampling=SamplingParams(max_tokens=8, seed=1, temperature=0.8),
        **kwargs,  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
async def test_closed_port_is_connection_error() -> None:
    backend = OpenAICompatBackend(
        BackendConfig(type="openai_compat", base_url="http://127.0.0.1:1", model="m"),
        connect_s=0.2,
        timeout_s=1.0,
    )
    with pytest.raises(BackendConnectionError, match="could not connect"):
        await backend.complete(_req())
    await backend.aclose()


@pytest.mark.asyncio
async def test_read_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow")

    backend = OpenAICompatBackend(
        BackendConfig(type="openai_compat", base_url="http://127.0.0.1:8080", model="m"),
        timeout_s=0.2,
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(BackendTimeoutError):
        await backend.complete(_req())
    await backend.aclose()


@pytest.mark.asyncio
async def test_interrupted_tcp_stream_keeps_partial_token() -> None:
    server = await asyncio.start_server(_close_after_first_event, "127.0.0.1", 0)
    host, port = server.sockets[0].getsockname()[:2]
    backend = OpenAICompatBackend(
        BackendConfig(type="openai_compat", base_url=f"http://{host}:{port}", model="m"),
        timeout_s=2.0,
        connect_s=0.5,
    )
    try:
        result = await backend.complete(_req(collect_text=True))
    finally:
        backend_close = backend.aclose()
        server.close()
        await server.wait_closed()
        await backend_close
    assert result.first_token_ns is not None
    assert result.output_tokens >= 1
    # Connection drop after the first event: we still recorded what we saw.
    assert result.text == "Hi" or result.error is not None


async def _close_after_first_event(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    await reader.read(4096)
    body = b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n'
    header = (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: text/event-stream\r\n"
        b"Content-Length: " + str(len(body)).encode() + b"\r\n"
        b"Connection: close\r\n"
        b"\r\n"
    )
    writer.write(header + body)
    await writer.drain()
    writer.close()
    await writer.wait_closed()


@pytest.mark.asyncio
async def test_truncated_http_200_is_excluded_from_percentiles() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n'
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    backend = OpenAICompatBackend(
        BackendConfig(type="openai_compat", base_url="http://127.0.0.1:8080", model="m"),
        transport=httpx.MockTransport(handler),
    )
    cfg = BenchConfig(
        warmup=0,
        iterations=3,
        suites=["short_chat"],
        quality=QualityConfig(enabled=False),
        memory={"enabled": False},
        backend=BackendConfig(type="openai_compat", base_url="http://127.0.0.1:8080", model="m"),
    )
    result = await run_benchmark(cfg, backend=backend)
    await backend.aclose()
    assert result.summaries[0].n_ok == 0
    assert result.summaries[0].n_err == 3
    assert all(t.error_kind == "truncated" for t in result.trials)
    assert result.summaries[0].ttft_ms.n == 0
    assert result.summaries[0].decode_tps.n == 0


@pytest.mark.asyncio
async def test_empty_stream_is_error_not_zero_tps() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"data: [DONE]\n\n")

    backend = OpenAICompatBackend(
        BackendConfig(type="openai_compat", base_url="http://127.0.0.1:8080", model="m"),
        transport=httpx.MockTransport(handler),
    )
    result = await backend.complete(_req())
    await backend.aclose()
    assert result.error_kind == "interrupted"
    assert result.first_token_ns is None
    assert result.output_tokens == 0


@pytest.mark.asyncio
async def test_sampling_params_not_rewritten() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        body = b'data: {"choices":[{"delta":{"content":"x"}}]}\n\ndata: [DONE]\n\n'
        return httpx.Response(200, content=body)

    backend = OpenAICompatBackend(
        BackendConfig(type="openai_compat", base_url="http://127.0.0.1:8080", model="m"),
        transport=httpx.MockTransport(handler),
    )
    await backend.complete(_req())
    await backend.aclose()
    assert seen["temperature"] == 0.8
    assert seen["top_p"] == 1.0
    assert seen["seed"] == 1


@pytest.mark.asyncio
async def test_huge_context_is_capped() -> None:
    cfg = BenchConfig(
        long_context_chars=2_000_001,
        suites=["long_context"],
        backend=BackendConfig(type="mock"),
        quality=QualityConfig(enabled=False),
    )
    with pytest.raises(ConfigError, match="capped"):
        await run_benchmark(cfg)


@pytest.mark.asyncio
async def test_concurrency_trial_counts_and_unique_ids() -> None:
    cfg = BenchConfig(
        warmup=1,
        iterations=12,
        concurrency=[4],
        suites=["short_chat"],
        backend=BackendConfig(
            type="mock", mock_output_tokens=4, mock_ttft_ns=0, mock_inter_token_ns=0
        ),
        quality=QualityConfig(enabled=False),
        memory={"enabled": False},
    )
    result = await run_benchmark(cfg)
    measured = [t for t in result.trials if not t.warmup]
    assert len(measured) == 12
    assert len({t.trial_id for t in result.trials}) == len(result.trials)
    assert result.summaries[0].n_ok == 12
    assert all(t.concurrency == 4 for t in measured)


@pytest.mark.asyncio
async def test_mock_interrupt_does_not_count_as_ok() -> None:
    backend = MockBackend(
        BackendConfig(mock_output_tokens=8),
        fault=MockFault(kind="interrupt", after_tokens=1),
    )
    cfg = BenchConfig(
        warmup=0,
        iterations=3,
        suites=["short_chat"],
        quality=QualityConfig(enabled=False),
        memory={"enabled": False},
        backend=BackendConfig(type="mock", mock_ttft_ns=0, mock_inter_token_ns=0),
    )
    result = await run_benchmark(cfg, backend=backend)
    assert result.summaries[0].n_ok == 0
    assert result.summaries[0].n_err == 3
    assert all(t.decode_tps is None or not t.ok for t in result.trials)


def test_corrupt_json_report(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ReportError):
        read_json(path)
    path.write_text('{"schema_version": "9.0.0"}', encoding="utf-8")
    with pytest.raises(ReportError):
        read_json(path)


@pytest.mark.asyncio
async def test_quality_strict_aborts() -> None:
    cfg = BenchConfig(
        backend=BackendConfig(type="mock", mock_ttft_ns=0, mock_inter_token_ns=0),
        quality=QualityConfig(enabled=True, expected_contains="THIS_WILL_NOT_MATCH", strict=True),
        warmup=0,
        iterations=1,
    )
    with pytest.raises(ConfigError, match="quality probe failed"):
        await run_benchmark(cfg)
