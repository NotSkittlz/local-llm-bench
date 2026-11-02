from __future__ import annotations

import json

import httpx
import pytest

from llmbench.backends.base import CompletionRequest
from llmbench.backends.openai_compat import OpenAICompatBackend
from llmbench.config import BackendConfig
from llmbench.exceptions import BackendConnectionError
from llmbench.schema import SamplingParams


def _req() -> CompletionRequest:
    return CompletionRequest(
        prompt="Say hi",
        sampling=SamplingParams(max_tokens=8, seed=1),
        collect_text=True,
    )


def _sse(*objs: dict) -> bytes:
    parts = [f"data: {json.dumps(o)}\n\n".encode() for o in objs]
    parts.append(b"data: [DONE]\n\n")
    return b"".join(parts)


@pytest.mark.asyncio
async def test_openai_stream_ttft_and_usage() -> None:
    body = _sse(
        {"choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo"}}]},
        {"usage": {"prompt_tokens": 4, "completion_tokens": 2}, "choices": [{"delta": {}}]},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/v1/chat/completions")
        payload = json.loads(request.content)
        assert payload["stream"] is True
        assert payload["temperature"] == 0.0
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    backend = OpenAICompatBackend(
        BackendConfig(type="openai_compat", base_url="http://127.0.0.1:8080", model="m"),
        transport=httpx.MockTransport(handler),
    )
    result = await backend.complete(_req())
    await backend.aclose()
    assert result.error is None
    assert result.text == "Hello"
    assert result.output_tokens == 2
    assert result.prompt_tokens == 4
    assert result.token_source == "server_usage"
    assert result.first_token_ns is not None


@pytest.mark.asyncio
async def test_openai_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b'{"error":"boom"}')

    backend = OpenAICompatBackend(
        BackendConfig(type="llamacpp", base_url="http://127.0.0.1:8080", model="m"),
        flavor="llamacpp",
        transport=httpx.MockTransport(handler),
    )
    result = await backend.complete(_req())
    await backend.aclose()
    assert result.error_kind == "http"
    assert "500" in (result.error or "")


@pytest.mark.asyncio
async def test_openai_bad_json_event() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"data: not-json\n\n",
            headers={"content-type": "text/event-stream"},
        )

    backend = OpenAICompatBackend(
        BackendConfig(type="vllm", base_url="http://127.0.0.1:8000", model="m"),
        flavor="vllm",
        transport=httpx.MockTransport(handler),
    )
    result = await backend.complete(_req())
    await backend.aclose()
    assert result.error_kind == "protocol"


@pytest.mark.asyncio
async def test_truncated_stream_without_terminal_is_not_ok() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n'
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    backend = OpenAICompatBackend(
        BackendConfig(type="openai_compat", base_url="http://127.0.0.1:8080", model="m"),
        transport=httpx.MockTransport(handler),
    )
    result = await backend.complete(_req())
    await backend.aclose()
    assert result.error_kind == "truncated"
    assert result.first_token_ns is not None
    assert result.output_tokens >= 1


@pytest.mark.asyncio
async def test_finish_reason_without_done_is_terminal() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = b'data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":"stop"}]}\n\n'
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    backend = OpenAICompatBackend(
        BackendConfig(type="openai_compat", base_url="http://127.0.0.1:8080", model="m"),
        transport=httpx.MockTransport(handler),
    )
    result = await backend.complete(_req())
    await backend.aclose()
    assert result.error is None


@pytest.mark.asyncio
async def test_llamacpp_cache_n_is_not_counted_as_prefill() -> None:
    from llmbench.metrics import fill_trial_derived, trial_prefill_tps
    from llmbench.schema import Trial

    body = _sse(
        {"choices": [{"delta": {"content": "x"}}]},
        {
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "timings": {
                "prompt_n": 1,
                "prompt_ms": 2.0,
                "predicted_n": 1,
                "predicted_ms": 8.0,
                "cache_n": 799,
            },
            "usage": {"prompt_tokens": 800, "completion_tokens": 1},
        },
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    backend = OpenAICompatBackend(
        BackendConfig(type="llamacpp", base_url="http://127.0.0.1:8080", model="m"),
        flavor="llamacpp",
        transport=httpx.MockTransport(handler),
    )
    result = await backend.complete(_req())
    await backend.aclose()
    assert result.server_timings is not None
    assert result.server_timings.cached_prompt_tokens == 799
    assert result.server_timings.uncached_prompt_tokens == 1
    assert result.server_timings.cache_hit is True
    assert trial_prefill_tps(result.server_timings) == 500.0  # 1 token / 0.002s
    trial = Trial(
        trial_id="t",
        suite="long_context",
        prompt_id="long-01",
        concurrency=1,
        ok=True,
        prompt_chars=10,
        prompt_tokens=result.prompt_tokens,
        start_ns=result.start_ns,
        first_token_ns=result.first_token_ns,
        last_token_ns=result.last_token_ns,
        end_ns=result.end_ns,
        server_timings=result.server_timings,
    )
    fill_trial_derived(trial)
    assert trial.prefill_source == "server_uncached"
    assert trial.prefill_tps == 500.0
    assert trial.prefill_tps_proxy is None or trial.prefill_tps_proxy != trial.prefill_tps


@pytest.mark.asyncio
async def test_connection_refused_message() -> None:
    backend = OpenAICompatBackend(
        BackendConfig(type="openai_compat", base_url="http://127.0.0.1:1", model="m"),
        connect_s=0.3,
        timeout_s=1.0,
    )
    with pytest.raises(BackendConnectionError, match="could not connect"):
        await backend.complete(_req())
    await backend.aclose()
