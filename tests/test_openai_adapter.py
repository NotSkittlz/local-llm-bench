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
async def test_connection_refused_message() -> None:
    backend = OpenAICompatBackend(
        BackendConfig(type="openai_compat", base_url="http://127.0.0.1:1", model="m"),
        connect_s=0.3,
        timeout_s=1.0,
    )
    with pytest.raises(BackendConnectionError, match="could not connect"):
        await backend.complete(_req())
    await backend.aclose()
