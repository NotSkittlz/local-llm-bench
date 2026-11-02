from __future__ import annotations

import json

import httpx
import pytest

from llmbench.backends.base import CompletionRequest
from llmbench.backends.ollama import OllamaBackend
from llmbench.config import BackendConfig
from llmbench.schema import SamplingParams


def _req() -> CompletionRequest:
    return CompletionRequest(
        prompt="hi",
        sampling=SamplingParams(max_tokens=4, seed=1),
        collect_text=True,
    )


@pytest.mark.asyncio
async def test_ollama_ndjson_and_server_timings() -> None:
    lines = [
        {"message": {"content": "Pa"}, "done": False},
        {"message": {"content": "ris"}, "done": False},
        {
            "message": {"content": ""},
            "done": True,
            "prompt_eval_count": 8,
            "prompt_eval_duration": 20_000_000,
            "eval_count": 2,
            "eval_duration": 10_000_000,
            "total_duration": 40_000_000,
        },
    ]
    body = b"".join(json.dumps(x).encode() + b"\n" for x in lines)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/api/chat")
        payload = json.loads(request.content)
        assert payload["options"]["temperature"] == 0.0
        return httpx.Response(200, content=body)

    backend = OllamaBackend(
        BackendConfig(type="ollama", base_url="http://127.0.0.1:11434", model="llama"),
        transport=httpx.MockTransport(handler),
    )
    result = await backend.complete(_req())
    await backend.aclose()
    assert result.error is None
    assert result.text == "Paris"
    assert result.output_tokens == 2
    assert result.prompt_tokens == 8
    assert result.server_timings is not None
    assert result.server_timings.prompt_ms == 20.0
    assert result.token_source == "server_usage"
    # No cached count → uncached breakdown unknown → prefill must stay null.
    from llmbench.metrics import trial_prefill_tps

    assert result.server_timings.uncached_prompt_tokens is None
    assert trial_prefill_tps(result.server_timings) is None


@pytest.mark.asyncio
async def test_ollama_truncated_without_done_is_not_ok() -> None:
    body = json.dumps({"message": {"content": "Hi"}, "done": False}).encode() + b"\n"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    backend = OllamaBackend(
        BackendConfig(type="ollama", base_url="http://127.0.0.1:11434", model="llama"),
        transport=httpx.MockTransport(handler),
    )
    result = await backend.complete(_req())
    await backend.aclose()
    assert result.error_kind == "truncated"


@pytest.mark.asyncio
async def test_ollama_cached_tokens_are_excluded_from_prefill() -> None:
    from llmbench.metrics import trial_prefill_tps

    lines = [
        {"message": {"content": "ok"}, "done": False},
        {
            "message": {"content": ""},
            "done": True,
            "prompt_eval_count": 1000,
            "prompt_eval_cached_count": 990,
            "prompt_eval_duration": 5_000_000,  # 5 ms uncached-only
            "eval_count": 1,
            "eval_duration": 8_000_000,
        },
    ]
    body = b"".join(json.dumps(x).encode() + b"\n" for x in lines)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    backend = OllamaBackend(
        BackendConfig(type="ollama", base_url="http://127.0.0.1:11434", model="llama"),
        transport=httpx.MockTransport(handler),
    )
    result = await backend.complete(_req())
    await backend.aclose()
    assert result.server_timings is not None
    assert result.server_timings.cached_prompt_tokens == 990
    assert result.server_timings.uncached_prompt_tokens == 10
    assert result.server_timings.cache_hit is True
    assert trial_prefill_tps(result.server_timings) == 2000.0
