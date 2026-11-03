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


@pytest.mark.asyncio
async def test_ollama_show_uses_model_key(monkeypatch) -> None:
    monkeypatch.delenv("OLLAMA_NUM_PARALLEL", raising=False)
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/api/show"):
            seen.update(json.loads(request.content))
            return httpx.Response(
                200,
                json={"details": {"family": "llama"}, "capabilities": ["completion", "thinking"]},
            )
        if request.url.path.endswith("/api/version"):
            return httpx.Response(200, json={"version": "0.0-test"})
        return httpx.Response(404)

    backend = OllamaBackend(
        BackendConfig(type="ollama", base_url="http://127.0.0.1:11434", model="llama3.2"),
        transport=httpx.MockTransport(handler),
    )
    info = await backend.info()
    await backend.aclose()
    assert seen.get("model") == "llama3.2"
    assert "name" not in seen
    assert info.extra.get("family") == "llama"
    assert info.extra.get("capabilities") == ["completion", "thinking"]
    assert "num_parallel" not in info.extra


@pytest.mark.asyncio
async def test_ollama_info_parses_context_length(monkeypatch) -> None:
    monkeypatch.delenv("OLLAMA_NUM_PARALLEL", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/api/show"):
            return httpx.Response(
                200,
                json={
                    "parameters": "num_ctx 8192\nstop <|eot_id|>",
                    "details": {"family": "qwen2"},
                    "model_info": {"qwen2.context_length": 32768, "general.parameter_count": 1},
                },
            )
        if request.url.path.endswith("/api/version"):
            return httpx.Response(200, json={"version": "0.0-test"})
        return httpx.Response(404)

    backend = OllamaBackend(
        BackendConfig(type="ollama", base_url="http://127.0.0.1:11434", model="qwen"),
        transport=httpx.MockTransport(handler),
    )
    info = await backend.info()
    await backend.aclose()
    assert info.extra.get("num_ctx") == 8192
    assert info.extra.get("context_length_max") == 32768
    assert "context_length" not in info.extra


@pytest.mark.asyncio
async def test_ollama_info_prefers_ps_runtime_context(monkeypatch) -> None:
    monkeypatch.delenv("OLLAMA_NUM_PARALLEL", raising=False)
    monkeypatch.delenv("OLLAMA_CONTEXT_LENGTH", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/api/show"):
            return httpx.Response(
                200,
                json={"model_info": {"qwen2.context_length": 262144}},
            )
        if request.url.path.endswith("/api/ps"):
            return httpx.Response(
                200,
                json={
                    "models": [
                        {"name": "qwen:latest", "model": "qwen", "context_length": 32768},
                    ]
                },
            )
        if request.url.path.endswith("/api/version"):
            return httpx.Response(200, json={"version": "0.0-test"})
        return httpx.Response(404)

    backend = OllamaBackend(
        BackendConfig(type="ollama", base_url="http://127.0.0.1:11434", model="qwen"),
        transport=httpx.MockTransport(handler),
    )
    info = await backend.info()
    await backend.aclose()
    assert info.extra.get("ps_context_length") == 32768
    assert info.extra.get("context_length_max") == 262144


@pytest.mark.asyncio
async def test_ollama_info_reads_context_length_env_on_loopback(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_CONTEXT_LENGTH", "32768")
    monkeypatch.delenv("OLLAMA_NUM_PARALLEL", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/api/show"):
            return httpx.Response(200, json={"details": {"family": "llama"}})
        if request.url.path.endswith("/api/version"):
            return httpx.Response(200, json={"version": "0.0-test"})
        return httpx.Response(404)

    backend = OllamaBackend(
        BackendConfig(type="ollama", base_url="http://127.0.0.1:11434", model="llama"),
        transport=httpx.MockTransport(handler),
    )
    info = await backend.info()
    await backend.aclose()
    assert info.extra.get("ollama_context_length_env") == 32768


@pytest.mark.asyncio
async def test_ollama_info_reads_num_parallel_from_env(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_NUM_PARALLEL", "4")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/api/show"):
            return httpx.Response(200, json={"details": {"family": "llama"}})
        if request.url.path.endswith("/api/version"):
            return httpx.Response(200, json={"version": "0.0-test"})
        return httpx.Response(404)

    backend = OllamaBackend(
        BackendConfig(type="ollama", base_url="http://127.0.0.1:11434", model="llama3.2"),
        transport=httpx.MockTransport(handler),
    )
    info = await backend.info()
    await backend.aclose()
    assert info.extra.get("num_parallel") == 4


@pytest.mark.asyncio
async def test_ollama_thinking_timestamps_first_generated_token() -> None:
    lines = [
        {"message": {"thinking": "plan"}, "done": False},
        {"message": {"content": "Hi"}, "done": False},
        {"message": {"content": ""}, "done": True, "done_reason": "stop", "eval_count": 2},
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
    assert result.text == "Hi"
    assert result.output_tokens == 2
    assert result.reasoning_tokens == 1
    assert result.content_tokens == 1
    assert result.first_token_ns is not None
    assert result.first_content_ns is not None
    assert result.finish_reason == "stop"
