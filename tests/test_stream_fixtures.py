"""Parse captured llama.cpp / vLLM / Ollama stream shapes."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from llmbench.backends.base import FLAG_NO_CONTENT_REASONING, CompletionRequest
from llmbench.backends.ollama import OllamaBackend
from llmbench.backends.openai_compat import OpenAICompatBackend
from llmbench.config import BackendConfig
from llmbench.metrics import fill_trial_derived, trial_prefill_tps
from llmbench.schema import SamplingParams, Trial

FIXTURES = Path(__file__).parent / "fixtures"


def _req() -> CompletionRequest:
    return CompletionRequest(
        prompt="Say hi",
        sampling=SamplingParams(max_tokens=8, seed=1),
        collect_text=True,
    )


@pytest.mark.asyncio
async def test_llamacpp_fixture_stream() -> None:
    body = (FIXTURES / "llamacpp_stream.sse").read_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    backend = OpenAICompatBackend(
        BackendConfig(type="llamacpp", base_url="http://127.0.0.1:8080", model="qwen2.5"),
        flavor="llamacpp",
        transport=httpx.MockTransport(handler),
    )
    result = await backend.complete(_req())
    await backend.aclose()
    assert result.error is None
    assert result.text == "Hello world"
    assert result.output_tokens == 2
    assert result.server_timings is not None
    assert result.server_timings.cached_prompt_tokens == 0
    assert result.server_timings.uncached_prompt_tokens == 12
    assert trial_prefill_tps(result.server_timings) == 300.0


@pytest.mark.asyncio
async def test_vllm_fixture_content_array_and_float_usage() -> None:
    body = (FIXTURES / "vllm_stream.sse").read_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    backend = OpenAICompatBackend(
        BackendConfig(type="vllm", base_url="http://127.0.0.1:8000", model="qwen2.5"),
        flavor="vllm",
        transport=httpx.MockTransport(handler),
    )
    result = await backend.complete(_req())
    await backend.aclose()
    assert result.error is None
    assert result.text == "Hi"
    assert result.output_tokens == 1
    assert result.prompt_tokens == 10
    assert result.token_source == "server_usage"
    assert trial_prefill_tps(result.server_timings) is None


@pytest.mark.asyncio
async def test_ollama_fixture_stream() -> None:
    body = (FIXTURES / "ollama_stream.ndjson").read_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    backend = OllamaBackend(
        BackendConfig(type="ollama", base_url="http://127.0.0.1:11434", model="llama3.2"),
        transport=httpx.MockTransport(handler),
    )
    result = await backend.complete(_req())
    await backend.aclose()
    assert result.error is None
    assert result.text == "Paris"
    assert result.token_source == "server_usage"
    assert result.server_timings is not None
    assert result.server_timings.uncached_prompt_tokens == 8
    assert trial_prefill_tps(result.server_timings) == 400.0


def _as_trial(result, *, suite: str = "short_chat") -> Trial:
    trial = Trial(
        trial_id="t",
        suite=suite,
        prompt_id="p",
        concurrency=1,
        ok=result.error is None,
        prompt_chars=10,
        prompt_tokens=result.prompt_tokens,
        output_tokens=result.output_tokens,
        output_chars=result.output_chars,
        start_ns=result.start_ns,
        first_byte_ns=result.first_byte_ns,
        first_token_ns=result.first_token_ns,
        first_content_ns=result.first_content_ns,
        last_token_ns=result.last_token_ns,
        end_ns=result.end_ns,
        server_timings=result.server_timings,
        token_source=result.token_source,  # type: ignore[arg-type]
        reasoning_tokens=result.reasoning_tokens,
        content_tokens=result.content_tokens,
        finish_reason=result.finish_reason,
    )
    return fill_trial_derived(trial)


@pytest.mark.asyncio
async def test_ollama_thinking_fixture_uses_server_decode() -> None:
    body = (FIXTURES / "ollama_thinking.ndjson").read_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    backend = OllamaBackend(
        BackendConfig(type="ollama", base_url="http://127.0.0.1:11434", model="qwen3.8:27b"),
        transport=httpx.MockTransport(handler),
    )
    result = await backend.complete(_req())
    await backend.aclose()
    assert result.error is None
    assert result.text == "Hello"
    assert result.reasoning_tokens == 3
    assert result.content_tokens == 1
    assert result.first_token_ns is not None
    assert result.first_content_ns is not None
    assert result.finish_reason == "stop"
    assert result.output_tokens == 100
    trial = _as_trial(result)
    assert trial.decode_source == "server"
    assert trial.decode_tps is not None
    assert abs(trial.decode_tps - (100 / 1.379)) < 1e-6


@pytest.mark.asyncio
async def test_llamacpp_reasoning_fixture_uses_server_decode() -> None:
    body = (FIXTURES / "llamacpp_reasoning.sse").read_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    backend = OpenAICompatBackend(
        BackendConfig(type="llamacpp", base_url="http://127.0.0.1:8080", model="qwen3.8:27b"),
        flavor="llamacpp",
        transport=httpx.MockTransport(handler),
    )
    result = await backend.complete(_req())
    await backend.aclose()
    assert result.error is None
    assert result.text == "Hi"
    assert result.reasoning_tokens == 3
    assert result.content_tokens == 1
    assert result.first_token_ns is not None
    assert result.first_content_ns is not None
    assert result.finish_reason == "stop"
    assert result.output_tokens == 128
    trial = _as_trial(result)
    assert trial.decode_source == "server"
    assert trial.decode_tps is not None
    assert abs(trial.decode_tps - (128 / 2.805)) < 1e-6


@pytest.mark.asyncio
async def test_ollama_thinking_length_is_ok_reasoning_only() -> None:
    body = (FIXTURES / "ollama_thinking_length.ndjson").read_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    backend = OllamaBackend(
        BackendConfig(type="ollama", base_url="http://127.0.0.1:11434", model="qwen3.8:27b"),
        transport=httpx.MockTransport(handler),
    )
    result = await backend.complete(_req())
    await backend.aclose()
    assert result.error is None
    assert result.error_kind is None
    assert FLAG_NO_CONTENT_REASONING in result.flags
    assert result.finish_reason == "length"
    assert result.content_tokens == 0
    assert result.reasoning_tokens == 3
    assert result.first_token_ns is not None
    assert result.first_content_ns is None
    trial = _as_trial(result)
    assert trial.ok is True
    assert trial.decode_source == "server"
    assert trial.decode_tps is not None
    assert abs(trial.decode_tps - (32 / 0.360)) < 1e-6


@pytest.mark.asyncio
async def test_llamacpp_reasoning_length_is_ok_reasoning_only() -> None:
    body = (FIXTURES / "llamacpp_reasoning_length.sse").read_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    backend = OpenAICompatBackend(
        BackendConfig(type="llamacpp", base_url="http://127.0.0.1:8080", model="qwen3.8:27b"),
        flavor="llamacpp",
        transport=httpx.MockTransport(handler),
    )
    result = await backend.complete(_req())
    await backend.aclose()
    assert result.error is None
    assert FLAG_NO_CONTENT_REASONING in result.flags
    assert result.finish_reason == "length"
    assert result.content_tokens == 0
    assert result.reasoning_tokens == 2
    assert result.first_content_ns is None
    trial = _as_trial(result)
    assert trial.decode_source == "server"
    assert abs(trial.decode_tps - (32 / 0.720)) < 1e-6  # type: ignore[operator]
