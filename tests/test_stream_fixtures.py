"""Parse captured llama.cpp / vLLM / Ollama stream shapes."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from llmbench.backends.base import CompletionRequest
from llmbench.backends.ollama import OllamaBackend
from llmbench.backends.openai_compat import OpenAICompatBackend
from llmbench.config import BackendConfig
from llmbench.metrics import trial_prefill_tps
from llmbench.schema import SamplingParams

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
