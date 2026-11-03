from __future__ import annotations

import json

import httpx
import pytest

from llmbench.backends.base import CompletionRequest
from llmbench.backends.ollama import OllamaBackend
from llmbench.backends.openai_compat import OpenAICompatBackend
from llmbench.config import BackendConfig, BenchConfig, MemoryConfig, ModelConfig, QualityConfig
from llmbench.runner import run_benchmark
from llmbench.sampling import map_sampling
from llmbench.schema import SamplingParams


def test_vllm_maps_repeat_penalty() -> None:
    wire = map_sampling(
        "vllm",
        SamplingParams(repeat_penalty=1.1, temperature=0.2, max_tokens=16),
    )
    assert wire.openai_body["repetition_penalty"] == 1.1
    assert "repeat_penalty" not in wire.openai_body
    assert wire.applied["repetition_penalty"] == 1.1
    assert "repeat_penalty" not in wire.dropped


def test_openai_compat_drops_nonstandard_fields() -> None:
    wire = map_sampling(
        "openai_compat",
        SamplingParams(top_k=40, min_p=0.05, repeat_penalty=1.1),
        context_length=8192,
    )
    assert "top_k" not in wire.openai_body
    assert "min_p" not in wire.openai_body
    assert "repeat_penalty" not in wire.openai_body
    assert set(wire.dropped) >= {"top_k", "min_p", "repeat_penalty", "context_length"}
    assert wire.warnings


def test_llamacpp_sends_native_names() -> None:
    wire = map_sampling(
        "llamacpp",
        SamplingParams(top_k=40, min_p=0.05, repeat_penalty=1.1, stop=["\n"]),
    )
    assert wire.openai_body["top_k"] == 40
    assert wire.openai_body["min_p"] == 0.05
    assert wire.openai_body["repeat_penalty"] == 1.1
    assert wire.openai_body["stop"] == ["\n"]


def test_ollama_sends_min_p_stop_and_num_ctx() -> None:
    wire = map_sampling(
        "ollama",
        SamplingParams(min_p=0.1, stop=["END"], max_tokens=32),
        context_length=4096,
    )
    assert wire.ollama_options["min_p"] == 0.1
    assert wire.ollama_options["stop"] == ["END"]
    assert wire.ollama_options["num_ctx"] == 4096
    assert wire.ollama_options["num_predict"] == 32
    assert "context_length" not in wire.dropped


def test_thinking_off_maps_per_backend() -> None:
    ollama = map_sampling("ollama", SamplingParams(max_tokens=8), thinking="off")
    assert ollama.ollama_extra["think"] is False
    assert ollama.applied["think"] is False
    llama = map_sampling("llamacpp", SamplingParams(max_tokens=8), thinking="off")
    kwargs = llama.openai_body["chat_template_kwargs"]
    assert kwargs["enable_thinking"] is False
    assert kwargs["reasoning_budget"] == 0
    vllm = map_sampling("vllm", SamplingParams(max_tokens=8), thinking="low")
    assert vllm.openai_body["chat_template_kwargs"]["enable_thinking"] is True
    assert vllm.openai_body["chat_template_kwargs"]["reasoning_effort"] == "low"
    defaulted = map_sampling("ollama", SamplingParams(max_tokens=8), thinking="default")
    assert "think" not in defaulted.ollama_extra


@pytest.mark.asyncio
async def test_vllm_http_payload_uses_repetition_penalty() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        body = b'data: {"choices":[{"delta":{"content":"x"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
        return httpx.Response(200, content=body)

    backend = OpenAICompatBackend(
        BackendConfig(type="vllm", base_url="http://127.0.0.1:8000", model="m"),
        flavor="vllm",
        transport=httpx.MockTransport(handler),
    )
    await backend.complete(
        CompletionRequest(
            prompt="hi",
            sampling=SamplingParams(repeat_penalty=1.05, max_tokens=4, seed=1),
        )
    )
    await backend.aclose()
    assert seen["repetition_penalty"] == 1.05
    assert "repeat_penalty" not in seen


@pytest.mark.asyncio
async def test_ollama_http_payload_includes_min_p_stop_num_ctx() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        body = (
            json.dumps({"message": {"content": "x"}, "done": True, "eval_count": 1}) + "\n"
        ).encode()
        return httpx.Response(200, content=body)

    backend = OllamaBackend(
        BackendConfig(type="ollama", base_url="http://127.0.0.1:11434", model="llama"),
        transport=httpx.MockTransport(handler),
    )
    await backend.complete(
        CompletionRequest(
            prompt="hi",
            sampling=SamplingParams(min_p=0.2, stop=["END"], max_tokens=4, seed=1),
            context_length=2048,
        )
    )
    await backend.aclose()
    opts = seen["options"]
    assert opts["min_p"] == 0.2
    assert opts["stop"] == ["END"]
    assert opts["num_ctx"] == 2048


@pytest.mark.asyncio
async def test_thinking_off_is_sent_on_the_wire() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            content=(
                json.dumps({"message": {"content": "x"}, "done": True, "eval_count": 1}) + "\n"
            ).encode(),
        )

    backend = OllamaBackend(
        BackendConfig(type="ollama", base_url="http://127.0.0.1:11434", model="llama"),
        transport=httpx.MockTransport(handler),
    )
    await backend.complete(
        CompletionRequest(
            prompt="hi",
            sampling=SamplingParams(max_tokens=4, seed=1),
            thinking="off",
        )
    )
    await backend.aclose()
    assert seen["think"] is False

    seen.clear()

    def llama_handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            content=b'data: {"choices":[{"delta":{"content":"x"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n',
        )

    llama = OpenAICompatBackend(
        BackendConfig(type="llamacpp", base_url="http://127.0.0.1:8080", model="m"),
        flavor="llamacpp",
        transport=httpx.MockTransport(llama_handler),
    )
    await llama.complete(
        CompletionRequest(
            prompt="hi",
            sampling=SamplingParams(max_tokens=4, seed=1),
            thinking="medium",
        )
    )
    await llama.aclose()
    kwargs = seen["chat_template_kwargs"]
    assert kwargs["enable_thinking"] is True
    assert kwargs["reasoning_effort"] == "medium"


@pytest.mark.asyncio
async def test_run_records_user_declared_metadata() -> None:
    cfg = BenchConfig(
        warmup=0,
        iterations=1,
        suites=["short_chat"],
        backend=BackendConfig(
            type="mock", mock_output_tokens=2, mock_ttft_ns=0, mock_inter_token_ns=0
        ),
        model=ModelConfig(name="mock-7b", quantization="Q4_K_M", context_length=8192),
        sampling=SamplingParams(top_k=8, max_tokens=8),
        quality=QualityConfig(enabled=False),
        memory=MemoryConfig(enabled=False),
    )
    result = await run_benchmark(cfg)
    assert result.model.quantization == "Q4_K_M"
    assert result.model.quantization_source == "user_declared"
    assert result.model.context_length == 8192
    assert result.model.context_length_source == "user_declared"
    assert "context_length" in result.sampling.dropped
    assert any("user-declared" in n for n in result.notes)
    html = __import__("llmbench.report", fromlist=["render_html"]).render_html([result])
    assert "user-declared, not applied" in html
