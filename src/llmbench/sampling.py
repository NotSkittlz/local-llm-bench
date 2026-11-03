"""Map requested sampling fields onto what each backend actually accepts.

The harness never silently rewrites temperature or max_tokens for speed.
It *does* rename fields (vLLM ``repetition_penalty``) and omits names a
flavor will ignore, recording the drop so the snapshot is honest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from llmbench.schema import SamplingParams

Flavor = Literal["openai_compat", "llamacpp", "vllm", "ollama", "mock"]

_OPENAI_CORE = ("temperature", "top_p", "max_tokens", "seed")


@dataclass(slots=True)
class SamplingWire:
    """Payload fragments plus an audit of what was sent vs dropped."""

    applied: dict[str, Any] = field(default_factory=dict)
    dropped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    openai_body: dict[str, Any] = field(default_factory=dict)
    ollama_options: dict[str, Any] = field(default_factory=dict)
    ollama_extra: dict[str, Any] = field(default_factory=dict)


def map_sampling(
    flavor: Flavor,
    sampling: SamplingParams,
    *,
    context_length: int | None = None,
    thinking: str = "default",
) -> SamplingWire:
    if flavor == "ollama":
        wire = _map_ollama(sampling, context_length)
    elif flavor == "vllm":
        wire = _map_vllm(sampling, context_length)
    elif flavor == "llamacpp":
        wire = _map_llamacpp(sampling, context_length)
    elif flavor == "mock":
        wire = _map_mock(sampling, context_length)
    else:
        wire = _map_openai_compat(sampling, context_length)
    _apply_thinking(wire, flavor, thinking)
    return wire


def _map_openai_compat(sampling: SamplingParams, context_length: int | None) -> SamplingWire:
    wire = SamplingWire()
    body = _core_openai(sampling)
    wire.applied.update({k: body[k] for k in _OPENAI_CORE})
    if sampling.stop:
        body["stop"] = list(sampling.stop)
        wire.applied["stop"] = list(sampling.stop)
    extras = _optional_requested(sampling)
    for name in extras:
        wire.dropped.append(name)
        wire.warnings.append(
            f"{name} is not a standard OpenAI Chat Completions field; "
            "not sent on backend openai_compat"
        )
    _drop_context(wire, context_length, "openai_compat")
    wire.openai_body = body
    return wire


def _map_llamacpp(sampling: SamplingParams, context_length: int | None) -> SamplingWire:
    wire = SamplingWire()
    body = _core_openai(sampling)
    wire.applied.update({k: body[k] for k in _OPENAI_CORE})
    if sampling.stop:
        body["stop"] = list(sampling.stop)
        wire.applied["stop"] = list(sampling.stop)
    if sampling.top_k is not None:
        body["top_k"] = sampling.top_k
        wire.applied["top_k"] = sampling.top_k
    if sampling.min_p is not None:
        body["min_p"] = sampling.min_p
        wire.applied["min_p"] = sampling.min_p
    if sampling.repeat_penalty is not None:
        body["repeat_penalty"] = sampling.repeat_penalty
        wire.applied["repeat_penalty"] = sampling.repeat_penalty
    _drop_context(wire, context_length, "llamacpp")
    wire.openai_body = body
    return wire


def _map_vllm(sampling: SamplingParams, context_length: int | None) -> SamplingWire:
    wire = SamplingWire()
    body = _core_openai(sampling)
    wire.applied.update({k: body[k] for k in _OPENAI_CORE})
    if sampling.stop:
        body["stop"] = list(sampling.stop)
        wire.applied["stop"] = list(sampling.stop)
    if sampling.top_k is not None:
        body["top_k"] = sampling.top_k
        wire.applied["top_k"] = sampling.top_k
    if sampling.min_p is not None:
        body["min_p"] = sampling.min_p
        wire.applied["min_p"] = sampling.min_p
    if sampling.repeat_penalty is not None:
        body["repetition_penalty"] = sampling.repeat_penalty
        wire.applied["repetition_penalty"] = sampling.repeat_penalty
    _drop_context(wire, context_length, "vllm")
    wire.openai_body = body
    return wire


def _map_ollama(sampling: SamplingParams, context_length: int | None) -> SamplingWire:
    wire = SamplingWire()
    options: dict[str, Any] = {
        "temperature": sampling.temperature,
        "top_p": sampling.top_p,
        "num_predict": sampling.max_tokens,
        "seed": sampling.seed,
    }
    wire.applied.update(
        {
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "num_predict": sampling.max_tokens,
            "seed": sampling.seed,
        }
    )
    if sampling.top_k is not None:
        options["top_k"] = sampling.top_k
        wire.applied["top_k"] = sampling.top_k
    if sampling.min_p is not None:
        options["min_p"] = sampling.min_p
        wire.applied["min_p"] = sampling.min_p
    if sampling.repeat_penalty is not None:
        options["repeat_penalty"] = sampling.repeat_penalty
        wire.applied["repeat_penalty"] = sampling.repeat_penalty
    if sampling.stop:
        options["stop"] = list(sampling.stop)
        wire.applied["stop"] = list(sampling.stop)
    if context_length is not None:
        options["num_ctx"] = context_length
        wire.applied["num_ctx"] = context_length
    wire.ollama_options = options
    return wire


def _map_mock(sampling: SamplingParams, context_length: int | None) -> SamplingWire:
    wire = SamplingWire()
    wire.applied = {
        "temperature": sampling.temperature,
        "top_p": sampling.top_p,
        "max_tokens": sampling.max_tokens,
        "seed": sampling.seed,
    }
    if sampling.top_k is not None:
        wire.applied["top_k"] = sampling.top_k
    if sampling.min_p is not None:
        wire.applied["min_p"] = sampling.min_p
    if sampling.repeat_penalty is not None:
        wire.applied["repeat_penalty"] = sampling.repeat_penalty
    if sampling.stop:
        wire.applied["stop"] = list(sampling.stop)
    if context_length is not None:
        wire.dropped.append("context_length")
        wire.warnings.append(
            "context_length is user-declared metadata; the mock backend does not apply it"
        )
    return wire


def _core_openai(sampling: SamplingParams) -> dict[str, Any]:
    return {
        "temperature": sampling.temperature,
        "top_p": sampling.top_p,
        "max_tokens": sampling.max_tokens,
        "seed": sampling.seed,
        "stream_options": {"include_usage": True},
    }


def _optional_requested(sampling: SamplingParams) -> list[str]:
    names: list[str] = []
    if sampling.top_k is not None:
        names.append("top_k")
    if sampling.min_p is not None:
        names.append("min_p")
    if sampling.repeat_penalty is not None:
        names.append("repeat_penalty")
    return names


def _apply_thinking(wire: SamplingWire, flavor: Flavor, thinking: str) -> None:
    if thinking in {"", "default"}:
        return
    if flavor == "ollama":
        value: Any = False if thinking == "off" else thinking
        wire.ollama_extra["think"] = value
        wire.applied["think"] = value
        return
    if flavor == "mock":
        wire.applied["thinking"] = thinking
        return
    kwargs: dict[str, Any] = {}
    if thinking == "off":
        kwargs["enable_thinking"] = False
        if flavor == "llamacpp":
            kwargs["reasoning_budget"] = 0
    else:
        kwargs["enable_thinking"] = True
        kwargs["reasoning_effort"] = thinking
    wire.openai_body["chat_template_kwargs"] = kwargs
    wire.applied["chat_template_kwargs"] = kwargs


def _drop_context(wire: SamplingWire, context_length: int | None, flavor: str) -> None:
    if context_length is None:
        return
    wire.dropped.append("context_length")
    wire.warnings.append(
        f"context_length={context_length} is user-declared metadata; "
        f"{flavor} does not take a per-request context size from this harness"
    )
