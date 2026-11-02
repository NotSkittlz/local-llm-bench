"""Backend adapter protocol.

Hot-path rules for every adapter:

1. Call ``time.perf_counter_ns()`` immediately before sending the request
   (``start_ns``) and on every inbound byte chunk *before* parsing.
2. Do not log, format, or hash tokens while streaming a speed trial.
3. Do not accumulate output text unless ``collect_text`` is true (quality
   probe only).
4. Do not sample memory here; that is ``llmbench.memory.MemorySampler``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from llmbench.schema import BackendInfo, SamplingParams, ServerTimings

FLAG_NO_CONTENT_REASONING = "no_content_reasoning_only"


def classify_stream_end(
    *,
    error: str | None,
    error_kind: str | None,
    saw_terminal: bool,
    missing_terminal: str,
    content_tokens: int,
    reasoning_tokens: int,
) -> tuple[str | None, str | None, list[str]]:
    """Distinguish a truncated stream from a finished reasoning-only response."""

    flags: list[str] = []
    if error is not None:
        return error, error_kind, flags
    if not saw_terminal:
        return missing_terminal, "truncated", flags
    if content_tokens <= 0 and reasoning_tokens > 0:
        flags.append(FLAG_NO_CONTENT_REASONING)
        return None, None, flags
    if content_tokens <= 0:
        return "stream ended without a content token", "interrupted", flags
    return None, None, flags


@dataclass(slots=True)
class CompletionRequest:
    prompt: str
    sampling: SamplingParams
    collect_text: bool = False
    record_inter_token: bool = False
    prompt_id: str = ""
    context_length: int | None = None
    flavor: str = "openai_compat"
    thinking: str = "default"


@dataclass(slots=True)
class StreamResult:
    start_ns: int
    end_ns: int
    first_byte_ns: int | None = None
    first_token_ns: int | None = None
    first_content_ns: int | None = None
    last_token_ns: int | None = None
    output_tokens: int = 0
    prompt_tokens: int | None = None
    output_chars: int = 0
    text: str | None = None
    error: str | None = None
    error_kind: str | None = None
    server_timings: ServerTimings | None = None
    token_source: str | None = None
    reasoning_tokens: int = 0
    content_tokens: int = 0
    finish_reason: str | None = None
    flags: list[str] = field(default_factory=list)
    inter_token_ns: list[int] = field(default_factory=list)


class Backend(Protocol):
    config_type: str

    async def aclose(self) -> None: ...

    async def info(self) -> BackendInfo: ...

    async def complete(self, request: CompletionRequest) -> StreamResult: ...
