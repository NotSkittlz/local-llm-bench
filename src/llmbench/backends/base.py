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


@dataclass(slots=True)
class CompletionRequest:
    prompt: str
    sampling: SamplingParams
    collect_text: bool = False
    record_inter_token: bool = False
    prompt_id: str = ""
    context_length: int | None = None
    flavor: str = "openai_compat"


@dataclass(slots=True)
class StreamResult:
    start_ns: int
    end_ns: int
    first_byte_ns: int | None = None
    first_token_ns: int | None = None
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
    inter_token_ns: list[int] = field(default_factory=list)


class Backend(Protocol):
    config_type: str

    async def aclose(self) -> None: ...

    async def info(self) -> BackendInfo: ...

    async def complete(self, request: CompletionRequest) -> StreamResult: ...
