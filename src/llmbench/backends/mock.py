"""In-process mock backend.

Used for tests, demos, and harness-overhead calibration. It never talks to
the network. Simulated compute is applied with ``asyncio.sleep`` only when
the configured delays are non-zero; zero-delay mode is the overhead probe.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass

from llmbench.backends.base import CompletionRequest, StreamResult
from llmbench.config import BackendConfig
from llmbench.metrics import approx_tokens_from_chars
from llmbench.schema import BackendInfo, ServerTimings


@dataclass(slots=True)
class MockFault:
    """Optional one-shot fault injection for adversarial tests."""

    kind: str | None = None  # timeout | interrupt | protocol | hang
    after_tokens: int = 0


class MockBackend:
    config_type = "mock"

    def __init__(self, config: BackendConfig | None = None, *, fault: MockFault | None = None) -> None:
        self.config = config or BackendConfig()
        self.fault = fault or MockFault()
        self.calls = 0

    async def aclose(self) -> None:
        return None

    async def info(self) -> BackendInfo:
        return BackendInfo(
            type="mock",
            base_url=None,
            version="mock-0.1",
            model_listed=self.config.model,
            extra={"synthetic": True},
        )

    async def complete(self, request: CompletionRequest) -> StreamResult:
        self.calls += 1
        start_ns = time.perf_counter_ns()
        n_out = max(1, self.config.mock_output_tokens)
        prompt_tokens = approx_tokens_from_chars(len(request.prompt))
        prefill_ns = self.config.mock_prompt_eval_ns_per_token * prompt_tokens

        fault = self.fault.kind
        if fault == "timeout":
            return StreamResult(
                start_ns=start_ns,
                end_ns=time.perf_counter_ns(),
                error="mock timeout",
                error_kind="timeout",
                prompt_tokens=prompt_tokens,
            )
        if fault == "hang":
            await asyncio.sleep(3600)
        if fault == "protocol":
            return StreamResult(
                start_ns=start_ns,
                end_ns=time.perf_counter_ns(),
                error="mock protocol error: not JSON",
                error_kind="protocol",
                prompt_tokens=prompt_tokens,
            )

        if self.config.mock_ttft_ns > 0:
            await asyncio.sleep(self.config.mock_ttft_ns / 1e9)
        elif prefill_ns > 0:
            await asyncio.sleep(prefill_ns / 1e9)

        first_token_ns = time.perf_counter_ns()
        first_byte_ns = first_token_ns
        last_token_ns = first_token_ns
        texts: list[str] = []
        inter: list[int] = []
        prev = first_token_ns
        produced = 0

        for i in range(n_out):
            if fault == "interrupt" and i >= self.fault.after_tokens:
                return StreamResult(
                    start_ns=start_ns,
                    end_ns=time.perf_counter_ns(),
                    first_byte_ns=first_byte_ns,
                    first_token_ns=first_token_ns,
                    last_token_ns=last_token_ns,
                    output_tokens=produced,
                    prompt_tokens=prompt_tokens,
                    output_chars=sum(len(t) for t in texts),
                    text="".join(texts) if request.collect_text else None,
                    error="mock stream interrupted",
                    error_kind="interrupted",
                    token_source="count_chunks",
                    inter_token_ns=inter if request.record_inter_token else [],
                )
            if i > 0:
                delay = self.config.mock_inter_token_ns
                if delay > 0:
                    await asyncio.sleep(delay / 1e9)
                now = time.perf_counter_ns()
                if request.record_inter_token:
                    inter.append(now - prev)
                last_token_ns = now
                prev = now
            produced += 1
            if request.collect_text:
                texts.append(_token_for(request.prompt, request.sampling.seed, i))

        end_ns = time.perf_counter_ns()
        prompt_ms = (self.config.mock_prompt_eval_ns_per_token * prompt_tokens) / 1e6
        completion_ms = (self.config.mock_inter_token_ns * max(0, n_out - 1)) / 1e6
        server = ServerTimings(
            prompt_tokens=prompt_tokens,
            prompt_ms=prompt_ms if prompt_ms > 0 else None,
            completion_tokens=n_out,
            completion_ms=completion_ms if completion_ms > 0 else None,
            total_ms=(end_ns - start_ns) / 1e6,
        )
        text = "".join(texts) if request.collect_text else None
        return StreamResult(
            start_ns=start_ns,
            end_ns=end_ns,
            first_byte_ns=first_byte_ns,
            first_token_ns=first_token_ns,
            last_token_ns=last_token_ns,
            output_tokens=produced,
            prompt_tokens=prompt_tokens,
            output_chars=len(text) if text is not None else produced,
            text=text,
            server_timings=server,
            token_source="count_chunks",
            inter_token_ns=inter if request.record_inter_token else [],
        )


def _token_for(prompt: str, seed: int, index: int) -> str:
    digest = hashlib.sha256(f"{seed}:{index}:{prompt}".encode()).hexdigest()
    # Stable 1-token-ish word from the hash, space-separated.
    return (" " if index else "") + digest[:6]
