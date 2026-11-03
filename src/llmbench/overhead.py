"""Harness overhead calibration against the in-process mock.

The mock is configured with zero simulated compute so the only work is
Python, the event loop, and ``perf_counter_ns``. That floor is recorded on
every run (including real-backend runs) so a reader can subtract it.
"""

from __future__ import annotations

import time

from llmbench.backends.base import CompletionRequest
from llmbench.backends.mock import MockBackend
from llmbench.config import BackendConfig
from llmbench.schema import HarnessOverhead, SamplingParams

_OVERHEAD_TOKENS = 32
_OVERHEAD_ITERS = 7


async def measure_harness_overhead(*, iterations: int = _OVERHEAD_ITERS) -> HarnessOverhead:
    backend = MockBackend(
        BackendConfig(
            type="mock",
            model="overhead-probe",
            mock_output_tokens=_OVERHEAD_TOKENS,
            mock_ttft_ns=0,
            mock_inter_token_ns=0,
        )
    )
    request = CompletionRequest(
        prompt="overhead",
        sampling=SamplingParams(max_tokens=_OVERHEAD_TOKENS, seed=0, temperature=0.0),
    )
    ttfts: list[float] = []
    e2es: list[float] = []
    # Warm the function/import caches once, then measure.
    await backend.complete(request)
    for _ in range(iterations):
        result = await backend.complete(request)
        ttfts.append((result.first_token_ns - result.start_ns) / 1e6)  # type: ignore[operator]
        e2es.append((result.end_ns - result.start_ns) / 1e6)
    await backend.aclose()

    stamp_cost = _timestamp_cost_ns()
    return HarnessOverhead(
        zero_delay_ttft_ms=_median(ttfts),
        zero_delay_e2e_ms=_median(e2es),
        timestamp_cost_ns=stamp_cost,
        tokens_in_probe=_OVERHEAD_TOKENS,
        iterations=iterations,
        notes=[
            "measured against in-process mock with zero simulated compute",
            "subtract zero_delay_ttft_ms from client TTFT for a lower bound on server time",
            "does not include HTTP/TCP; OpenAI-compatible adapters add loopback cost on top",
        ],
    )


def _timestamp_cost_ns(n: int = 20_000) -> float:
    t0 = time.perf_counter_ns()
    acc = 0
    for _ in range(n):
        acc ^= time.perf_counter_ns()
    t1 = time.perf_counter_ns()
    if acc == 0:  # keep the xor live
        return (t1 - t0) / n
    return (t1 - t0) / n


def _median(xs: list[float]) -> float:
    ys = sorted(xs)
    mid = len(ys) // 2
    if len(ys) % 2:
        return ys[mid]
    return 0.5 * (ys[mid - 1] + ys[mid])
