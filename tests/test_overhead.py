from __future__ import annotations

import pytest

from llmbench.overhead import measure_harness_overhead


@pytest.mark.asyncio
async def test_overhead_is_small_and_positive() -> None:
    overhead = await measure_harness_overhead(iterations=5)
    assert overhead.zero_delay_ttft_ms >= 0.0
    assert overhead.zero_delay_e2e_ms >= overhead.zero_delay_ttft_ms * 0.5
    # Client-only mock should stay in the low millisecond range on CI.
    assert overhead.zero_delay_ttft_ms < 20.0
    assert overhead.timestamp_cost_ns < 20_000
    assert overhead.tokens_in_probe >= 1
