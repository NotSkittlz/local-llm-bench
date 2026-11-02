from __future__ import annotations

import math

from llmbench.metrics import (
    approx_tokens_from_chars,
    percentile,
    summarize,
    trial_decode_tps,
    trial_prefill_tps,
    trial_ttft_ms,
)
from llmbench.schema import ServerTimings


def test_percentile_empty() -> None:
    assert percentile([], 50) is None


def test_percentile_interpolation() -> None:
    xs = [0.0, 10.0, 20.0, 30.0]
    assert percentile(xs, 0) == 0.0
    assert percentile(xs, 100) == 30.0
    assert percentile(xs, 50) == 15.0


def test_summarize_skips_nan() -> None:
    s = summarize([1.0, math.nan, 3.0])
    assert s.n == 2
    assert s.p50 is not None


def test_ttft_and_decode_tps() -> None:
    start = 0
    first = 40_000_000  # 40ms
    last = 140_000_000  # 100ms of decode for 11 tokens -> 10 decode tokens
    assert trial_ttft_ms(start, first) == 40.0
    tps = trial_decode_tps(11, first, last)
    assert tps is not None
    assert abs(tps - 100.0) < 1e-9


def test_decode_tps_none_for_single_token() -> None:
    assert trial_decode_tps(1, 0, 10) is None


def test_prefill_prefers_server_timings() -> None:
    server = ServerTimings(prompt_tokens=100, prompt_ms=50.0)
    proxy = trial_prefill_tps(100, 200.0, server)
    assert proxy == 2000.0  # 100 tokens / 0.05s


def test_prefill_proxy_from_ttft() -> None:
    tps = trial_prefill_tps(80, 40.0, None)
    assert tps == 2000.0


def test_approx_tokens() -> None:
    assert approx_tokens_from_chars(0) == 1
    assert approx_tokens_from_chars(16) == 4
