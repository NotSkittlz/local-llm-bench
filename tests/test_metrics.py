from __future__ import annotations

import math

from llmbench.metrics import (
    approx_tokens_from_chars,
    fill_trial_derived,
    percentile,
    summarize,
    trial_decode_tps,
    trial_prefill_tps,
    trial_prefill_tps_proxy,
    trial_ttft_ms,
)
from llmbench.schema import ServerTimings, Trial


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


def test_prefill_uses_uncached_tokens_only() -> None:
    server = ServerTimings(
        prompt_tokens=1000,
        cached_prompt_tokens=990,
        uncached_prompt_tokens=10,
        prompt_ms=5.0,
        uncached_prompt_ms=5.0,
        cache_hit=True,
    )
    assert trial_prefill_tps(server) == 2000.0  # 10 / 0.005s, not 1000/0.005s


def test_prefill_null_when_fully_cached() -> None:
    server = ServerTimings(
        prompt_tokens=800,
        cached_prompt_tokens=800,
        uncached_prompt_tokens=0,
        prompt_ms=0.0,
        uncached_prompt_ms=0.0,
        cache_hit=True,
    )
    assert trial_prefill_tps(server) is None


def test_prefill_null_without_uncached_breakdown() -> None:
    # Ollama: all-token count + uncached duration, no cached count → refuse.
    server = ServerTimings(prompt_tokens=100, prompt_ms=50.0)
    assert trial_prefill_tps(server) is None


def test_prefill_proxy_is_separate_and_not_mixed() -> None:
    assert trial_prefill_tps(None) is None
    assert trial_prefill_tps_proxy(80, 40.0) == 2000.0
    trial = Trial(
        trial_id="t",
        suite="short_chat",
        prompt_id="p",
        concurrency=1,
        ok=True,
        prompt_chars=10,
        prompt_tokens=80,
        start_ns=0,
        first_token_ns=40_000_000,
        end_ns=80_000_000,
    )
    fill_trial_derived(trial)
    assert trial.prefill_tps is None
    assert trial.prefill_tps_proxy == 2000.0
    assert trial.prefill_source == "proxy_ttft"


def test_summarize_does_not_mix_proxy_and_server() -> None:
    server = summarize([2000.0, 2100.0])
    proxy = summarize([50_000.0])
    assert server.p50 is not None and server.p50 < 3000
    assert proxy.p50 == 50_000.0


def test_approx_tokens() -> None:
    assert approx_tokens_from_chars(0) == 1
    assert approx_tokens_from_chars(16) == 4
