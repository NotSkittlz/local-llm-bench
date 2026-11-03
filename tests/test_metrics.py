from __future__ import annotations

import math

from llmbench.metrics import (
    FLAG_DECODE_DISAGREE,
    approx_tokens_from_chars,
    bimodal_prefill_ratio,
    classify_cache_hit,
    fill_trial_derived,
    percentile,
    summarize,
    trial_decode_tps,
    trial_decode_tps_server,
    trial_prefill_time_ms,
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
        suite="long_context",
        prompt_id="p",
        concurrency=1,
        ok=True,
        prompt_chars=1200,
        prompt_tokens=300,
        start_ns=0,
        first_token_ns=40_000_000,
        end_ns=80_000_000,
    )
    fill_trial_derived(trial)
    assert trial.prefill_tps is None
    assert trial.prefill_tps_proxy == 7500.0
    assert trial.prefill_source == "proxy_ttft"


def test_short_prompt_prefill_is_not_meaningful() -> None:
    trial = Trial(
        trial_id="t",
        suite="short_chat",
        prompt_id="p",
        concurrency=1,
        ok=True,
        prompt_chars=10,
        prompt_tokens=19,
        start_ns=0,
        first_token_ns=40_000_000,
        end_ns=80_000_000,
        server_timings=ServerTimings(
            prompt_tokens=19,
            uncached_prompt_tokens=19,
            uncached_prompt_ms=121.0,
        ),
    )
    fill_trial_derived(trial)
    assert trial.prefill_tps is None
    assert trial.prefill_tps_proxy is None
    assert trial.prefill_source == "not_meaningful_short_prompt"


def test_summarize_does_not_mix_proxy_and_server() -> None:
    server = summarize([2000.0, 2100.0])
    proxy = summarize([50_000.0])
    assert server.p50 is not None and server.p50 < 3000
    assert proxy.p50 == 50_000.0


def test_approx_tokens() -> None:
    assert approx_tokens_from_chars(0) == 1
    assert approx_tokens_from_chars(16) == 4


def test_template_prefix_cache_is_not_a_hit() -> None:
    hit, frac = classify_cache_hit(42, 62, uncached=20)
    assert hit is False
    assert frac is not None
    assert abs(frac - 42 / 62) < 1e-9
    hit_long, frac_long = classify_cache_hit(42, 3450, uncached=3408)
    assert hit_long is False
    assert frac_long is not None and frac_long < 0.02
    real, real_frac = classify_cache_hit(799, 800, uncached=1)
    assert real is True
    assert real_frac is not None and real_frac > 0.9


def test_server_decode_matches_ollama_eval_ground_truth() -> None:
    # Direct Ollama /api/generate: eval_count=100 in 1379 ms = 72.5 tok/s.
    server = ServerTimings(completion_tokens=100, completion_ms=1379.0)
    tps = trial_decode_tps_server(server)
    assert tps is not None
    assert abs(tps - (100 / 1.379)) < 1e-9


def test_server_decode_matches_llamacpp_predicted_ground_truth() -> None:
    # llama.cpp timings: predicted_n=128 in 2805 ms = ~45.6 tok/s.
    server = ServerTimings(completion_tokens=128, completion_ms=2805.0)
    tps = trial_decode_tps_server(server)
    assert tps is not None
    assert abs(tps - (128 / 2.805)) < 1e-9


def test_fill_prefers_server_decode_when_client_span_is_content_only() -> None:
    # Live bug: output_tokens includes thinking, but first_token was first content.
    # Client span of 20 chars / 65 us looked like 1934 tok/s; server was 45.6.
    trial = Trial(
        trial_id="t",
        suite="short_chat",
        prompt_id="chat-03",
        concurrency=1,
        ok=True,
        prompt_chars=20,
        output_tokens=128,
        reasoning_tokens=122,
        content_tokens=6,
        start_ns=0,
        first_token_ns=2_983_221_800,  # first *content* (wrong span)
        last_token_ns=3_048_861_100,
        end_ns=3_100_000_000,
        server_timings=ServerTimings(completion_tokens=128, completion_ms=2804.361),
    )
    fill_trial_derived(trial)
    assert trial.decode_source == "server"
    assert trial.decode_tps_server is not None
    assert abs(trial.decode_tps_server - (128 / 2.804361)) < 1e-6
    assert trial.decode_tps == trial.decode_tps_server
    assert FLAG_DECODE_DISAGREE in trial.flags
    assert trial.ttft_ms is not None


def test_short_output_does_not_flag_decode_disagree() -> None:
    trial = Trial(
        trial_id="t",
        suite="short_chat",
        prompt_id="chat-03",
        concurrency=1,
        ok=True,
        prompt_chars=20,
        output_tokens=8,
        start_ns=0,
        first_token_ns=1_000_000,
        last_token_ns=2_000_000,
        end_ns=3_000_000,
        server_timings=ServerTimings(completion_tokens=8, completion_ms=200.0),
    )
    fill_trial_derived(trial)
    assert FLAG_DECODE_DISAGREE not in trial.flags


def test_bimodal_prefill_ratio_flags_large_p90_p50() -> None:
    tight = [1400.0, 1450.0, 1500.0, 1550.0]
    assert bimodal_prefill_ratio(tight) is None
    spread = [1400.0, 1400.0, 1400.0, 12_000.0]
    ratio = bimodal_prefill_ratio(spread)
    assert ratio is not None
    assert ratio >= 3.0
    assert bimodal_prefill_ratio(spread[:3]) is None


def test_trial_prefill_time_prefers_uncached_ms() -> None:
    trial = Trial(
        trial_id="t",
        suite="long_context",
        prompt_id="lc-01",
        concurrency=1,
        ok=True,
        prompt_chars=1000,
        output_tokens=8,
        start_ns=0,
        first_token_ns=1,
        last_token_ns=2,
        end_ns=3,
        server_timings=ServerTimings(uncached_prompt_ms=11_000.0, prompt_ms=1_400.0),
    )
    assert trial_prefill_time_ms(trial) == 11_000.0
