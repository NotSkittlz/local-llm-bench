from __future__ import annotations

import time

from llmbench.clocks import check_trial_clocks, interval_ns, wall_vs_monotonic_drift_ms


def test_perf_counter_is_monotonic() -> None:
    a = interval_ns()
    b = interval_ns()
    assert b >= a


def test_clock_anomalies_detect_inverted_range() -> None:
    flags = check_trial_clocks(
        start_ns=100,
        first_byte_ns=90,
        first_token_ns=80,
        last_token_ns=70,
        end_ns=50,
    )
    codes = {f.code for f in flags}
    assert "end_before_start" in codes
    assert "first_byte_before_start" in codes
    assert "first_token_before_first_byte" in codes


def test_healthy_trial_has_no_anomalies() -> None:
    flags = check_trial_clocks(
        start_ns=0,
        first_byte_ns=10,
        first_token_ns=20,
        last_token_ns=50,
        end_ns=60,
    )
    assert flags == []


def test_wall_vs_monotonic_small_on_sleep() -> None:
    start_m = time.perf_counter_ns()
    start_w = time.time_ns()
    time.sleep(0.02)
    drift = wall_vs_monotonic_drift_ms(
        time.perf_counter_ns() - start_m,
        time.time_ns() - start_w,
    )
    # 20ms sleep; drift should be well under 50ms on a healthy clock.
    assert drift < 50.0
