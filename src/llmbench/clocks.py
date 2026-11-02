"""Timing primitives for the benchmark harness.

Interval measurements **must** use ``time.perf_counter_ns``:

* It is monotonic (NTP steps and POSIX ``clock_settime`` do not move it).
* It is the highest-resolution clock Python exposes for elapsed time.

Wall-clock timestamps (when a run started) use ``time.time_ns`` and are
**not** used to compute TTFT, decode rate, or any duration.

The hot path in adapters calls ``time.perf_counter_ns`` directly to avoid an
extra Python function call per token. Helpers here are for setup, overhead
calibration, and anomaly checks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


def interval_ns() -> int:
    """Monotonic nanoseconds suitable for subtracting from a later sample."""

    return time.perf_counter_ns()


def wall_ns() -> int:
    """Unix epoch nanoseconds; may jump. Do not subtract for intervals."""

    return time.time_ns()


def ns_to_ms(ns: int | float | None) -> float | None:
    if ns is None:
        return None
    return ns / 1_000_000.0


def ns_to_s(ns: int | float | None) -> float | None:
    if ns is None:
        return None
    return ns / 1_000_000_000.0


@dataclass(frozen=True, slots=True)
class ClockAnomaly:
    """A timing relationship that cannot be physically right."""

    code: str
    detail: str


def check_trial_clocks(
    *,
    start_ns: int,
    first_byte_ns: int | None,
    first_token_ns: int | None,
    last_token_ns: int | None,
    end_ns: int,
) -> list[ClockAnomaly]:
    """Return anomalies rather than mutating timings.

    Callers record the flags on the trial so a fast-looking number that came
    from a broken clock relationship is visible in the report.
    """

    anomalies: list[ClockAnomaly] = []
    if end_ns < start_ns:
        anomalies.append(
            ClockAnomaly(
                "end_before_start",
                f"end_ns={end_ns} < start_ns={start_ns}",
            )
        )
    if first_byte_ns is not None and first_byte_ns < start_ns:
        anomalies.append(
            ClockAnomaly(
                "first_byte_before_start",
                f"first_byte_ns={first_byte_ns} < start_ns={start_ns}",
            )
        )
    if first_token_ns is not None and first_byte_ns is not None and first_token_ns < first_byte_ns:
        anomalies.append(
            ClockAnomaly(
                "first_token_before_first_byte",
                f"first_token_ns={first_token_ns} < first_byte_ns={first_byte_ns}",
            )
        )
    if last_token_ns is not None and first_token_ns is not None and last_token_ns < first_token_ns:
        anomalies.append(
            ClockAnomaly(
                "last_token_before_first",
                f"last_token_ns={last_token_ns} < first_token_ns={first_token_ns}",
            )
        )
    if first_token_ns is not None and first_token_ns > end_ns:
        anomalies.append(
            ClockAnomaly(
                "first_token_after_end",
                f"first_token_ns={first_token_ns} > end_ns={end_ns}",
            )
        )
    wall_span = None
    try:
        # Cheap sanity: if the process slept, perf counter should still move.
        _ = wall_span
    except Exception:  # pragma: no cover - placeholder kept for symmetry
        pass
    return anomalies


def wall_vs_monotonic_drift_ms(monotonic_elapsed_ns: int, wall_elapsed_ns: int) -> float:
    """Absolute drift between wall clock and perf counter, in milliseconds."""

    return abs(wall_elapsed_ns - monotonic_elapsed_ns) / 1_000_000.0
