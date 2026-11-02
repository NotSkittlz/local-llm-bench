"""Pure metric math. No I/O, no backend calls.

Token accounting:

* **TTFT** is client-observed time from request start to first *content* token.
  It includes queueing, prefill, and the first decode step. It is not "prefill
  time" unless the server says so via ``ServerTimings``.
* **Decode tok/s** uses ``max(output_tokens - 1, 0) / (end - first_token)``.
  The first token is attributed to TTFT, not the decode loop.
* **Prefill tok/s** prefers server prompt timings. Otherwise it is the
  *proxy* ``prompt_tokens / TTFT``, which over-states prefill work.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence

from llmbench.clocks import ns_to_ms, ns_to_s
from llmbench.schema import PercentileSummary, ServerTimings, Trial


def percentile(xs: Sequence[float], p: float) -> float | None:
    """Linear-interpolation percentile. ``p`` is in 0..100."""

    if not xs:
        return None
    if not 0.0 <= p <= 100.0:
        raise ValueError(f"percentile p must be in [0, 100], got {p}")
    seq = sorted(xs)
    if len(seq) == 1:
        return float(seq[0])
    k = (p / 100.0) * (len(seq) - 1)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return float(seq[int(k)])
    weight = k - lo
    return float(seq[lo] * (1.0 - weight) + seq[hi] * weight)


def summarize(xs: Iterable[float]) -> PercentileSummary:
    values = [float(x) for x in xs if x is not None and not math.isnan(x)]
    if not values:
        return PercentileSummary(n=0)
    return PercentileSummary(
        n=len(values),
        mean=statistics.fmean(values),
        p50=percentile(values, 50),
        p95=percentile(values, 95),
        min=min(values),
        max=max(values),
    )


def trial_ttft_ms(start_ns: int, first_token_ns: int | None) -> float | None:
    if first_token_ns is None:
        return None
    return ns_to_ms(first_token_ns - start_ns)


def trial_e2e_ms(start_ns: int, end_ns: int) -> float | None:
    return ns_to_ms(end_ns - start_ns)


def trial_decode_ms(first_token_ns: int | None, last_token_ns: int | None) -> float | None:
    if first_token_ns is None or last_token_ns is None:
        return None
    return ns_to_ms(last_token_ns - first_token_ns)


def trial_decode_tps(
    output_tokens: int | None,
    first_token_ns: int | None,
    last_token_ns: int | None,
) -> float | None:
    if output_tokens is None or output_tokens <= 1:
        return None
    if first_token_ns is None or last_token_ns is None:
        return None
    decode_s = ns_to_s(last_token_ns - first_token_ns)
    if decode_s is None or decode_s <= 0:
        return None
    return (output_tokens - 1) / decode_s


def trial_prefill_tps(
    prompt_tokens: int | None,
    ttft_ms: float | None,
    server: ServerTimings | None,
) -> float | None:
    if server is not None and server.prompt_tokens and server.prompt_ms:
        if server.prompt_ms > 0:
            return server.prompt_tokens / (server.prompt_ms / 1000.0)
    if prompt_tokens is None or ttft_ms is None or ttft_ms <= 0:
        return None
    return prompt_tokens / (ttft_ms / 1000.0)


def fill_trial_derived(trial: Trial) -> Trial:
    """Populate TTFT / decode / prefill fields from raw timestamps."""

    trial.ttft_ms = trial_ttft_ms(trial.start_ns, trial.first_token_ns)
    trial.e2e_ms = trial_e2e_ms(trial.start_ns, trial.end_ns)
    trial.decode_ms = trial_decode_ms(trial.first_token_ns, trial.last_token_ns)
    trial.decode_tps = trial_decode_tps(
        trial.output_tokens,
        trial.first_token_ns,
        trial.last_token_ns,
    )
    trial.prefill_tps = trial_prefill_tps(
        trial.prompt_tokens,
        trial.ttft_ms,
        trial.server_timings,
    )
    return trial


def approx_tokens_from_chars(n_chars: int) -> int:
    """Last-resort token estimate (~4 chars/token). Mark token_source=approximate."""

    return max(1, n_chars // 4)
