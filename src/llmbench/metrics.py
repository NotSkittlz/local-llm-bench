"""Pure metric math. No I/O, no backend calls.

Token accounting:

* **TTFT** (``ttft_ms``) is client-observed time from request start to the
  first generated token of *any* kind (reasoning or content).
* **TTFT-to-content** (``ttft_content_ms``) is time to the first *content*
  token — time to answer on a reasoning model.
* **Decode tok/s** prefers server-reported rates when present (Ollama
  ``eval_count / eval_duration``, llama.cpp ``predicted_n / predicted_ms``).
  The client fallback is ``(output_tokens - 1) / (last - first)`` over the
  full generated span (first token of any kind → last token of any kind).
  A >25% client-vs-server disagreement is flagged on the trial.
* **Prefill tok/s** is uncached prompt tokens / uncached prompt milliseconds
  from ``ServerTimings``. Missing uncached counts yield ``null``, never a
  TTFT proxy labelled as server prefill. The proxy is stored separately as
  ``prefill_tps_proxy`` and is never mixed into the same percentile.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence

from llmbench.clocks import ns_to_ms, ns_to_s
from llmbench.schema import PercentileSummary, ServerTimings, Trial

DECODE_DISAGREE_RATIO = 0.25
FLAG_DECODE_DISAGREE = "decode_client_server_disagree"
# llama.cpp always caches the chat-template prefix (~42 tokens on Qwen).
TEMPLATE_PREFIX_MAX_TOKENS = 64
CACHE_HIT_MIN_FRACTION = 0.25


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
    """Client decode: (n - 1) / span from first generated token of any kind."""

    if output_tokens is None or output_tokens <= 1:
        return None
    if first_token_ns is None or last_token_ns is None:
        return None
    decode_s = ns_to_s(last_token_ns - first_token_ns)
    if decode_s is None or decode_s <= 0:
        return None
    return (output_tokens - 1) / decode_s


def trial_decode_tps_server(server: ServerTimings | None) -> float | None:
    """Server decode: completion_tokens / completion_ms (Ollama eval, llama.cpp predicted)."""

    if server is None:
        return None
    tokens = server.completion_tokens
    ms = server.completion_ms
    if tokens is None or ms is None or tokens <= 1 or ms <= 0:
        return None
    return tokens / (ms / 1000.0)


def decode_disagrees(client: float | None, server: float | None) -> bool:
    if client is None or server is None or server <= 0:
        return False
    return abs(client - server) / server > DECODE_DISAGREE_RATIO


def classify_cache_hit(
    cached: int | None,
    prompt_tokens: int | None,
    uncached: int | None = None,
) -> tuple[bool | None, float | None]:
    """A few cached prefix tokens are not a prompt-body cache hit."""

    fraction = None
    if cached is not None and prompt_tokens is not None and prompt_tokens > 0:
        fraction = cached / prompt_tokens
    if cached is None:
        return None, fraction
    if cached <= 0:
        return False, 0.0 if fraction is None else fraction
    if uncached == 0 and prompt_tokens is not None and prompt_tokens > TEMPLATE_PREFIX_MAX_TOKENS:
        return True, fraction
    if cached <= TEMPLATE_PREFIX_MAX_TOKENS:
        return False, fraction
    if fraction is not None:
        return fraction >= CACHE_HIT_MIN_FRACTION, fraction
    return True, fraction


def trial_prefill_tps(server: ServerTimings | None) -> float | None:
    """Uncached prompt tokens / uncached prompt ms. Null if either is missing."""

    if server is None:
        return None
    tokens = server.uncached_prompt_tokens
    ms = server.uncached_prompt_ms
    if ms is None:
        ms = server.prompt_ms
    if tokens is None or ms is None or tokens <= 0 or ms <= 0:
        return None
    return tokens / (ms / 1000.0)


def trial_prefill_tps_proxy(
    prompt_tokens: int | None,
    ttft_ms: float | None,
) -> float | None:
    """Client proxy: prompt_tokens / TTFT. Over-states prefill; never call it server prefill."""

    if prompt_tokens is None or ttft_ms is None or prompt_tokens <= 0 or ttft_ms <= 0:
        return None
    return prompt_tokens / (ttft_ms / 1000.0)


def fill_trial_derived(trial: Trial) -> Trial:
    """Populate TTFT / decode / prefill fields from raw timestamps."""

    trial.ttft_ms = trial_ttft_ms(trial.start_ns, trial.first_token_ns)
    trial.ttft_content_ms = trial_ttft_ms(trial.start_ns, trial.first_content_ns)
    trial.e2e_ms = trial_e2e_ms(trial.start_ns, trial.end_ns)
    trial.decode_ms = trial_decode_ms(trial.first_token_ns, trial.last_token_ns)
    client = trial_decode_tps(
        trial.output_tokens,
        trial.first_token_ns,
        trial.last_token_ns,
    )
    server = trial_decode_tps_server(trial.server_timings)
    trial.decode_tps_client = client
    trial.decode_tps_server = server
    if server is not None:
        trial.decode_tps = server
        trial.decode_source = "server"
    elif client is not None:
        trial.decode_tps = client
        trial.decode_source = "client"
    else:
        trial.decode_tps = None
        trial.decode_source = None
    if decode_disagrees(client, server) and FLAG_DECODE_DISAGREE not in trial.flags:
        trial.flags.append(FLAG_DECODE_DISAGREE)
    trial.prefill_tps = trial_prefill_tps(trial.server_timings)
    trial.prefill_tps_proxy = trial_prefill_tps_proxy(trial.prompt_tokens, trial.ttft_ms)
    if trial.prefill_tps is not None:
        trial.prefill_source = "server_uncached"
    elif trial.prefill_tps_proxy is not None:
        trial.prefill_source = "proxy_ttft"
    else:
        trial.prefill_source = None
    return trial


def approx_tokens_from_chars(n_chars: int) -> int:
    """Last-resort token estimate (~4 chars/token). Mark token_source=approximate."""

    return max(1, n_chars // 4)
