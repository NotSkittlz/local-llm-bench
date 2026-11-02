"""Optional quality sanity check.

A speed number is meaningless if the server started sampling garbage or
switched quantization. This probe is a *separate* greedy request; it never
rewrites the sampling parameters of the timed suite.

Checks (all optional except "did we get text"):

* SHA-256 of the greedy output, compared to a previously recorded digest.
* Substring match (default: the France/Paris prompt).
"""

from __future__ import annotations

import hashlib

from llmbench.backends.base import Backend, CompletionRequest
from llmbench.config import BenchConfig
from llmbench.schema import QualityProbe, SamplingParams


async def run_quality_probe(backend: Backend, cfg: BenchConfig) -> QualityProbe:
    q = cfg.quality
    if not q.enabled:
        return QualityProbe(enabled=False, notes=["quality probe disabled"])

    greedy = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        seed=cfg.seed,
        max_tokens=q.max_tokens,
    )
    request = CompletionRequest(
        prompt=q.prompt,
        sampling=greedy,
        collect_text=True,
        prompt_id="quality",
    )
    notes = [
        "quality probe uses greedy sampling and does not alter suite sampling params",
        f"suite sampling remains temperature={cfg.sampling.temperature} top_p={cfg.sampling.top_p}",
    ]
    try:
        result = await backend.complete(request)
    except Exception as exc:  # connection errors surface as a failed probe
        return QualityProbe(
            enabled=True,
            ok=False,
            prompt=q.prompt,
            notes=[*notes, f"probe request failed: {exc}"],
        )

    text = result.text or ""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest() if text else None
    contains_hit = None
    if q.expected_contains:
        contains_hit = q.expected_contains.lower() in text.lower()
    hash_match = None
    if q.expected_sha256 and digest:
        hash_match = digest == q.expected_sha256.lower()

    ok = result.error is None and bool(text)
    if contains_hit is False:
        ok = False
    if hash_match is False:
        ok = False

    ttft_ms = None
    if result.first_token_ns is not None:
        ttft_ms = (result.first_token_ns - result.start_ns) / 1e6

    if result.error:
        notes.append(result.error)
    if not text:
        notes.append("empty greedy output")

    return QualityProbe(
        enabled=True,
        ok=ok,
        greedy=True,
        temperature=0.0,
        top_p=1.0,
        prompt=q.prompt,
        output_text=text[:2000] if text else None,
        output_sha256=digest,
        expected_sha256=q.expected_sha256,
        expected_contains=q.expected_contains,
        contains_hit=contains_hit,
        hash_match=hash_match,
        ttft_ms=ttft_ms,
        output_tokens=result.output_tokens,
        notes=notes,
    )
