"""Cheap checks before a suite wave so we do not fire doomed requests."""

from __future__ import annotations

from typing import Any

from llmbench.config import BenchConfig
from llmbench.metrics import approx_tokens_from_chars, summarize
from llmbench.schema import BackendInfo, SuiteSummary
from llmbench.workloads import load_suite


def server_n_ctx(info: BackendInfo, cfg: BenchConfig) -> tuple[int | None, int | None, str]:
    """Return (n_ctx_per_slot, parallel_slots, source label).

    llama.cpp ``default_generation_settings.n_ctx`` is the **per-slot** window
    (``n_ctx_slot``), not a shared pool to multiply by concurrency.
    Ollama ``num_ctx`` is likewise per request; ``OLLAMA_NUM_PARALLEL`` is the
    slot count when the server exposes it.
    """

    extra = info.extra or {}
    if cfg.backend.type == "llamacpp":
        props = extra.get("props") if isinstance(extra.get("props"), dict) else {}
        slots = _as_int(props.get("total_slots")) if isinstance(props, dict) else None
        dgs = props.get("default_generation_settings") if isinstance(props, dict) else None
        n_ctx = _as_int(dgs.get("n_ctx")) if isinstance(dgs, dict) else None
        return n_ctx, slots, "llama.cpp /props n_ctx (per slot)"
    if cfg.backend.type == "ollama":
        n_ctx, ctx_source = _ollama_n_ctx(info, cfg)
        slots = _as_int(extra.get("num_parallel")) or _as_int(extra.get("parallel"))
        return n_ctx, slots, ctx_source
    return None, None, ""


def estimate_suite_budget(cfg: BenchConfig, suite: str) -> tuple[int, int]:
    prompts = load_suite(suite, cfg)
    prompt_tokens = max(approx_tokens_from_chars(len(p.text)) for p in prompts)
    max_tokens = max(p.max_tokens or cfg.sampling.max_tokens for p in prompts)
    return prompt_tokens, max_tokens


def context_overflow_message(
    cfg: BenchConfig,
    info: BackendInfo,
    *,
    suite: str,
    concurrency: int,
) -> str | None:
    """Skip only when a *single* request cannot fit in the per-slot window."""

    n_ctx, slots, source = server_n_ctx(info, cfg)
    if n_ctx is None or n_ctx <= 0:
        return None
    prompt_tokens, max_tokens = estimate_suite_budget(cfg, suite)
    needed = prompt_tokens + max_tokens
    if needed <= n_ctx:
        return None
    slot_bit = f", total_slots={slots}" if slots is not None else ""
    return (
        f"skipping {suite} concurrency={concurrency}: "
        f"~{prompt_tokens} prompt + {max_tokens} max_tokens = {needed} tokens "
        f"exceeds per-slot server context {n_ctx} ({source}{slot_bit})"
    )


def context_queueing_note(
    cfg: BenchConfig,
    info: BackendInfo,
    *,
    suite: str,
    concurrency: int,
) -> str | None:
    """When C is above parallel slots the extra requests queue; do not skip."""

    _n_ctx, slots, source = server_n_ctx(info, cfg)
    if slots is None or slots <= 0 or concurrency <= slots:
        return None
    return (
        f"{suite} concurrency={concurrency} exceeds parallel slots={slots} "
        f"({source}); extra in-flight requests will queue"
    )


def skipped_summary(suite: str, concurrency: int, message: str) -> SuiteSummary:
    return SuiteSummary(
        name=suite,
        concurrency=concurrency,
        n_ok=0,
        n_err=0,
        n_warmup=0,
        ttft_ms=summarize([]),
        e2e_ms=summarize([]),
        decode_tps=summarize([]),
        prefill_tps=summarize([]),
        output_tokens=summarize([]),
        notes=[message],
    )


def _ollama_n_ctx(info: BackendInfo, cfg: BenchConfig) -> tuple[int | None, str]:
    if cfg.model.context_length is not None:
        return cfg.model.context_length, "Ollama options.num_ctx"
    extra = info.extra or {}
    num_ctx = _as_int(extra.get("num_ctx"))
    if num_ctx is not None and num_ctx > 0:
        return num_ctx, "Ollama /api/show num_ctx"
    ctx = _as_int(extra.get("context_length"))
    if ctx is not None and ctx > 0:
        return ctx, "Ollama /api/show context_length"
    return None, "Ollama (num_ctx unknown)"


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None
