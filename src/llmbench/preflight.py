"""Cheap checks before a suite wave so we do not fire doomed requests."""

from __future__ import annotations

from typing import Any

from llmbench.config import BenchConfig
from llmbench.metrics import approx_tokens_from_chars, summarize
from llmbench.schema import BackendInfo, SuiteSummary
from llmbench.workloads import load_suite


def server_n_ctx(info: BackendInfo, cfg: BenchConfig) -> tuple[int | None, int | None, str]:
    """Return (n_ctx, total_slots, source label)."""

    extra = info.extra or {}
    if cfg.backend.type == "llamacpp":
        props = extra.get("props") if isinstance(extra.get("props"), dict) else {}
        slots = _as_int(props.get("total_slots")) if isinstance(props, dict) else None
        dgs = props.get("default_generation_settings") if isinstance(props, dict) else None
        n_ctx = _as_int(dgs.get("n_ctx")) if isinstance(dgs, dict) else None
        return n_ctx, slots, "llama.cpp /props n_ctx"
    if cfg.backend.type == "ollama":
        n_ctx = cfg.model.context_length
        source = "Ollama options.num_ctx" if n_ctx is not None else "Ollama (num_ctx unknown)"
        return n_ctx, None, source
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
    n_ctx, slots, source = server_n_ctx(info, cfg)
    if n_ctx is None or n_ctx <= 0:
        return None
    prompt_tokens, max_tokens = estimate_suite_budget(cfg, suite)
    needed = concurrency * (prompt_tokens + max_tokens)
    if needed <= n_ctx:
        return None
    slot_bit = f", total_slots={slots}" if slots is not None else ""
    return (
        f"skipping {suite} concurrency={concurrency}: "
        f"{concurrency} x (~{prompt_tokens} prompt + {max_tokens} max_tokens) "
        f"= {needed} tokens exceeds server context {n_ctx} ({source}{slot_bit})"
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


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None
