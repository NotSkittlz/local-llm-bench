"""Stable result schema for JSON/CSV export.

Schema version ``1.0.0`` is the v0.1 contract. Fields may be added in a
backward-compatible way; renaming or changing units requires a major bump.

All durations in the serialized document are **milliseconds** (float) or
**nanoseconds** (int) as indicated by the suffix. Rates are tokens/second.
Memory is **mebibytes** (MiB).
"""

from __future__ import annotations

import sys
from typing import Any, Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0.0"

DataLineage = Literal["measured", "synthetic_mock"]


class PercentileSummary(BaseModel):
    """Distribution summary. Empty input yields nulls, not zeros."""

    n: int
    mean: float | None = None
    p50: float | None = None
    p95: float | None = None
    min: float | None = None
    max: float | None = None


class SamplingParams(BaseModel):
    """Requested sampling configuration and an audit of what went on the wire.

    The harness never silently rewrites temperature / max_tokens for a speed
    run. Flavor-specific names are mapped (vLLM ``repetition_penalty``);
    fields a backend will ignore are listed in ``dropped`` rather than
    displayed as applied. A quality probe uses a *separate* request.
    """

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int | None = None
    min_p: float | None = None
    repeat_penalty: float | None = None
    seed: int = 42
    max_tokens: int = 128
    max_tokens_min: int | None = None
    max_tokens_max: int | None = None
    stop: list[str] = Field(default_factory=list)
    applied: dict[str, Any] = Field(default_factory=dict)
    dropped: list[str] = Field(default_factory=list)


class ModelInfo(BaseModel):
    name: str
    quantization: str | None = None
    quantization_source: Literal["user_declared"] | None = None
    context_length: int | None = None
    context_length_source: Literal["user_declared", "applied_num_ctx"] | None = None
    family: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class BackendInfo(BaseModel):
    type: str
    base_url: str | None = None
    version: str | None = None
    model_listed: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class GPUInfo(BaseModel):
    index: int = 0
    name: str
    vendor: str | None = None
    vram_total_mib: float | None = None
    vram_source: str | None = None


class HardwareInfo(BaseModel):
    os: str
    os_release: str
    arch: str
    hostname: str | None = None
    cpu_name: str | None = None
    cpu_cores_logical: int | None = None
    cpu_cores_physical: int | None = None
    ram_total_mib: float | None = None
    gpus: list[GPUInfo] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class MemorySample(BaseModel):
    t_ms: float
    client_rss_mib: float | None = None
    backend_rss_mib: float | None = None
    gpu_vram_used_mib: float | None = None
    gpu_vram_total_mib: float | None = None
    gpu_index: int | None = None
    source: str


class MemorySummary(BaseModel):
    sample_interval_ms: float
    n_samples: int
    client_rss_peak_mib: float | None = None
    backend_rss_peak_mib: float | None = None
    gpu_vram_peak_mib: float | None = None
    gpu_vram_total_mib: float | None = None
    gpu_source: str | None = None
    sampler: str = "thread"


class HarnessOverhead(BaseModel):
    """Client-side cost measured against the in-process mock backend.

    ``zero_delay_ttft_ms`` is the time from ``perf_counter_ns`` before
    issuing a mock completion to the first mock token when the mock is
    configured with zero simulated compute. That is the floor the harness
    adds to every TTFT number.
    """

    zero_delay_ttft_ms: float
    zero_delay_e2e_ms: float
    timestamp_cost_ns: float
    tokens_in_probe: int
    iterations: int
    notes: list[str] = Field(default_factory=list)


class HarnessInfo(BaseModel):
    name: str = "local-llm-bench"
    version: str
    schema_version: str = SCHEMA_VERSION
    python: str = Field(default_factory=lambda: sys.version.split()[0])
    overhead: HarnessOverhead | None = None


class ServerTimings(BaseModel):
    """Optional timings reported by the server itself (Ollama, llama.cpp).

    Prefill tok/s uses **uncached** prompt tokens and **uncached** prompt
    milliseconds only. Cached tokens (llama.cpp ``cache_n``, Ollama
    ``prompt_eval_cached_count``) are stored so a cache hit is visible
    instead of looking like infinite prefill.
    """

    prompt_tokens: int | None = None
    prompt_ms: float | None = None
    cached_prompt_tokens: int | None = None
    uncached_prompt_tokens: int | None = None
    uncached_prompt_ms: float | None = None
    completion_tokens: int | None = None
    completion_ms: float | None = None
    load_ms: float | None = None
    total_ms: float | None = None
    cache_hit: bool | None = None
    cached_prompt_fraction: float | None = None


class Trial(BaseModel):
    """One completion, warmup or measured."""

    trial_id: str
    suite: str
    prompt_id: str
    concurrency: int
    wave: int = 0
    warmup: bool = False
    ok: bool
    error: str | None = None
    error_kind: str | None = None
    clock_anomalies: list[str] = Field(default_factory=list)

    prompt_chars: int
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    output_chars: int | None = None

    start_ns: int
    first_byte_ns: int | None = None
    first_token_ns: int | None = None
    first_content_ns: int | None = None
    last_token_ns: int | None = None
    end_ns: int

    ttft_ms: float | None = None
    ttft_content_ms: float | None = None
    e2e_ms: float | None = None
    decode_ms: float | None = None
    decode_tps: float | None = None
    decode_tps_client: float | None = None
    decode_tps_server: float | None = None
    decode_source: Literal["server", "client"] | None = None
    prefill_tps: float | None = None
    prefill_tps_proxy: float | None = None
    prefill_source: Literal["server_uncached", "proxy_ttft"] | None = None
    inter_token_ms: list[float] | None = None

    server_timings: ServerTimings | None = None
    token_source: (
        Literal["server_usage", "server_timings", "count_chunks", "approximate"] | None
    ) = None
    reasoning_tokens: int | None = None
    content_tokens: int | None = None
    max_tokens: int | None = None
    finish_reason: str | None = None
    flags: list[str] = Field(default_factory=list)


class SuiteSummary(BaseModel):
    name: str
    concurrency: int
    n_ok: int
    n_err: int
    n_warmup: int
    ttft_ms: PercentileSummary
    e2e_ms: PercentileSummary
    decode_tps: PercentileSummary
    prefill_tps: PercentileSummary
    prefill_tps_proxy: PercentileSummary | None = None
    output_tokens: PercentileSummary
    cache_hits: int = 0
    throughput_output_tps: float | None = None
    throughput_requests_per_s: float | None = None
    wall_ms: float | None = None
    notes: list[str] = Field(default_factory=list)


class QualityProbe(BaseModel):
    enabled: bool
    ok: bool | None = None
    greedy: bool = True
    temperature: float = 0.0
    top_p: float = 1.0
    prompt: str | None = None
    output_text: str | None = None
    output_sha256: str | None = None
    expected_sha256: str | None = None
    expected_contains: str | None = None
    contains_hit: bool | None = None
    hash_match: bool | None = None
    ttft_ms: float | None = None
    output_tokens: int | None = None
    notes: list[str] = Field(default_factory=list)


class RunConfigSnapshot(BaseModel):
    seed: int
    warmup: int
    iterations: int
    concurrency: list[int]
    timeout_s: float
    suites: list[str]
    record_inter_token: bool = False
    memory_sample_interval_ms: float = 250.0
    quality_enabled: bool = True
    collect_output_text: bool = False


class RunResult(BaseModel):
    schema_version: str = SCHEMA_VERSION
    run_id: str
    created_at: str
    data_lineage: DataLineage = "measured"
    synthetic: bool = False
    label: str | None = None
    notes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    config: RunConfigSnapshot
    sampling: SamplingParams
    model: ModelInfo
    backend: BackendInfo
    hardware: HardwareInfo
    harness: HarnessInfo
    quality: QualityProbe | None = None
    memory: MemorySummary | None = None
    memory_samples: list[MemorySample] = Field(default_factory=list)

    summaries: list[SuiteSummary] = Field(default_factory=list)
    trials: list[Trial] = Field(default_factory=list)

    def measured_trials(self) -> list[Trial]:
        return [t for t in self.trials if not t.warmup]
