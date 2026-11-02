"""Benchmark orchestrator.

The runner is deliberately boring on the hot path: issue completions, wait,
record ``StreamResult`` fields. Aggregation happens after the suite wave
finishes. Memory sampling is a side thread started before the first warmup.
"""

from __future__ import annotations

import asyncio
import sys
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

from llmbench import __version__
from llmbench.backends import Backend, build_backend
from llmbench.backends.base import CompletionRequest, StreamResult
from llmbench.clocks import check_trial_clocks, wall_ns, wall_vs_monotonic_drift_ms
from llmbench.config import BenchConfig
from llmbench.exceptions import ConfigError, LLMBenchError
from llmbench.hardware import probe_hardware
from llmbench.memory import MemorySampler
from llmbench.metrics import fill_trial_derived, summarize
from llmbench.overhead import measure_harness_overhead
from llmbench.quality import run_quality_probe
from llmbench.sampling import map_sampling
from llmbench.schema import (
    BackendInfo,
    HarnessInfo,
    ModelInfo,
    RunConfigSnapshot,
    RunResult,
    SamplingParams,
    SuiteSummary,
    Trial,
)
from llmbench.workloads import Prompt, load_suite, split_suite

ProgressCb = Callable[[str], None]


async def run_benchmark(
    cfg: BenchConfig,
    *,
    progress: ProgressCb | None = None,
    backend: Backend | None = None,
) -> RunResult:
    _validate(cfg)
    own_backend = backend is None
    backend = backend or build_backend(cfg)
    run_id = uuid.uuid4().hex
    created = datetime.now(UTC).isoformat()
    emit = progress or (lambda _msg: None)
    warnings: list[str] = []
    notes: list[str] = []

    emit("probing hardware")
    hardware = probe_hardware()
    emit("measuring harness overhead (in-process mock)")
    overhead = await measure_harness_overhead()
    if overhead.zero_delay_ttft_ms > 5.0:
        warnings.append(
            f"harness zero-delay TTFT is {overhead.zero_delay_ttft_ms:.2f} ms; "
            "the client may be overloaded"
        )

    emit("querying backend metadata")
    try:
        backend_info = await backend.info()
    except LLMBenchError as exc:
        backend_info = BackendInfo(type=cfg.backend.type, base_url=cfg.backend.base_url)
        warnings.append(f"backend info failed: {exc}")
    except Exception as exc:
        backend_info = BackendInfo(type=cfg.backend.type, base_url=cfg.backend.base_url)
        warnings.append(f"backend info failed: {exc}")

    sampler: MemorySampler | None = None
    if cfg.memory.enabled:
        sampler = MemorySampler(
            interval_ms=cfg.memory.sample_interval_ms,
            backend_pid=cfg.backend.pid,
            gpu_index=cfg.memory.gpu_index,
        )
        sampler.start()

    quality = None
    if cfg.quality.enabled:
        emit("running quality probe (greedy, not part of speed stats)")
        quality = await run_quality_probe(backend, cfg)
        if quality.ok is False:
            warnings.append("quality probe failed; speed numbers may not be comparable")
            if cfg.quality.strict:
                if sampler:
                    sampler.stop()
                if own_backend:
                    await backend.aclose()
                raise ConfigError("quality probe failed and quality.strict is set")

    trials: list[Trial] = []
    summaries: list[SuiteSummary] = []
    wall0 = wall_ns()
    mono0 = time.perf_counter_ns()

    try:
        for suite in cfg.suites:
            load_suite(suite, cfg)  # fail fast on unknown
            for conc in cfg.concurrency:
                emit(f"suite {suite}  concurrency={conc}")
                suite_trials, summary = await _run_suite(
                    backend,
                    cfg,
                    suite=suite,
                    concurrency=conc,
                    emit=emit,
                )
                trials.extend(suite_trials)
                summaries.append(summary)
                if summary.n_err:
                    warnings.append(f"{suite} concurrency={conc}: {summary.n_err} failed trial(s)")
    finally:
        memory_summary = sampler.stop() if sampler else None
        memory_samples = sampler.samples if sampler else []
        if own_backend:
            await backend.aclose()

    drift = wall_vs_monotonic_drift_ms(time.perf_counter_ns() - mono0, wall_ns() - wall0)
    if drift > 250.0:
        warnings.append(
            f"wall clock drifted {drift:.0f} ms from perf_counter over the run; "
            "durations still use perf_counter"
        )

    synthetic = cfg.backend.type == "mock"
    lineage = "synthetic_mock" if synthetic else "measured"
    if cfg.backend.allow_remote and cfg.backend.base_url:
        warnings.append(
            f"allow_remote is set; prompts may leave this machine via {cfg.backend.base_url}"
        )
    if synthetic:
        notes.append("synthetic timings from the in-process mock backend; not hardware inference")
        if cfg.backend.mock_ttft_ns == 0 and cfg.backend.mock_inter_token_ns == 0:
            warnings.append("mock delays are zero; reported tok/s is harness speed, not a model")

    flavor = cfg.backend.type
    wire = map_sampling(flavor, cfg.sampling, context_length=cfg.model.context_length)
    sampling_snap = cfg.sampling.model_copy(deep=True)
    sampling_snap.applied = dict(wire.applied)
    sampling_snap.dropped = list(wire.dropped)
    sent_max = [t.max_tokens for t in trials if t.max_tokens is not None]
    if sent_max:
        sampling_snap.max_tokens_min = min(sent_max)
        sampling_snap.max_tokens_max = max(sent_max)
    for w in wire.warnings:
        if w not in warnings:
            warnings.append(w)

    q_source = "user_declared" if cfg.model.quantization else None
    if cfg.model.quantization:
        notes.append(
            "quantization is user-declared metadata; the harness does not load or quantize weights"
        )
    if cfg.backend.type == "ollama" and cfg.model.context_length is not None:
        ctx_source: str | None = "applied_num_ctx"
        notes.append("context_length sent as Ollama options.num_ctx")
    elif cfg.model.context_length is not None:
        ctx_source = "user_declared"
        notes.append("context_length is user-declared metadata; not sent as a server argument")
    else:
        ctx_source = None

    model = ModelInfo(
        name=cfg.model.name or cfg.backend.model,
        quantization=cfg.model.quantization,
        quantization_source=q_source,  # type: ignore[arg-type]
        context_length=cfg.model.context_length,
        context_length_source=ctx_source,  # type: ignore[arg-type]
        family=cfg.model.family,
    )

    return RunResult(
        run_id=run_id,
        created_at=created,
        data_lineage=lineage,  # type: ignore[arg-type]
        synthetic=synthetic,
        label=cfg.output.label,
        notes=notes,
        warnings=warnings,
        config=RunConfigSnapshot(
            seed=cfg.seed,
            warmup=cfg.warmup,
            iterations=cfg.iterations,
            concurrency=list(cfg.concurrency),
            timeout_s=cfg.timeout_s,
            suites=list(cfg.suites),
            record_inter_token=cfg.record_inter_token,
            memory_sample_interval_ms=cfg.memory.sample_interval_ms,
            quality_enabled=cfg.quality.enabled,
            collect_output_text=cfg.collect_output_text,
        ),
        sampling=sampling_snap,
        model=model,
        backend=backend_info,
        hardware=hardware,
        harness=HarnessInfo(version=__version__, python=sys.version.split()[0], overhead=overhead),
        quality=quality,
        memory=memory_summary,
        memory_samples=memory_samples,
        summaries=summaries,
        trials=trials,
    )


async def _run_suite(
    backend: Backend,
    cfg: BenchConfig,
    *,
    suite: str,
    concurrency: int,
    emit: ProgressCb,
) -> tuple[list[Trial], SuiteSummary]:
    warmup_prompts, measured_prompts = split_suite(cfg, suite, cfg.warmup, cfg.iterations)
    trials: list[Trial] = []

    for i, prompt in enumerate(warmup_prompts):
        emit(f"  warmup {i + 1}/{cfg.warmup}")
        trials.append(
            await _one_trial(
                backend,
                cfg,
                prompt=prompt,
                concurrency=1,
                wave=0,
                warmup=True,
            )
        )

    phase_start = time.perf_counter_ns()
    if concurrency == 1:
        for i, prompt in enumerate(measured_prompts):
            emit(f"  run {i + 1}/{cfg.iterations}")
            trials.append(
                await _one_trial(
                    backend,
                    cfg,
                    prompt=prompt,
                    concurrency=1,
                    wave=i,
                    warmup=False,
                )
            )
    else:
        # Keep `concurrency` in flight until `iterations` complete (open loop).
        sem = asyncio.Semaphore(concurrency)
        done = 0
        lock = asyncio.Lock()

        async def guarded(idx: int, prompt: Prompt) -> Trial:
            nonlocal done
            async with sem:
                trial = await _one_trial(
                    backend,
                    cfg,
                    prompt=prompt,
                    concurrency=concurrency,
                    wave=idx // concurrency,
                    warmup=False,
                )
            async with lock:
                done += 1
                emit(f"  run {done}/{cfg.iterations} (in-flight ≤ {concurrency})")
            return trial

        measured = await asyncio.gather(*[guarded(i, p) for i, p in enumerate(measured_prompts)])
        trials.extend(measured)
    phase_end = time.perf_counter_ns()

    measured = [t for t in trials if not t.warmup]
    ok_trials = [t for t in measured if t.ok]
    wall_ms = (phase_end - phase_start) / 1e6
    out_tokens = sum(t.output_tokens or 0 for t in ok_trials)
    wall_s = wall_ms / 1000.0 if wall_ms > 0 else None
    throughput_tps = (out_tokens / wall_s) if wall_s else None
    rps = (len(ok_trials) / wall_s) if wall_s else None
    notes: list[str] = []
    if concurrency > 1:
        notes.append(
            "throughput is completed output tokens / wall time of the measured phase "
            "(includes queueing when the server serializes)"
        )
    server_prefill = [t.prefill_tps for t in ok_trials if t.prefill_tps is not None]
    proxy_prefill = [t.prefill_tps_proxy for t in ok_trials if t.prefill_tps_proxy is not None]
    cache_hits = sum(
        1
        for t in ok_trials
        if t.server_timings is not None
        and (t.server_timings.cache_hit or (t.server_timings.cached_prompt_tokens or 0) > 0)
    )
    if server_prefill:
        notes.append("prefill tok/s is uncached prompt tokens / uncached prompt ms")
    elif proxy_prefill:
        notes.append(
            "server did not report uncached prompt timings; prefill p50 is omitted "
            "(prompt_tokens/TTFT proxy is stored separately and not mixed in)"
        )
    if cache_hits:
        notes.append(f"{cache_hits}/{len(ok_trials)} measured trials reported cached prompt tokens")
    summary = SuiteSummary(
        name=suite,
        concurrency=concurrency,
        n_ok=len(ok_trials),
        n_err=sum(1 for t in measured if not t.ok),
        n_warmup=sum(1 for t in trials if t.warmup),
        ttft_ms=summarize(t.ttft_ms for t in ok_trials if t.ttft_ms is not None),
        e2e_ms=summarize(t.e2e_ms for t in ok_trials if t.e2e_ms is not None),
        decode_tps=summarize(t.decode_tps for t in ok_trials if t.decode_tps is not None),
        prefill_tps=summarize(server_prefill),
        prefill_tps_proxy=summarize(proxy_prefill) if proxy_prefill else None,
        output_tokens=summarize(float(t.output_tokens or 0) for t in ok_trials),
        throughput_output_tps=throughput_tps,
        throughput_requests_per_s=rps,
        wall_ms=wall_ms,
        cache_hits=cache_hits,
        notes=notes,
    )
    return trials, summary


async def _one_trial(
    backend: Backend,
    cfg: BenchConfig,
    *,
    prompt: Prompt,
    concurrency: int,
    wave: int,
    warmup: bool,
) -> Trial:
    sampling = _sampling_for(cfg, prompt)
    request = CompletionRequest(
        prompt=prompt.text,
        sampling=sampling,
        collect_text=cfg.collect_output_text,
        record_inter_token=cfg.record_inter_token,
        prompt_id=prompt.prompt_id,
        context_length=cfg.model.context_length,
        flavor=cfg.backend.type,
    )
    try:
        result: StreamResult = await backend.complete(request)
    except LLMBenchError as exc:
        now = time.perf_counter_ns()
        trial = Trial(
            trial_id=uuid.uuid4().hex,
            suite=prompt.suite,
            prompt_id=prompt.prompt_id,
            concurrency=concurrency,
            wave=wave,
            warmup=warmup,
            ok=False,
            error=str(exc),
            error_kind=type(exc).__name__,
            prompt_chars=len(prompt.text),
            start_ns=now,
            end_ns=now,
            max_tokens=sampling.max_tokens,
        )
        return fill_trial_derived(trial)

    anomalies = check_trial_clocks(
        start_ns=result.start_ns,
        first_byte_ns=result.first_byte_ns,
        first_token_ns=result.first_token_ns,
        last_token_ns=result.last_token_ns,
        end_ns=result.end_ns,
    )
    inter_ms = None
    if result.inter_token_ns:
        inter_ms = [ns / 1e6 for ns in result.inter_token_ns]
    trial = Trial(
        trial_id=uuid.uuid4().hex,
        suite=prompt.suite,
        prompt_id=prompt.prompt_id,
        concurrency=concurrency,
        wave=wave,
        warmup=warmup,
        ok=result.error is None,
        error=result.error,
        error_kind=result.error_kind,
        clock_anomalies=[a.code for a in anomalies],
        prompt_chars=len(prompt.text),
        prompt_tokens=result.prompt_tokens,
        output_tokens=result.output_tokens,
        output_chars=result.output_chars,
        start_ns=result.start_ns,
        first_byte_ns=result.first_byte_ns,
        first_token_ns=result.first_token_ns,
        last_token_ns=result.last_token_ns,
        end_ns=result.end_ns,
        inter_token_ms=inter_ms,
        server_timings=result.server_timings,
        token_source=result.token_source,  # type: ignore[arg-type]
        max_tokens=sampling.max_tokens,
    )
    return fill_trial_derived(trial)


def _sampling_for(cfg: BenchConfig, prompt: Prompt) -> SamplingParams:
    sampling = cfg.sampling.model_copy()
    sampling.seed = cfg.seed
    if prompt.max_tokens is not None:
        sampling.max_tokens = prompt.max_tokens
    return sampling


def _validate(cfg: BenchConfig) -> None:
    if cfg.iterations < 1:
        raise ConfigError("iterations must be >= 1 for a measured run")
    if cfg.long_context_chars > 2_000_000:
        raise ConfigError("long_context_chars is capped at 2,000,000 to avoid client OOM")
