from __future__ import annotations

import pytest

from llmbench.backends.mock import MockBackend
from llmbench.config import BackendConfig, BenchConfig, MemoryConfig, QualityConfig
from llmbench.export import read_json, write_csv, write_json
from llmbench.runner import run_benchmark
from llmbench.schema import BackendInfo, SamplingParams


@pytest.mark.asyncio
async def test_runner_mock_populates_schema(tmp_path) -> None:
    cfg = BenchConfig(
        warmup=1,
        iterations=4,
        concurrency=[1, 2],
        suites=["short_chat"],
        backend=BackendConfig(
            type="mock",
            model="mock-7b",
            mock_output_tokens=8,
            mock_ttft_ns=2_000_000,
            mock_inter_token_ns=500_000,
        ),
        sampling=SamplingParams(temperature=0.2, max_tokens=8, seed=42),
        memory=MemoryConfig(sample_interval_ms=50),
        quality=QualityConfig(enabled=True),
    )
    backend = MockBackend(cfg.backend)
    result = await run_benchmark(cfg, backend=backend)
    assert result.synthetic is True
    assert result.data_lineage == "synthetic_mock"
    assert result.harness.overhead is not None
    assert result.quality is not None and result.quality.ok is True
    assert result.sampling.temperature == 0.2  # speed run params preserved
    assert {s.concurrency for s in result.summaries} == {1, 2}
    measured = [t for t in result.trials if not t.warmup]
    assert len(measured) == 8  # 4 iters * 2 conc levels
    assert all(t.ttft_ms is not None for t in measured if t.ok)
    assert result.memory is not None
    assert result.memory.n_samples >= 1

    json_path = tmp_path / "run.json"
    csv_path = tmp_path / "run.csv"
    write_json(result, json_path)
    write_csv(result, csv_path)
    again = read_json(json_path)
    assert again.run_id == result.run_id
    assert csv_path.read_text(encoding="utf-8").splitlines()[0].startswith("run_id")


@pytest.mark.asyncio
async def test_warmup_not_in_summary_n_ok() -> None:
    cfg = BenchConfig(
        warmup=2,
        iterations=3,
        concurrency=[1],
        suites=["code"],
        backend=BackendConfig(
            type="mock", mock_output_tokens=4, mock_ttft_ns=0, mock_inter_token_ns=0
        ),
        quality=QualityConfig(enabled=False),
        memory=MemoryConfig(enabled=False),
    )
    result = await run_benchmark(cfg)
    summary = result.summaries[0]
    assert summary.n_warmup == 2
    assert summary.n_ok == 3
    assert summary.n_err == 0


@pytest.mark.asyncio
async def test_runner_does_not_replay_warmup_prompts() -> None:
    cfg = BenchConfig(
        seed=42,
        warmup=2,
        iterations=8,
        concurrency=[1],
        suites=["short_chat"],
        backend=BackendConfig(
            type="mock",
            mock_output_tokens=2,
            mock_ttft_ns=0,
            mock_inter_token_ns=0,
        ),
        quality=QualityConfig(enabled=False),
        memory=MemoryConfig(enabled=False),
    )
    result = await run_benchmark(cfg)
    warm = [t.prompt_id for t in result.trials if t.warmup]
    measured = [t.prompt_id for t in result.trials if not t.warmup]
    assert warm != measured[:2]
    assert set(warm).isdisjoint(measured)


@pytest.mark.asyncio
async def test_runner_warmup_disjoint_when_iterations_exceed_suite() -> None:
    cfg = BenchConfig(
        seed=42,
        warmup=2,
        iterations=16,
        concurrency=[1],
        suites=["short_chat"],
        backend=BackendConfig(
            type="mock",
            mock_output_tokens=2,
            mock_ttft_ns=0,
            mock_inter_token_ns=0,
        ),
        quality=QualityConfig(enabled=False),
        memory=MemoryConfig(enabled=False),
    )
    result = await run_benchmark(cfg)
    warm = [t.prompt_id for t in result.trials if t.warmup]
    measured = [t.prompt_id for t in result.trials if not t.warmup]
    assert len(measured) == 16
    assert set(warm).isdisjoint(measured)


@pytest.mark.asyncio
async def test_long_context_sweep_does_not_reuse_prefixes() -> None:
    cfg = BenchConfig(
        seed=42,
        warmup=0,
        iterations=2,
        concurrency=[1, 4],
        suites=["long_context"],
        long_context_chars=4000,
        backend=BackendConfig(
            type="mock",
            mock_output_tokens=2,
            mock_ttft_ns=0,
            mock_inter_token_ns=0,
        ),
        quality=QualityConfig(enabled=False),
        memory=MemoryConfig(enabled=False),
    )
    result = await run_benchmark(cfg)
    measured = [t for t in result.trials if not t.warmup]
    assert len(measured) == 4
    ids = [t.prompt_id for t in measured]
    assert len(set(ids)) == 4
    assert any("-c1-" in i for i in ids)
    assert any("-c4-" in i for i in ids)


@pytest.mark.asyncio
async def test_code_suite_records_max_tokens_range() -> None:
    cfg = BenchConfig(
        warmup=0,
        iterations=6,
        concurrency=[1],
        suites=["code"],
        backend=BackendConfig(
            type="mock", mock_output_tokens=2, mock_ttft_ns=0, mock_inter_token_ns=0
        ),
        sampling=SamplingParams(max_tokens=128),
        quality=QualityConfig(enabled=False),
        memory=MemoryConfig(enabled=False),
    )
    result = await run_benchmark(cfg)
    assert result.sampling.max_tokens == 128
    assert result.sampling.max_tokens_min == 96
    assert result.sampling.max_tokens_max == 256
    assert {t.max_tokens for t in result.trials} <= {96, 128, 192, 256}
    assert min(t.max_tokens or 0 for t in result.trials) == 96
    html = __import__("llmbench.report", fromlist=["render_html"]).render_html([result])
    assert "96-256" in html


@pytest.mark.asyncio
async def test_summary_prefill_excludes_ttft_proxy() -> None:
    cfg = BenchConfig(
        warmup=0,
        iterations=3,
        concurrency=[1],
        suites=["short_chat"],
        backend=BackendConfig(
            type="mock",
            mock_output_tokens=4,
            mock_ttft_ns=2_000_000,
            mock_inter_token_ns=500_000,
            mock_prompt_eval_ns_per_token=0,
        ),
        quality=QualityConfig(enabled=False),
        memory=MemoryConfig(enabled=False),
    )
    result = await run_benchmark(cfg)
    summary = result.summaries[0]
    assert summary.prefill_tps.n == 0
    assert summary.prefill_tps.p50 is None
    assert all(t.prefill_tps is None for t in result.trials if t.ok)
    assert all(t.prefill_source == "not_meaningful_short_prompt" for t in result.trials if t.ok)
    assert any("256" in n for n in summary.notes)


class _ThinkingBackend(MockBackend):
    async def info(self) -> BackendInfo:
        return BackendInfo(type="mock", extra={"capabilities": ["completion", "thinking"]})


@pytest.mark.asyncio
async def test_thinking_model_warns_when_budget_is_too_small() -> None:
    cfg = BenchConfig(
        warmup=0,
        iterations=1,
        suites=["short_chat"],
        backend=BackendConfig(
            type="mock", mock_output_tokens=2, mock_ttft_ns=0, mock_inter_token_ns=0
        ),
        quality=QualityConfig(enabled=True, max_tokens=16),
        memory=MemoryConfig(enabled=False),
    )
    result = await run_benchmark(cfg, backend=_ThinkingBackend(cfg.backend))
    assert any("reasoning-capable" in w for w in result.warnings)
    assert result.config.thinking == "default"

    cfg.thinking = "off"
    quiet = await run_benchmark(cfg, backend=_ThinkingBackend(cfg.backend))
    assert not any("reasoning-capable" in w for w in quiet.warnings)
