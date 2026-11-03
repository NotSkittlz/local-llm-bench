from __future__ import annotations

import pytest

from llmbench.config import BackendConfig, BenchConfig, MemoryConfig, QualityConfig
from llmbench.report import _context_label, render_html, render_markdown
from llmbench.runner import run_benchmark
from llmbench.schema import ModelInfo, SamplingParams


@pytest.mark.asyncio
async def test_html_is_self_contained_and_flags_synthetic() -> None:
    cfg = BenchConfig(
        warmup=0,
        iterations=3,
        concurrency=[1],
        suites=["short_chat"],
        backend=BackendConfig(
            type="mock",
            mock_output_tokens=6,
            model="mock-7b",
            mock_ttft_ns=0,
            mock_inter_token_ns=0,
        ),
        sampling=SamplingParams(max_tokens=6),
        quality=QualityConfig(enabled=True),
        memory=MemoryConfig(enabled=False),
        output={"label": "demo-mock"},
    )
    run = await run_benchmark(cfg)
    html = render_html([run])
    assert "http://cdn" not in html.lower()
    assert "https://cdn" not in html.lower()
    assert "<script src=" not in html
    assert "Synthetic data" in html
    assert "data-theme" in html
    assert "Harness" in html or "harness" in html.lower()
    assert "TTFT" in html
    assert "µs" in html or "ms" in html
    assert "Prefill p50 (server)" in html
    assert "Prefill proxy p50" in html
    assert "51177" not in html
    # Server prefill is omitted when the mock has no uncached prompt_ms.
    assert "never mixed" in html.lower() or "not mixed" in html.lower() or "Prefill proxy" in html
    md = render_markdown([run])
    assert "Synthetic" in md
    assert "short_chat" in md


@pytest.mark.asyncio
async def test_compare_two_runs() -> None:
    a = await run_benchmark(
        BenchConfig(
            warmup=0,
            iterations=2,
            suites=["code"],
            backend=BackendConfig(
                type="mock",
                mock_output_tokens=4,
                model="a",
                mock_ttft_ns=0,
                mock_inter_token_ns=0,
            ),
            quality=QualityConfig(enabled=False),
            memory=MemoryConfig(enabled=False),
            output={"label": "machine-a"},
        )
    )
    b = await run_benchmark(
        BenchConfig(
            warmup=0,
            iterations=2,
            suites=["code"],
            backend=BackendConfig(
                type="mock",
                mock_output_tokens=8,
                model="b",
                mock_ttft_ns=0,
                mock_inter_token_ns=0,
            ),
            quality=QualityConfig(enabled=False),
            memory=MemoryConfig(enabled=False),
            output={"label": "machine-b"},
        )
    )
    html = render_html([a, b])
    assert "Comparison" in html
    assert "machine-a" in html and "machine-b" in html
    assert "Raw measured trials — machine-a" in html
    assert "Raw measured trials — machine-b" in html
    assert 'aria-label="' in html
    assert "prefers-color-scheme" in html
    assert "<caption>" in html
    md = render_markdown([a, b])
    assert "machine-a" in md and "machine-b" in md


@pytest.mark.asyncio
async def test_concurrency_chart_has_one_series_per_suite() -> None:
    run = await run_benchmark(
        BenchConfig(
            warmup=0,
            iterations=2,
            concurrency=[1, 2],
            suites=["short_chat", "code"],
            backend=BackendConfig(
                type="mock",
                mock_output_tokens=4,
                mock_ttft_ns=0,
                mock_inter_token_ns=0,
            ),
            quality=QualityConfig(enabled=False),
            memory=MemoryConfig(enabled=False),
        )
    )
    html = render_html([run])
    assert "per suite" in html
    assert ">short_chat</text>" in html
    assert ">code</text>" in html


def test_context_length_max_none_renders_unknown() -> None:
    label = _context_label(ModelInfo(name="m", context_length=8192, context_length_source="server"))
    assert "8192" in label
    assert "model max unknown" in label
    assert "None" not in label
    known = _context_label(
        ModelInfo(
            name="m",
            context_length=32768,
            context_length_max=262144,
            context_length_source="ollama_ps",
        )
    )
    assert "model max 262144" in known
    assert "unknown" not in known
