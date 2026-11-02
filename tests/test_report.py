from __future__ import annotations

import pytest

from llmbench.config import BackendConfig, BenchConfig, MemoryConfig, QualityConfig
from llmbench.report import render_html, render_markdown
from llmbench.runner import run_benchmark
from llmbench.schema import SamplingParams


@pytest.mark.asyncio
async def test_html_is_self_contained_and_flags_synthetic() -> None:
    cfg = BenchConfig(
        warmup=0,
        iterations=3,
        concurrency=[1],
        suites=["short_chat"],
        backend=BackendConfig(type="mock", mock_output_tokens=6, model="mock-7b"),
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
    assert "Prefill" in html
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
            backend=BackendConfig(type="mock", mock_output_tokens=4, model="a"),
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
            backend=BackendConfig(type="mock", mock_output_tokens=8, model="b"),
            quality=QualityConfig(enabled=False),
            memory=MemoryConfig(enabled=False),
            output={"label": "machine-b"},
        )
    )
    html = render_html([a, b])
    assert "Comparison" in html
    assert "machine-a" in html and "machine-b" in html
    md = render_markdown([a, b])
    assert "machine-a" in md and "machine-b" in md
