from __future__ import annotations

import hashlib

import pytest

from llmbench.backends.mock import MockBackend
from llmbench.config import BackendConfig, BenchConfig, QualityConfig
from llmbench.quality import run_quality_probe


@pytest.mark.asyncio
async def test_quality_probe_passes_on_mock_paris() -> None:
    cfg = BenchConfig(backend=BackendConfig(type="mock", mock_ttft_ns=0, mock_inter_token_ns=0))
    backend = MockBackend(cfg.backend)
    probe = await run_quality_probe(backend, cfg)
    assert probe.enabled is True
    assert probe.ok is True
    assert probe.contains_hit is True
    assert probe.output_text is not None
    assert probe.output_text.startswith("Paris")
    assert probe.output_sha256 == hashlib.sha256(probe.output_text.encode()).hexdigest()
    await backend.aclose()


@pytest.mark.asyncio
async def test_quality_probe_hash_mismatch() -> None:
    cfg = BenchConfig(
        backend=BackendConfig(type="mock", mock_ttft_ns=0, mock_inter_token_ns=0),
        quality=QualityConfig(expected_sha256="deadbeef" * 8, expected_contains=None),
    )
    backend = MockBackend(cfg.backend)
    probe = await run_quality_probe(backend, cfg)
    assert probe.hash_match is False
    assert probe.ok is False
    await backend.aclose()


@pytest.mark.asyncio
async def test_disabled_probe() -> None:
    cfg = BenchConfig(quality=QualityConfig(enabled=False))
    backend = MockBackend(cfg.backend)
    probe = await run_quality_probe(backend, cfg)
    assert probe.enabled is False
    assert probe.ok is None
    await backend.aclose()
