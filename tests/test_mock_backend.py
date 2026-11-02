from __future__ import annotations

import pytest

from llmbench.backends.base import CompletionRequest
from llmbench.backends.mock import MockBackend, MockFault
from llmbench.config import BackendConfig
from llmbench.schema import SamplingParams


def _req(**kwargs: object) -> CompletionRequest:
    return CompletionRequest(
        prompt="hello",
        sampling=SamplingParams(max_tokens=8, seed=1),
        **kwargs,  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
async def test_zero_delay_mock_is_fast() -> None:
    backend = MockBackend(
        BackendConfig(mock_output_tokens=16, mock_ttft_ns=0, mock_inter_token_ns=0)
    )
    result = await backend.complete(_req())
    assert result.error is None
    assert result.output_tokens == 16
    assert result.first_token_ns is not None
    ttft_ms = (result.first_token_ns - result.start_ns) / 1e6
    assert ttft_ms < 20.0  # harness floor, not simulated compute


@pytest.mark.asyncio
async def test_timed_mock_matches_configured_delay() -> None:
    backend = MockBackend(
        BackendConfig(
            mock_output_tokens=5,
            mock_ttft_ns=30_000_000,
            mock_inter_token_ns=10_000_000,
        )
    )
    result = await backend.complete(_req())
    ttft_ms = (result.first_token_ns - result.start_ns) / 1e6  # type: ignore[operator]
    decode_ms = (result.last_token_ns - result.first_token_ns) / 1e6  # type: ignore[operator]
    assert 25.0 <= ttft_ms <= 80.0
    assert 30.0 <= decode_ms <= 120.0


@pytest.mark.asyncio
async def test_collect_text_is_deterministic() -> None:
    backend = MockBackend(BackendConfig(mock_output_tokens=4))
    a = await backend.complete(_req(collect_text=True))
    b = await backend.complete(_req(collect_text=True))
    assert a.text == b.text
    assert a.text and len(a.text) > 0


@pytest.mark.asyncio
async def test_interrupt_fault_returns_partial() -> None:
    backend = MockBackend(
        BackendConfig(mock_output_tokens=8),
        fault=MockFault(kind="interrupt", after_tokens=2),
    )
    result = await backend.complete(_req())
    assert result.error_kind == "interrupted"
    assert result.output_tokens == 2
    assert result.first_token_ns is not None


@pytest.mark.asyncio
async def test_timeout_fault() -> None:
    backend = MockBackend(fault=MockFault(kind="timeout"))
    result = await backend.complete(_req())
    assert result.error_kind == "timeout"
    assert result.first_token_ns is None
