from __future__ import annotations

import time

from llmbench.memory import MemorySampler
from llmbench.schema import MemorySample


def test_sampler_collects_on_background_thread() -> None:
    n = {"i": 0}

    def snap() -> MemorySample:
        n["i"] += 1
        return MemorySample(t_ms=0.0, client_rss_mib=10.0 + n["i"], source="test")

    sampler = MemorySampler(interval_ms=50, snapshot=snap)
    sampler.start()
    time.sleep(0.18)
    summary = sampler.stop()
    assert summary.n_samples >= 2
    assert summary.client_rss_peak_mib is not None
    assert summary.client_rss_peak_mib >= 12.0
    ts = [s.t_ms for s in sampler.samples]
    assert ts == sorted(ts)
    assert sampler._thread is not None
    assert not sampler._thread.is_alive()


def test_sampler_stop_without_start_is_empty() -> None:
    sampler = MemorySampler(interval_ms=100, snapshot=lambda: MemorySample(t_ms=0, source="x"))
    summary = sampler.stop()
    assert summary.n_samples == 0
