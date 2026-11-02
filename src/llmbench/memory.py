"""Out-of-band memory sampler.

Design constraints (measurement integrity):

* Sampling lives on a **daemon thread**, never on the asyncio request path.
* The thread nices itself (and, when possible, moves to a non-zero CPU) so
  it loses the scheduler contest against the event loop and the server.
* Default interval is 250ms. NVML reads are cheap; ``nvidia-smi`` subprocess
  is not, so subprocess fallback uses a 1s floor.
* The sampler never parses tokens, never logs, and never takes the GIL for
  longer than one status snapshot.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Callable

from llmbench.hardware import current_process_rss_mib, process_rss_mib
from llmbench.schema import MemorySample, MemorySummary

SnapshotFn = Callable[[], MemorySample]


class MemorySampler:
    def __init__(
        self,
        *,
        interval_ms: float = 250.0,
        backend_pid: int | None = None,
        gpu_index: int = 0,
        snapshot: SnapshotFn | None = None,
    ) -> None:
        self.interval_ms = interval_ms
        self.backend_pid = backend_pid
        self.gpu_index = gpu_index
        self._custom_snapshot = snapshot
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._samples: list[MemorySample] = []
        self._t0 = 0
        self._gpu_source: str | None = None
        self._gpu_total: float | None = None
        self._nvml: _NvmlSession | None = None
        self._last_smi_ns = 0
        self._cached_smi: tuple[float | None, float | None] = (None, None)

    def start(self) -> None:
        self._t0 = time.perf_counter_ns()
        self._stop.clear()
        if self._custom_snapshot is None:
            self._nvml = _NvmlSession.open(self.gpu_index)
        self._thread = threading.Thread(
            target=self._run,
            name="llmbench-mem",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> MemorySummary:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_ms / 1000.0 * 4))
        if self._nvml is not None:
            self._nvml.close()
            self._nvml = None
        samples = self.samples
        client_peaks = [s.client_rss_mib for s in samples if s.client_rss_mib is not None]
        backend_peaks = [s.backend_rss_mib for s in samples if s.backend_rss_mib is not None]
        gpu_peaks = [s.gpu_vram_used_mib for s in samples if s.gpu_vram_used_mib is not None]
        return MemorySummary(
            sample_interval_ms=self.interval_ms,
            n_samples=len(samples),
            client_rss_peak_mib=max(client_peaks) if client_peaks else None,
            backend_rss_peak_mib=max(backend_peaks) if backend_peaks else None,
            gpu_vram_peak_mib=max(gpu_peaks) if gpu_peaks else None,
            gpu_vram_total_mib=self._gpu_total,
            gpu_source=self._gpu_source,
            sampler="thread",
        )

    @property
    def samples(self) -> list[MemorySample]:
        with self._lock:
            return list(self._samples)

    def _run(self) -> None:
        _deprioritize()
        self._record()
        interval_s = max(0.05, self.interval_ms / 1000.0)
        while not self._stop.wait(interval_s):
            self._record()

    def _record(self) -> None:
        sample = self._custom_snapshot() if self._custom_snapshot else self._default_snapshot()
        sample.t_ms = (time.perf_counter_ns() - self._t0) / 1_000_000.0
        if sample.gpu_vram_total_mib is not None:
            self._gpu_total = sample.gpu_vram_total_mib
        if sample.source:
            self._gpu_source = sample.source
        with self._lock:
            self._samples.append(sample)

    def _default_snapshot(self) -> MemorySample:
        gpu_used, gpu_total, source = self._gpu_memory()
        backend_rss = process_rss_mib(self.backend_pid) if self.backend_pid else None
        return MemorySample(
            t_ms=0.0,
            client_rss_mib=current_process_rss_mib(),
            backend_rss_mib=backend_rss,
            gpu_vram_used_mib=gpu_used,
            gpu_vram_total_mib=gpu_total,
            gpu_index=self.gpu_index if gpu_used is not None else None,
            source=source or "rss",
        )

    def _gpu_memory(self) -> tuple[float | None, float | None, str | None]:
        if self._nvml is not None:
            used, total = self._nvml.read()
            if used is not None:
                return used, total, "nvml"
        now = time.perf_counter_ns()
        # nvidia-smi forks a process; do not do that at 4 Hz.
        if now - self._last_smi_ns >= 1_000_000_000:
            self._cached_smi = _nvidia_smi_memory(self.gpu_index)
            self._last_smi_ns = now
        used, total = self._cached_smi
        if used is not None:
            return used, total, "nvidia-smi"
        return None, None, None


class _NvmlSession:
    """Keep NVML initialized for the life of the sampler."""

    def __init__(self, nvml: object, handle: object) -> None:
        self._nvml = nvml
        self._handle = handle

    @classmethod
    def open(cls, gpu_index: int) -> _NvmlSession | None:
        try:
            import pynvml  # type: ignore[import-not-found]
        except ImportError:
            return None
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        except Exception:
            with contextlib.suppress(Exception):
                pynvml.nvmlShutdown()
            return None
        return cls(pynvml, handle)

    def read(self) -> tuple[float | None, float | None]:
        try:
            mem = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)  # type: ignore[attr-defined]
            return mem.used / (1024 * 1024), mem.total / (1024 * 1024)
        except Exception:
            return None, None

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._nvml.nvmlShutdown()  # type: ignore[attr-defined]


def _deprioritize() -> None:
    with contextlib.suppress(OSError):
        os.nice(10)
    try:
        ncpu = os.cpu_count() or 1
        if ncpu > 1 and hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, {ncpu - 1})
    except (OSError, AttributeError):
        pass


def _nvidia_smi_memory(gpu_index: int) -> tuple[float | None, float | None]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None, None
    try:
        proc = subprocess.run(
            [
                exe,
                "--query-gpu=index,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None, None
    if proc.returncode != 0:
        return None, None
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            if int(parts[0]) != gpu_index:
                continue
            return float(parts[1]), float(parts[2])
        except ValueError:
            continue
    return None, None
