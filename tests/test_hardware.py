from __future__ import annotations

from llmbench.hardware import current_process_rss_mib, probe_gpus, probe_hardware


def test_probe_hardware_never_raises() -> None:
    info = probe_hardware()
    assert info.os
    assert info.arch
    assert info.ram_total_mib is not None
    assert info.ram_total_mib > 0
    assert info.cpu_cores_logical is not None


def test_probe_gpus_is_list() -> None:
    gpus, notes = probe_gpus()
    assert isinstance(gpus, list)
    assert isinstance(notes, list)


def test_client_rss_positive() -> None:
    rss = current_process_rss_mib()
    assert rss > 1.0
