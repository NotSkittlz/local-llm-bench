from __future__ import annotations

import sys
import types

from llmbench.hardware import (
    _cpu_name,
    _probe_nvml,
    current_process_rss_mib,
    probe_gpus,
    probe_hardware,
)


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


def test_windows_cpu_reads_processor_name_string(monkeypatch) -> None:
    winreg = types.ModuleType("winreg")
    winreg.HKEY_LOCAL_MACHINE = 1

    def open_key(hive: object, path: str) -> object:
        assert "CentralProcessor" in path
        return object()

    def query_value_ex(key: object, name: str) -> tuple[str, int]:
        assert name == "ProcessorNameString"
        return ("AMD Ryzen 9 7950X3D 16-Core Processor", 1)

    def close_key(key: object) -> None:
        return None

    winreg.OpenKey = open_key  # type: ignore[attr-defined]
    winreg.QueryValueEx = query_value_ex  # type: ignore[attr-defined]
    winreg.CloseKey = close_key  # type: ignore[attr-defined]
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "winreg", winreg)
    name, notes = _cpu_name()
    assert name == "AMD Ryzen 9 7950X3D 16-Core Processor"
    assert notes == []


def test_pynvml_missing_mentions_optional_extra_once() -> None:
    notes: list[str] = []
    _probe_nvml(notes)
    extra = [n for n in notes if "pynvml" in n or "[nvidia]" in n]
    assert len(extra) <= 1
    if extra:
        assert "optional extra" in extra[0]
        assert extra[0].count("pynvml") <= 1
