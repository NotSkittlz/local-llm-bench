"""Best-effort hardware and OS metadata.

Probes must never raise into the runner. Missing tools become ``notes``.
GPU VRAM uses NVML when the optional extra is installed, then ``nvidia-smi``,
then vendor-specific CLIs. None of these run on the token hot path.
"""

from __future__ import annotations

import contextlib
import os
import platform
import shutil
import subprocess
import sys
from typing import Any

import psutil

from llmbench.schema import GPUInfo, HardwareInfo

_NVIDIA_SMI_TIMEOUT_S = 2.0


def probe_hardware() -> HardwareInfo:
    notes: list[str] = []
    gpus, gpu_notes = probe_gpus()
    notes.extend(gpu_notes)
    cpu_name, cpu_notes = _cpu_name()
    notes.extend(cpu_notes)
    vm = psutil.virtual_memory()
    return HardwareInfo(
        os=platform.system(),
        os_release=f"{platform.release()} ({platform.version()})",
        arch=platform.machine(),
        hostname=platform.node() or None,
        cpu_name=cpu_name,
        cpu_cores_logical=psutil.cpu_count(logical=True),
        cpu_cores_physical=psutil.cpu_count(logical=False),
        ram_total_mib=vm.total / (1024 * 1024),
        gpus=gpus,
        notes=notes,
    )


def probe_gpus() -> tuple[list[GPUInfo], list[str]]:
    notes: list[str] = []
    gpus = _probe_nvml(notes)
    if gpus:
        return gpus, notes
    gpus = _probe_nvidia_smi(notes)
    if gpus:
        return gpus, notes
    gpus = _probe_apple(notes)
    if gpus:
        return gpus, notes
    gpus = _probe_rocm(notes)
    if gpus:
        return gpus, notes
    notes.append("no GPU probe succeeded (NVML, nvidia-smi, Apple, ROCm)")
    return [], notes


def _cpu_name() -> tuple[str | None, list[str]]:
    notes: list[str] = []
    if sys.platform == "win32":
        name = _windows_cpu_name(notes)
        if name:
            return name, notes
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as fh:
                text = fh.read()
        except OSError as exc:
            notes.append(f"/proc/cpuinfo unreadable: {exc}")
        else:
            for line in text.splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip(), notes
    brand = platform.processor() or None
    if not brand:
        notes.append("CPU model name unavailable")
    return brand, notes


def _windows_cpu_name(notes: list[str]) -> str | None:
    try:
        import winreg  # type: ignore[import-not-found]
    except ImportError:
        winreg = None  # type: ignore[assignment]
    if winreg is not None:
        try:
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            try:
                value, _typ = winreg.QueryValueEx(key, "ProcessorNameString")
            finally:
                winreg.CloseKey(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        except OSError as exc:
            notes.append(f"winreg ProcessorNameString unreadable: {exc}")
    name = _windows_cpu_name_wmic(notes)
    if name:
        return name
    return None


def _windows_cpu_name_wmic(notes: list[str]) -> str | None:
    for argv in (
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_Processor | Select-Object -First 1).Name",
        ],
        ["wmic", "cpu", "get", "Name"],
    ):
        exe = shutil.which(argv[0])
        if not exe:
            continue
        try:
            proc = subprocess.run(
                [exe, *argv[1:]],
                check=False,
                capture_output=True,
                text=True,
                timeout=3.0,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            notes.append(f"{argv[0]} CPU name failed: {exc}")
            continue
        if proc.returncode != 0:
            continue
        lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
        lines = [ln for ln in lines if ln.lower() not in {"name", "name="}]
        if lines:
            return lines[0]
    return None


def _probe_nvml(notes: list[str]) -> list[GPUInfo]:
    try:
        import pynvml  # type: ignore[import-not-found]
    except ImportError:
        notes.append("optional extra [nvidia] (pynvml) not installed")
        return []
    try:
        pynvml.nvmlInit()
    except Exception as exc:
        notes.append(f"NVML init failed: {exc}")
        return []
    try:
        count = int(pynvml.nvmlDeviceGetCount())
        gpus: list[GPUInfo] = []
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            raw_name = pynvml.nvmlDeviceGetName(handle)
            name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total = mem.total / (1024 * 1024)
            except Exception:
                total = None
            gpus.append(
                GPUInfo(
                    index=i,
                    name=name,
                    vendor="nvidia",
                    vram_total_mib=total,
                    vram_source="nvml",
                )
            )
        return gpus
    except Exception as exc:
        notes.append(f"NVML enumeration failed: {exc}")
        return []
    finally:
        with contextlib.suppress(Exception):
            pynvml.nvmlShutdown()


def _probe_nvidia_smi(notes: list[str]) -> list[GPUInfo]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        notes.append("nvidia-smi not on PATH")
        return []
    try:
        proc = subprocess.run(
            [
                exe,
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=_NVIDIA_SMI_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        notes.append(f"nvidia-smi failed: {exc}")
        return []
    if proc.returncode != 0:
        notes.append(f"nvidia-smi exit {proc.returncode}: {proc.stderr.strip()[:200]}")
        return []
    gpus: list[GPUInfo] = []
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[0])
            total = float(parts[2])
        except ValueError:
            continue
        gpus.append(
            GPUInfo(
                index=idx,
                name=parts[1],
                vendor="nvidia",
                vram_total_mib=total,
                vram_source="nvidia-smi",
            )
        )
    return gpus


def _probe_apple(notes: list[str]) -> list[GPUInfo]:
    if sys.platform != "darwin":
        return []
    exe = shutil.which("system_profiler")
    if not exe:
        notes.append("system_profiler not on PATH")
        return []
    try:
        proc = subprocess.run(
            [exe, "SPDisplaysDataType", "-json"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        notes.append(f"system_profiler failed: {exc}")
        return []
    if proc.returncode != 0:
        notes.append("system_profiler returned non-zero")
        return []
    try:
        import json

        payload: Any = json.loads(proc.stdout)
    except json.JSONDecodeError:
        notes.append("system_profiler JSON parse failed")
        return []
    gpus: list[GPUInfo] = []
    displays = payload.get("SPDisplaysDataType") or []
    for i, item in enumerate(displays):
        name = item.get("sppci_model") or item.get("_name") or "Apple GPU"
        # Unified memory: report system RAM as an upper bound, labelled as such.
        vm = psutil.virtual_memory()
        gpus.append(
            GPUInfo(
                index=i,
                name=str(name),
                vendor="apple",
                vram_total_mib=vm.total / (1024 * 1024),
                vram_source="apple_unified_memory_upper_bound",
            )
        )
    if not gpus:
        notes.append("Apple displays list empty")
    return gpus


def _probe_rocm(notes: list[str]) -> list[GPUInfo]:
    exe = shutil.which("rocm-smi")
    if not exe:
        return []
    try:
        proc = subprocess.run(
            [exe, "--showproductname", "--showmeminfo", "vram"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        notes.append(f"rocm-smi failed: {exc}")
        return []
    if proc.returncode != 0:
        notes.append(f"rocm-smi exit {proc.returncode}")
        return []
    name = "AMD GPU"
    for line in proc.stdout.splitlines():
        if "Card series" in line or ("GPU[" in line and ":" in line):
            name = line.split(":")[-1].strip() or name
            break
    total = None
    for line in proc.stdout.splitlines():
        lower = line.lower()
        if "vram" in lower and "total" in lower:
            nums = [tok for tok in line.replace(",", " ").split() if tok.isdigit()]
            if nums:
                # rocm-smi often reports bytes.
                val = float(nums[-1])
                total = val / (1024 * 1024) if val > 10_000 else val
            break
    notes.append("AMD VRAM via rocm-smi is best-effort; treat as approximate")
    return [
        GPUInfo(
            index=0,
            name=name,
            vendor="amd",
            vram_total_mib=total,
            vram_source="rocm-smi",
        )
    ]


def current_process_rss_mib() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def process_rss_mib(pid: int) -> float | None:
    try:
        return psutil.Process(pid).memory_info().rss / (1024 * 1024)
    except (psutil.Error, OSError):
        return None
