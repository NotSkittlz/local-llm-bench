"""YAML + CLI configuration.

Precedence: CLI flags > YAML file > built-in defaults.

The harness never infers sampling parameters from the backend. Whatever is
in this config is what we send, and what we record.
"""

from __future__ import annotations

import ipaddress
import socket
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from llmbench.exceptions import ConfigError
from llmbench.schema import SamplingParams

BackendType = Literal["mock", "openai_compat", "llamacpp", "vllm", "ollama"]

BUILTIN_SUITES = ("short_chat", "long_context", "code")


class BackendConfig(BaseModel):
    type: BackendType = "mock"
    base_url: str | None = None
    model: str = "mock-7b"
    api_key: str | None = None
    # Mock-only simulated compute (nanoseconds). Zero means "as fast as Python"
    # and is reserved for the overhead probe; the default demo is human-scale.
    mock_ttft_ns: int = 40_000_000
    mock_inter_token_ns: int = 8_000_000
    mock_output_tokens: int = 32
    mock_prompt_eval_ns_per_token: int = 0
    # Optional PID whose RSS we sample as "backend process".
    pid: int | None = None
    # Off-box URLs are refused unless the user opts in explicitly.
    allow_remote: bool = False

    @field_validator("base_url")
    @classmethod
    def _strip_slash(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.rstrip("/")

    @model_validator(mode="after")
    def _local_only(self) -> BackendConfig:
        if self.type != "mock" and not self.base_url:
            default = {
                "ollama": "http://127.0.0.1:11434",
                "llamacpp": "http://127.0.0.1:8080",
                "vllm": "http://127.0.0.1:8000",
                "openai_compat": "http://127.0.0.1:8080",
            }[self.type]
            self.base_url = default
        if self.base_url:
            _assert_local_url(self.base_url, allow_remote=self.allow_remote)
        return self


class ModelConfig(BaseModel):
    name: str | None = None
    quantization: str | None = None
    context_length: int | None = None
    family: str | None = None


class QualityConfig(BaseModel):
    enabled: bool = True
    prompt: str = "The capital of France is"
    expected_sha256: str | None = None
    expected_contains: str | None = "Paris"
    max_tokens: int = 16
    strict: bool = False


class MemoryConfig(BaseModel):
    enabled: bool = True
    sample_interval_ms: float = 250.0
    gpu_index: int = 0


class OutputConfig(BaseModel):
    json_path: str | None = None
    csv_path: str | None = None
    html_path: str | None = None
    markdown_path: str | None = None
    label: str | None = None


class BenchConfig(BaseModel):
    seed: int = 42
    warmup: int = 2
    iterations: int = 8
    concurrency: list[int] = Field(default_factory=lambda: [1])
    timeout_s: float = 120.0
    connect_timeout_s: float = 5.0
    suites: list[str] = Field(default_factory=lambda: ["short_chat"])
    record_inter_token: bool = False
    collect_output_text: bool = False
    long_context_chars: int = 16_000
    backend: BackendConfig = Field(default_factory=BackendConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    sampling: SamplingParams = Field(default_factory=SamplingParams)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @field_validator("concurrency")
    @classmethod
    def _conc(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("concurrency must be a non-empty list")
        for item in value:
            if item < 1:
                raise ValueError("concurrency values must be >= 1")
        return value

    @field_validator("suites")
    @classmethod
    def _suites(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("at least one suite is required")
        unknown = [s for s in value if s not in BUILTIN_SUITES]
        if unknown:
            raise ValueError(f"unknown suite(s) {unknown}; built-ins are {list(BUILTIN_SUITES)}")
        return value

    @field_validator("warmup", "iterations")
    @classmethod
    def _nonneg(cls, value: int) -> int:
        if value < 0:
            raise ValueError("warmup/iterations must be >= 0")
        return value

    @model_validator(mode="after")
    def _model_name(self) -> BenchConfig:
        if self.model.name is None:
            self.model.name = self.backend.model
        self.sampling.seed = self.seed
        return self


_ALLOWED_NETWORKS = (
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("fe80::/10"),
)


def _assert_local_url(url: str, *, allow_remote: bool = False) -> None:
    """Refuse non-local URLs unless ``allow_remote`` is set."""

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ConfigError(f"backend URL must be http(s), got {url!r}")
    host = parsed.hostname
    if not host:
        raise ConfigError(f"backend URL has no host: {url!r}")
    if allow_remote:
        return
    addresses = _resolve_host_ips(host)
    public = [ip for ip in addresses if not _is_local_ip(ip)]
    if public:
        shown = ", ".join(str(ip) for ip in public)
        raise ConfigError(
            f"refusing non-local backend URL {url!r} (resolved {shown}). "
            "Pass --allow-remote if you really intend to leave the machine."
        )


def _resolve_host_ips(host: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        return [_unwrap_mapped(literal)]
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise ConfigError(f"cannot resolve backend host {host!r}: {exc}") from exc
    addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    seen: set[str] = set()
    for info in infos:
        sockaddr = info[4]
        raw = sockaddr[0]
        try:
            ip = _unwrap_mapped(ipaddress.ip_address(raw))
        except ValueError:
            continue
        key = str(ip)
        if key in seen:
            continue
        seen.add(key)
        addresses.append(ip)
    if not addresses:
        raise ConfigError(f"cannot resolve backend host {host!r} to an IP address")
    return addresses


def _unwrap_mapped(
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return ip.ipv4_mapped
    return ip


def _is_local_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if ip.is_loopback or ip.is_unspecified:
        return True
    return any(ip in network for network in _ALLOWED_NETWORKS)


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read config {path}: {exc}") from exc
    try:
        data = yaml.safe_load(raw) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"config root in {path} must be a mapping")
    return data


def config_from_mapping(data: dict[str, Any]) -> BenchConfig:
    try:
        return BenchConfig.model_validate(data)
    except ConfigError:
        raise
    except Exception as exc:  # pydantic ValidationError
        raise ConfigError(str(exc)) from exc


def load_config(path: Path | None) -> BenchConfig:
    if path is None:
        return BenchConfig()
    return config_from_mapping(load_yaml(path))


def apply_cli_overrides(cfg: BenchConfig, overrides: dict[str, Any]) -> BenchConfig:
    """Shallow-merge dotted overrides from CLI flags."""

    payload = cfg.model_dump()
    for key, value in overrides.items():
        if value is None:
            continue
        _set_dotted(payload, key, value)
    if "backend.model" in overrides and "model.name" not in overrides:
        payload.setdefault("model", {})["name"] = None
    return config_from_mapping(payload)


def _set_dotted(payload: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur: dict[str, Any] = payload
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value
