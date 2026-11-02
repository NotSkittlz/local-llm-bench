"""Inference backend adapters."""

from __future__ import annotations

from llmbench.backends.base import Backend, CompletionRequest, StreamResult
from llmbench.backends.mock import MockBackend
from llmbench.backends.ollama import OllamaBackend
from llmbench.backends.openai_compat import OpenAICompatBackend
from llmbench.config import BackendConfig, BenchConfig
from llmbench.exceptions import ConfigError


def build_backend(cfg: BenchConfig) -> Backend:
    kind = cfg.backend.type
    if kind == "mock":
        return MockBackend(cfg.backend)
    if kind in {"openai_compat", "llamacpp", "vllm"}:
        return OpenAICompatBackend(cfg.backend, flavor=kind, timeout_s=cfg.timeout_s, connect_s=cfg.connect_timeout_s)
    if kind == "ollama":
        return OllamaBackend(cfg.backend, timeout_s=cfg.timeout_s, connect_s=cfg.connect_timeout_s)
    raise ConfigError(f"unknown backend type {kind!r}")


__all__ = [
    "Backend",
    "BackendConfig",
    "CompletionRequest",
    "MockBackend",
    "OllamaBackend",
    "OpenAICompatBackend",
    "StreamResult",
    "build_backend",
]
