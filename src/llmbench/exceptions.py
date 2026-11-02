"""User-facing error types.

CLI code catches ``LLMBenchError`` and prints ``str(exc)`` without a traceback
by default. Internal bugs still surface as unexpected exceptions.
"""

from __future__ import annotations


class LLMBenchError(Exception):
    """Base error for expected failures (config, backend, I/O)."""


class ConfigError(LLMBenchError):
    """Invalid YAML, flags, or combination of options."""


class BackendError(LLMBenchError):
    """The inference server rejected a request or spoke an unexpected protocol."""


class BackendConnectionError(BackendError):
    """TCP/HTTP connection to a local endpoint failed."""


class BackendTimeoutError(BackendError):
    """Connect or read timeout."""


class StreamInterruptedError(BackendError):
    """The token stream ended before a terminal event."""


class ReportError(LLMBenchError):
    """Result files could not be loaded or rendered."""
