"""Human-readable HTTP / SSE error text from local inference servers."""

from __future__ import annotations

import json
from typing import Any


def error_message_from_payload(value: Any) -> str | None:
    """Pull a human server message out of an OpenAI/llama.cpp error object."""

    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text[:300] if text else None
    if isinstance(value, dict):
        for key in ("message", "msg"):
            nested = error_message_from_payload(value.get(key))
            if nested:
                return nested
        inner = value.get("error")
        if inner is not None and inner is not value:
            nested = error_message_from_payload(inner)
            if nested:
                return nested
        return str(value)[:300]
    return str(value)[:300]


def format_http_error(status: int, body: bytes) -> str:
    text = body.decode("utf-8", errors="replace")[:1000]
    message = None
    stripped = text.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            message = error_message_from_payload(json.loads(stripped))
        except (json.JSONDecodeError, ValueError):
            message = None
    if message:
        return f"HTTP {status}: {message}"
    snippet = text.replace("\n", " ").strip()[:300]
    return f"HTTP {status}: {snippet}" if snippet else f"HTTP {status}"
