"""Minimal SSE splitter used by OpenAI-compatible streaming.

The splitter is allocation-light: it appends to a ``bytearray`` and yields
complete ``data:`` payloads. Timestamping happens in the adapter *before*
this function sees the chunk.
"""

from __future__ import annotations


def feed_sse(buffer: bytearray, chunk: bytes) -> list[bytes]:
    """Append ``chunk`` and return complete SSE data payloads (raw JSON bytes)."""

    buffer.extend(chunk)
    return _drain_sse(buffer, final=False)


def flush_sse(buffer: bytearray) -> list[bytes]:
    """Emit any trailing event after the socket closes (no terminating blank line)."""

    return _drain_sse(buffer, final=True)


def _drain_sse(buffer: bytearray, *, final: bool) -> list[bytes]:
    events: list[bytes] = []
    while True:
        sep = buffer.find(b"\n\n")
        cr = buffer.find(b"\r\n\r\n")
        if sep < 0 and cr < 0:
            break
        if sep < 0 or (cr >= 0 and cr < sep):
            raw = bytes(buffer[:cr])
            del buffer[: cr + 4]
        else:
            raw = bytes(buffer[:sep])
            del buffer[: sep + 2]
        payload = _event_data(raw)
        if payload is not None:
            events.append(payload)
    if final and buffer:
        payload = _event_data(bytes(buffer))
        buffer.clear()
        if payload is not None:
            events.append(payload)
    return events


def _event_data(raw: bytes) -> bytes | None:
    data_parts: list[bytes] = []
    for line in raw.splitlines():
        if line.startswith(b"data:"):
            data_parts.append(line[5:].lstrip())
    if not data_parts:
        return None
    return b"\n".join(data_parts)
