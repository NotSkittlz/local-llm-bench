from __future__ import annotations

from llmbench.backends.sse import feed_sse, flush_sse


def test_feed_sse_splits_events() -> None:
    buf = bytearray()
    events = feed_sse(buf, b'data: {"a":1}\n\ndata: {"b":2}\n\n')
    assert events == [b'{"a":1}', b'{"b":2}']
    assert not buf


def test_feed_sse_holds_partial() -> None:
    buf = bytearray()
    assert feed_sse(buf, b"data: {\"a\":") == []
    events = feed_sse(buf, b"1}\n\n")
    assert events == [b'{"a":1}']


def test_flush_trailing_event() -> None:
    buf = bytearray()
    feed_sse(buf, b'data: {"z":1}\n')
    assert flush_sse(buf) == [b'{"z":1}']
    assert not buf


def test_crlf_events() -> None:
    buf = bytearray()
    events = feed_sse(buf, b'data: {"a":1}\r\n\r\n')
    assert events == [b'{"a":1}']
