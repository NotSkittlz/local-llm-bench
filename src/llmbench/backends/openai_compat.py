"""OpenAI-compatible chat completions client (llama.cpp, vLLM, generic).

Local endpoints only. Streaming is required for TTFT. Token timestamps use
``time.perf_counter_ns`` on each ``aiter_bytes`` chunk *before* SSE parsing.

We send the sampling parameters from config unchanged. We do not lower
temperature, change max_tokens, or rewrite stop sequences for a speed run.
"""

from __future__ import annotations

import json
import time
from typing import Any, Literal

import httpx

from llmbench.backends.base import CompletionRequest, StreamResult
from llmbench.backends.sse import feed_sse, flush_sse
from llmbench.config import BackendConfig
from llmbench.exceptions import BackendConnectionError, BackendTimeoutError
from llmbench.sampling import map_sampling
from llmbench.schema import BackendInfo, ServerTimings

Flavor = Literal["openai_compat", "llamacpp", "vllm"]

_DONE = b"[DONE]"


class OpenAICompatBackend:
    config_type = "openai_compat"

    def __init__(
        self,
        config: BackendConfig,
        *,
        flavor: Flavor = "openai_compat",
        timeout_s: float = 120.0,
        connect_s: float = 5.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.config = config
        self.flavor = flavor
        self._client = httpx.AsyncClient(
            base_url=config.base_url or "http://127.0.0.1:8080",
            timeout=httpx.Timeout(timeout_s, connect=connect_s),
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
            headers=_headers(config),
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def info(self) -> BackendInfo:
        version = None
        listed = None
        extra: dict[str, Any] = {"flavor": self.flavor}
        try:
            models = await self._client.get("/v1/models")
            if models.status_code == 200:
                body = models.json()
                data = body.get("data") or []
                if data:
                    listed = data[0].get("id")
                    extra["models"] = [d.get("id") for d in data[:16]]
        except httpx.HTTPError:
            extra["models_error"] = "unreachable"
        if self.flavor == "llamacpp":
            try:
                props = await self._client.get("/props")
                if props.status_code == 200:
                    extra["props"] = _small_json(props.json())
                    version = str(props.json().get("version") or version)
            except httpx.HTTPError:
                pass
        if self.flavor == "vllm":
            try:
                version_r = await self._client.get("/version")
                if version_r.status_code == 200:
                    version = str(version_r.json().get("version") or version_r.text)[:80]
            except httpx.HTTPError:
                pass
        return BackendInfo(
            type=self.flavor,
            base_url=str(self._client.base_url),
            version=version,
            model_listed=listed or self.config.model,
            extra=extra,
        )

    async def complete(self, request: CompletionRequest) -> StreamResult:
        payload = _chat_payload(self.config.model, request, flavor=self.flavor)
        start_ns = time.perf_counter_ns()
        first_byte_ns: int | None = None
        first_token_ns: int | None = None
        last_token_ns: int | None = None
        prev_token_ns: int | None = None
        n_tokens = 0
        n_reasoning = 0
        n_chars = 0
        text_parts: list[str] = []
        inter: list[int] = []
        buffer = bytearray()
        usage: dict[str, Any] | None = None
        timings: dict[str, Any] | None = None
        error: str | None = None
        error_kind: str | None = None
        saw_done = False
        saw_finish = False

        def consume(events: list[bytes], now: int) -> None:
            nonlocal first_token_ns, last_token_ns, prev_token_ns
            nonlocal n_tokens, n_reasoning, n_chars, usage, timings, error, error_kind
            nonlocal saw_done, saw_finish
            for event in events:
                if event.strip() == _DONE:
                    saw_done = True
                    continue
                delta, u, t, err, finish, reasoning = _parse_openai_event(event)
                if finish:
                    saw_finish = True
                if err:
                    error = err
                    error_kind = "protocol"
                    continue
                if u:
                    usage = u
                if t:
                    timings = t
                if reasoning:
                    n_reasoning += 1
                if delta:
                    n_tokens += 1
                    n_chars += len(delta)
                    if first_token_ns is None:
                        first_token_ns = now
                    last_token_ns = now
                    if request.record_inter_token and prev_token_ns is not None:
                        inter.append(now - prev_token_ns)
                    prev_token_ns = now
                    if request.collect_text:
                        text_parts.append(delta)

        try:
            async with self._client.stream("POST", "/v1/chat/completions", json=payload) as resp:
                if resp.status_code >= 400:
                    body = await resp.aread()
                    error = f"HTTP {resp.status_code}: {body[:300]!r}"
                    error_kind = "http"
                else:
                    async for chunk in resp.aiter_bytes():
                        now = time.perf_counter_ns()
                        if first_byte_ns is None:
                            first_byte_ns = now
                        consume(feed_sse(buffer, chunk), now)
                    consume(flush_sse(buffer), time.perf_counter_ns())
        except httpx.ConnectError as exc:
            raise BackendConnectionError(
                f"could not connect to {self._client.base_url} ({exc}). "
                "Is the local server running?"
            ) from exc
        except httpx.TimeoutException as exc:
            raise BackendTimeoutError(f"timeout talking to {self._client.base_url}: {exc}") from exc
        except httpx.HTTPError as exc:
            error = str(exc)
            error_kind = "http"

        end_ns = time.perf_counter_ns()
        if error is None and not saw_done and not saw_finish:
            error = "stream ended without a terminal event ([DONE] or finish_reason)"
            error_kind = "truncated"
        if first_token_ns is None and error is None:
            error = "stream ended without a content token"
            error_kind = "interrupted"

        prompt_tokens, completion_tokens, token_source = _token_counts(
            usage, timings, n_tokens, request.prompt
        )
        server = _server_timings(timings, usage)
        return StreamResult(
            start_ns=start_ns,
            end_ns=end_ns,
            first_byte_ns=first_byte_ns,
            first_token_ns=first_token_ns,
            last_token_ns=last_token_ns,
            output_tokens=completion_tokens if completion_tokens is not None else n_tokens,
            prompt_tokens=prompt_tokens,
            output_chars=n_chars,
            text="".join(text_parts) if request.collect_text else None,
            error=error,
            error_kind=error_kind,
            server_timings=server,
            token_source=token_source,
            reasoning_tokens=n_reasoning,
            inter_token_ns=inter if request.record_inter_token else [],
        )


def _headers(config: BackendConfig) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    return headers


def _chat_payload(
    model: str, request: CompletionRequest, *, flavor: Flavor = "openai_compat"
) -> dict[str, Any]:
    wire = map_sampling(flavor, request.sampling, context_length=request.context_length)
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": request.prompt}],
        "stream": True,
        **wire.openai_body,
    }
    return payload


def _parse_openai_event(
    raw: bytes,
) -> tuple[
    str | None, dict[str, Any] | None, dict[str, Any] | None, str | None, str | None, str | None
]:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, None, None, f"invalid SSE JSON: {exc}", None, None
    if not isinstance(obj, dict):
        return None, None, None, "SSE JSON was not an object", None, None
    if obj.get("error"):
        return None, None, None, str(obj["error"])[:300], None, None
    usage = obj.get("usage") if isinstance(obj.get("usage"), dict) else None
    timings = obj.get("timings") if isinstance(obj.get("timings"), dict) else None
    delta_text = None
    finish_reason = None
    reasoning = None
    choices = obj.get("choices") or []
    if choices:
        choice = choices[0]
        fr = choice.get("finish_reason")
        if isinstance(fr, str) and fr:
            finish_reason = fr
        delta = choice.get("delta") or choice.get("message") or {}
        delta_text = _content_text(delta.get("content"))
        reasoning = _content_text(
            delta.get("reasoning_content") or delta.get("reasoning") or delta.get("thinking")
        )
    return delta_text, usage, timings, None, finish_reason, reasoning


def _content_text(content: Any) -> str | None:
    if isinstance(content, str) and content:
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str) and item:
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str) and text:
                    parts.append(text)
        return "".join(parts) or None
    return None


def _token_counts(
    usage: dict[str, Any] | None,
    timings: dict[str, Any] | None,
    chunk_tokens: int,
    prompt: str,
) -> tuple[int | None, int | None, str]:
    if usage:
        pt = _as_int(usage.get("prompt_tokens"))
        ct = _as_int(usage.get("completion_tokens"))
        if pt is not None or ct is not None:
            return pt, ct if ct is not None else chunk_tokens, "server_usage"
    if timings:
        pt = _as_int(timings.get("prompt_n"))
        ct = _as_int(timings.get("predicted_n"))
        if pt is not None or ct is not None:
            return pt, ct if ct is not None else chunk_tokens, "server_timings"
    if chunk_tokens:
        return None, chunk_tokens, "count_chunks"
    return max(1, len(prompt) // 4), None, "approximate"


def _server_timings(
    timings: dict[str, Any] | None,
    usage: dict[str, Any] | None,
) -> ServerTimings | None:
    if not timings and not usage:
        return None
    prompt_ms = None
    completion_ms = None
    prompt_tokens = None
    completion_tokens = None
    cached = None
    uncached = None
    uncached_ms = None
    if timings:
        # llama.cpp: prompt_n / prompt_ms are uncached; cache_n is the hit.
        uncached = _as_int(timings.get("prompt_n"))
        cached = _as_int(timings.get("cache_n"))
        prompt_ms = _as_float(timings.get("prompt_ms"))
        uncached_ms = prompt_ms
        completion_tokens = _as_int(timings.get("predicted_n"))
        completion_ms = _as_float(timings.get("predicted_ms"))
        if uncached is not None and cached is not None:
            prompt_tokens = uncached + cached
        else:
            prompt_tokens = uncached
        # Optional vLLM-style per-request metrics if a server attaches them.
        # Explicit None checks: 0 is a valid uncached count (full cache hit).
        if uncached is None:
            uncached = _as_int(timings.get("uncached_tokens"))
        if uncached is None:
            uncached = _as_int(timings.get("num_prompt_tokens"))
        if uncached_ms is None:
            uncached_ms = _as_float(timings.get("prompt_eval_ms"))
        if uncached_ms is None:
            uncached_ms = _as_float(timings.get("time_to_first_token_ms"))
    if usage:
        usage_prompt = _as_int(usage.get("prompt_tokens"))
        if prompt_tokens is None:
            prompt_tokens = usage_prompt
        if completion_tokens is None:
            completion_tokens = _as_int(usage.get("completion_tokens"))
        if cached is None:
            cached = _as_int(usage.get("prompt_tokens_cached"))
        if cached is None:
            details = usage.get("prompt_tokens_details")
            if isinstance(details, dict):
                cached = _as_int(details.get("cached_tokens"))
    cache_hit = None
    if cached is not None:
        cache_hit = cached > 0
    if uncached is None and prompt_tokens is not None and cached is not None:
        uncached = max(0, prompt_tokens - cached)
    if not any(
        v is not None
        for v in (prompt_tokens, prompt_ms, cached, uncached, completion_tokens, completion_ms)
    ):
        return None
    return ServerTimings(
        prompt_tokens=prompt_tokens,
        prompt_ms=prompt_ms,
        cached_prompt_tokens=cached,
        uncached_prompt_tokens=uncached,
        uncached_prompt_ms=uncached_ms,
        completion_tokens=completion_tokens,
        completion_ms=completion_ms,
        cache_hit=cache_hit,
    )


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _as_float(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def _small_json(value: Any, limit: int = 40) -> Any:
    if isinstance(value, dict):
        return {str(k): value[k] for k, _ in zip(list(value)[:limit], range(limit), strict=False)}
    return value
