"""Ollama HTTP adapter (``/api/chat`` streaming NDJSON).

Ollama reports ``prompt_eval_duration`` / ``eval_duration`` in nanoseconds on
the final object. We record those as ``ServerTimings``. Client TTFT is the
first generated token of any kind (``message.thinking`` or ``message.content``).
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx

from llmbench.backends.base import CompletionRequest, StreamResult, classify_stream_end
from llmbench.config import BackendConfig
from llmbench.exceptions import BackendConnectionError, BackendTimeoutError
from llmbench.http_errors import format_http_error
from llmbench.metrics import classify_cache_hit
from llmbench.sampling import map_sampling
from llmbench.schema import BackendInfo, ServerTimings


class OllamaBackend:
    config_type = "ollama"

    def __init__(
        self,
        config: BackendConfig,
        *,
        timeout_s: float = 120.0,
        connect_s: float = 5.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.config = config
        self._client = httpx.AsyncClient(
            base_url=config.base_url or "http://127.0.0.1:11434",
            timeout=httpx.Timeout(timeout_s, connect=connect_s),
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def info(self) -> BackendInfo:
        version = None
        extra: dict[str, Any] = {}
        try:
            ver = await self._client.get("/api/version")
            if ver.status_code == 200:
                version = str(ver.json().get("version") or ver.text)[:80]
        except httpx.HTTPError:
            extra["version_error"] = "unreachable"
        try:
            show = await self._client.post("/api/show", json={"model": self.config.model})
            if show.status_code == 200:
                body = show.json()
                extra["details"] = {
                    k: body.get(k)
                    for k in ("modelfile", "parameters", "details")
                    if k in body and k != "modelfile"
                }
                if isinstance(body.get("details"), dict):
                    extra["family"] = body["details"].get("family")
                    extra["quantization"] = body["details"].get("quantization_level")
                caps = body.get("capabilities")
                if isinstance(caps, list):
                    extra["capabilities"] = [str(c) for c in caps]
        except httpx.HTTPError:
            pass
        raw_parallel = os.environ.get("OLLAMA_NUM_PARALLEL")
        parsed_parallel = _as_int(raw_parallel) if raw_parallel else None
        if parsed_parallel is not None and parsed_parallel > 0:
            extra["num_parallel"] = parsed_parallel
        return BackendInfo(
            type="ollama",
            base_url=str(self._client.base_url),
            version=version,
            model_listed=self.config.model,
            extra=extra,
        )

    async def complete(self, request: CompletionRequest) -> StreamResult:
        wire = map_sampling(
            "ollama",
            request.sampling,
            context_length=request.context_length,
            thinking=request.thinking,
        )
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "stream": True,
            "options": wire.ollama_options,
            **wire.ollama_extra,
        }
        start_ns = time.perf_counter_ns()
        first_byte_ns: int | None = None
        first_token_ns: int | None = None
        first_content_ns: int | None = None
        last_token_ns: int | None = None
        prev_token_ns: int | None = None
        n_content = 0
        n_reasoning = 0
        n_chars = 0
        text_parts: list[str] = []
        inter: list[int] = []
        buf = bytearray()
        final: dict[str, Any] | None = None
        error: str | None = None
        error_kind: str | None = None
        saw_done = False
        finish_reason: str | None = None

        try:
            async with self._client.stream("POST", "/api/chat", json=payload) as resp:
                if resp.status_code >= 400:
                    body = await resp.aread()
                    error = format_http_error(resp.status_code, body)
                    error_kind = "http"
                else:
                    async for chunk in resp.aiter_bytes():
                        now = time.perf_counter_ns()
                        if first_byte_ns is None:
                            first_byte_ns = now
                        buf.extend(chunk)
                        while True:
                            nl = buf.find(b"\n")
                            if nl < 0:
                                break
                            line = bytes(buf[:nl]).strip()
                            del buf[: nl + 1]
                            if not line:
                                continue
                            try:
                                obj = json.loads(line)
                            except json.JSONDecodeError as exc:
                                error = f"invalid NDJSON: {exc}"
                                error_kind = "protocol"
                                continue
                            if not isinstance(obj, dict):
                                continue
                            if obj.get("error"):
                                error = str(obj["error"])[:300]
                                error_kind = "http"
                            msg = obj.get("message") or {}
                            content = msg.get("content") if isinstance(msg, dict) else None
                            thinking = msg.get("thinking") if isinstance(msg, dict) else None
                            generated = False
                            if isinstance(thinking, str) and thinking:
                                n_reasoning += 1
                                generated = True
                            if isinstance(content, str) and content:
                                n_content += 1
                                n_chars += len(content)
                                if first_content_ns is None:
                                    first_content_ns = now
                                generated = True
                                if request.collect_text:
                                    text_parts.append(content)
                            if generated:
                                if first_token_ns is None:
                                    first_token_ns = now
                                last_token_ns = now
                                if request.record_inter_token and prev_token_ns is not None:
                                    inter.append(now - prev_token_ns)
                                prev_token_ns = now
                            if obj.get("done"):
                                saw_done = True
                                final = obj
                                done_reason = obj.get("done_reason")
                                if isinstance(done_reason, str) and done_reason:
                                    finish_reason = done_reason
                    tail = bytes(buf).strip()
                    if tail:
                        try:
                            obj = json.loads(tail)
                        except json.JSONDecodeError:
                            pass
                        else:
                            if isinstance(obj, dict) and obj.get("done"):
                                saw_done = True
                                final = obj
                                done_reason = obj.get("done_reason")
                                if isinstance(done_reason, str) and done_reason:
                                    finish_reason = done_reason
        except httpx.ConnectError as exc:
            raise BackendConnectionError(
                f"could not connect to Ollama at {self._client.base_url} ({exc}). "
                "Start Ollama or pass --backend mock."
            ) from exc
        except httpx.TimeoutException as exc:
            raise BackendTimeoutError(f"timeout talking to Ollama: {exc}") from exc
        except httpx.HTTPError as exc:
            error = str(exc)
            error_kind = "http"

        end_ns = time.perf_counter_ns()
        error, error_kind, flags = classify_stream_end(
            error=error,
            error_kind=error_kind,
            saw_terminal=saw_done,
            missing_terminal="stream ended without a terminal event (done:true)",
            content_tokens=n_content,
            reasoning_tokens=n_reasoning,
        )

        server = None
        generated = n_content + n_reasoning
        prompt_tokens = None
        completion_tokens = generated or None
        token_source = "count_chunks"
        if final:
            total_prompt = _as_int(final.get("prompt_eval_count"))
            cached = _as_int(final.get("prompt_eval_cached_count"))
            uncached_ms = _ns_to_ms(final.get("prompt_eval_duration"))
            uncached = None
            # prompt_eval_count is all tokens; prompt_eval_duration is uncached-only.
            # Without cached count we cannot form an honest uncached/ms rate.
            if total_prompt is not None and cached is not None:
                uncached = max(0, total_prompt - cached)
            cache_hit, fraction = classify_cache_hit(cached, total_prompt, uncached)
            server = ServerTimings(
                prompt_tokens=total_prompt,
                prompt_ms=uncached_ms,
                cached_prompt_tokens=cached,
                uncached_prompt_tokens=uncached,
                uncached_prompt_ms=uncached_ms,
                completion_tokens=_as_int(final.get("eval_count")),
                completion_ms=_ns_to_ms(final.get("eval_duration")),
                load_ms=_ns_to_ms(final.get("load_duration")),
                total_ms=_ns_to_ms(final.get("total_duration")),
                cache_hit=cache_hit,
                cached_prompt_fraction=fraction,
            )
            if server.prompt_tokens is not None:
                prompt_tokens = server.prompt_tokens
            if server.completion_tokens is not None:
                completion_tokens = server.completion_tokens
                token_source = "server_usage"

        return StreamResult(
            start_ns=start_ns,
            end_ns=end_ns,
            first_byte_ns=first_byte_ns,
            first_token_ns=first_token_ns,
            first_content_ns=first_content_ns,
            last_token_ns=last_token_ns,
            output_tokens=completion_tokens or 0,
            prompt_tokens=prompt_tokens,
            output_chars=n_chars,
            text="".join(text_parts) if request.collect_text else None,
            error=error,
            error_kind=error_kind,
            server_timings=server,
            token_source=token_source,
            reasoning_tokens=n_reasoning,
            content_tokens=n_content,
            finish_reason=finish_reason,
            flags=flags,
            inter_token_ns=inter if request.record_inter_token else [],
        )


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _ns_to_ms(value: Any) -> float | None:
    if isinstance(value, int | float):
        return float(value) / 1e6
    return None
