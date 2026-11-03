# Result schema 1.0.0

Canonical document: JSON written by `llmbench run`. CSV is a denormalized
view of `trials`. HTML is derived and not a source of truth.

Units:

- `*_ms` — milliseconds (float)
- `*_ns` — nanoseconds (int), raw timestamps only
- `*_tps` — tokens / second
- `*_mib` — mebibytes (1024² bytes)

Top-level fields (abridged):

| Field | Meaning |
| --- | --- |
| `schema_version` | `1.0.0` |
| `data_lineage` | `measured` or `synthetic_mock` |
| `synthetic` | `true` iff the mock backend produced the timings |
| `config` | seed, warmup, iterations, concurrency, suites |
| `sampling` | requested params plus `applied` / `dropped` and `max_tokens_min`/`max` |
| `model` | name, quantization, `context_length` (effective), `context_length_max` (architecture), plus `context_length_source` (`user_declared`, `applied_num_ctx`, `server`, `ollama_ps`, `ollama_num_ctx`, `ollama_env`, `ollama_default`) |
| `backend` | type, base_url, version |
| `hardware` | CPU, RAM, GPUs, OS |
| `harness.overhead` | zero-delay mock TTFT floor |
| `quality` | greedy probe |
| `memory` | peak RSS / VRAM |
| `summaries` | per suite × concurrency percentiles |
| `trials` | every warmup and measured completion |

`token_source` on a trial is one of `server_usage`, `server_timings`,
`count_chunks`, `approximate`. Do not mix sources in a comparison.

Prefill: `prefill_tps` is uncached-only server timing. `prefill_tps_proxy`
is `prompt_tokens / TTFT`. `prefill_source` is `server_uncached` or
`proxy_ttft`, or `not_meaningful_short_prompt` (prompt under 256 tokens).
Summaries never mix them. Prefill percentiles omit short prompts. `server_timings` includes
`cached_prompt_tokens`, `uncached_prompt_tokens`, `uncached_prompt_ms`,
`cache_hit`, and `cached_prompt_fraction`. `cache_hit` ignores a short
chat-template prefix (≤64 tokens). vLLM `usage.prompt_tokens_details.cached_tokens` is
mapped onto `cached_prompt_tokens` when present.

A truncated HTTP 200 stream (no `[DONE]` / `finish_reason` / `done`) is
`ok=false`, `error_kind=truncated`. `reasoning_tokens` and `content_tokens`
are counted separately. `first_token_ns` is the first generated token of any
kind; `first_content_ns` / `ttft_content_ms` is time to the first answer
token. Decode tok/s prefers server timings (`decode_source=server`) with a
client fallback over the full generated span; a >25% disagreement is
flagged `decode_client_server_disagree` when `output_tokens >= 16`. A completed stream with only
reasoning (`done_reason`/`finish_reason=length`) is `ok` with flag
`no_content_reasoning_only`. `config.thinking` records `--thinking`.

Adding fields is backward compatible. Renaming or changing units requires a
major version bump.
