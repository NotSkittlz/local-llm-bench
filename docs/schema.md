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
| `model` | name, quantization, context_length, plus `*_source` (`user_declared` or `applied_num_ctx`) |
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
`proxy_ttft`. Summaries never mix them. `server_timings` includes
`cached_prompt_tokens`, `uncached_prompt_tokens`, `uncached_prompt_ms`,
and `cache_hit`.

A truncated HTTP 200 stream (no `[DONE]` / `finish_reason` / `done`) is
`ok=false`, `error_kind=truncated`. `reasoning_tokens` are counted
separately from decode output.

Adding fields is backward compatible. Renaming or changing units requires a
major version bump.
