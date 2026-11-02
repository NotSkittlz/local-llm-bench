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
| `sampling` | exact parameters sent on speed trials |
| `model` | name, quantization, context_length |
| `backend` | type, base_url, version |
| `hardware` | CPU, RAM, GPUs, OS |
| `harness.overhead` | zero-delay mock TTFT floor |
| `quality` | greedy probe |
| `memory` | peak RSS / VRAM |
| `summaries` | per suite × concurrency percentiles |
| `trials` | every warmup and measured completion |

`token_source` on a trial is one of `server_usage`, `server_timings`,
`count_chunks`, `approximate`. Do not mix sources in a comparison.

Adding fields is backward compatible. Renaming or changing units requires a
major version bump.
