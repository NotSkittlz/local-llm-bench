# Methodology

local-llm-bench measures **client-observed** inference behaviour of a local
server you already run. It is not a substitute for a paper's experimental
section, and it will not make a slow server look fast by hiding work in the
client.

If a number can be gamed by the harness, we either do not report it or we
label the proxy explicitly.

## Clocks

All intervals use `time.perf_counter_ns`.

- Monotonic: NTP steps and `clock_settime` do not move it.
- High resolution: typically tens of nanoseconds on Linux.

Wall-clock `time.time_ns` is stored only as `created_at` / run identity.
If wall time and the monotonic clock disagree by more than 250 ms over a
run, the result document records a warning. Durations still come from
`perf_counter`.

Adapters timestamp **before** parsing each inbound byte chunk:

1. `start_ns` immediately before the HTTP write (or mock call).
2. `first_byte_ns` on the first `aiter_bytes` chunk.
3. `first_token_ns` when that chunk contains the first non-empty content delta.
4. `last_token_ns` on the last content delta.
5. `end_ns` when the stream terminates.

If several tokens share a TCP segment they share a timestamp. That is honest:
the client cannot have seen them earlier. Decode tok/s may look bursty on
highly buffered servers.

## Metrics

### Time to first token (TTFT)

Elapsed time from `start_ns` to `first_token_ns`, in milliseconds.

TTFT includes:

- kernel / loopback / HTTP overhead
- server queueing (especially under concurrency > parallel slots)
- prompt prefill
- the first decode step
- harness overhead (see below)

TTFT is **not** prefill time. Ollama's `prompt_eval_duration` and llama.cpp
`timings.prompt_ms` are stored separately as `server_timings` when the server
sends them.

### Decode tokens / second

```
(output_tokens - 1) / (last_token_ns - first_token_ns)
```

The first token is attributed to TTFT, not the decode loop. A one-token
completion has no decode rate (`null`, not `0`).

### Prefill tokens / second

1. Prefer `server_timings.prompt_tokens / server_timings.prompt_ms`.
2. Otherwise the **proxy** `prompt_tokens / TTFT`.

The proxy over-states prefill work because TTFT is not only prefill. The
report and JSON field notes make that visible. Do not compare proxy prefill
across backends that do not expose prompt timings.

### End-to-end latency

`end_ns - start_ns`. Summaries report p50 and p95 over **successful
measured** trials. Warmup is stored but excluded from percentiles.

Percentiles use linear interpolation over the sorted sample
(`k = p/100 * (n-1)`). Empty sets serialize as `n=0` with nulls, never as
zeros.

### Throughput under concurrency

For `concurrency = C`, the runner keeps up to `C` requests in flight until
`iterations` complete (open loop, `asyncio.Semaphore`).

Two numbers:

- **Per-request** TTFT / e2e / decode tok/s (same formulas). Under load these
  include queueing.
- **Wall throughput**: sum of output tokens of successful trials / wall time
  of the measured phase, plus requests/second.

If the server serializes (common for a default llama.cpp server), wall
throughput will not scale with `C`. That is a property of the server, and
the report keeps both views so you can see it.

### Memory

A niced daemon thread samples at 250 ms by default (configurable). It never
runs on the token path.

| Source | What you get | Cadence |
| --- | --- | --- |
| NVML (`nvidia-ml-py`) | used + total VRAM, persistent session | every sample |
| `nvidia-smi` | used + total VRAM | **≥ 1 s** (fork is not free) |
| Apple `system_profiler` | name; unified memory as an upper bound | probe only |
| `rocm-smi` | best-effort VRAM | probe / slow |
| `psutil` | client RSS; optional backend PID RSS | every sample |

GPU memory is **device-level**, not "this model only", if other processes
share the GPU. Apple numbers are not comparable to CUDA VRAM.

## Harness overhead

Every run, including real backends, first measures the in-process mock with
**zero simulated compute**. The median TTFT of that probe is
`harness.overhead.zero_delay_ttft_ms`.

That floor is Python + the event loop + `perf_counter_ns`. It does **not**
include HTTP or TCP; OpenAI-compatible adapters add loopback cost on top.

A healthy Linux box is typically well under a millisecond. If the floor is
several milliseconds, the client is overloaded and the run is suspect.

`llmbench overhead` reprints the probe without running suites.

## Quality probe

Speed numbers can hide a quality regression (wrong quantization, greedy vs
sampled, truncated context, a model that only emits one token).

When enabled, the harness issues a **separate** greedy request
(`temperature=0`, `top_p=1`, `top_k=1`) with a short prompt. It records:

- SHA-256 of the output
- optional substring match (default: France / Paris)
- optional expected digest for bitwise stability

The probe **does not** rewrite sampling parameters of the timed suites.
Whatever you set for the speed run is what is sent and what is recorded.

`--quality-strict` / `quality.strict: true` aborts the run if the probe fails.

The mock backend answers the default France prompt with `Paris` so demos
exercise a passing probe. That is not a claim about any real model.

## Workloads

| Suite | Intent | Default `max_tokens` |
| --- | --- | --- |
| `short_chat` | 12 fixed physics-flavoured chat prompts | config `sampling.max_tokens` |
| `code` | 6 code-writing / review prompts | per-prompt 96–256 |
| `long_context` | seedable generated passage to `long_context_chars` | 32 |

Prompt order is shuffled with `random.Random(seed)` plus a **stable** SHA-256
of the suite name. We do not use Python's salted `hash()`.

Warmup uses the same pool, sequential, discarded from summaries. Use it.
Cold KV-cache and first-weight-load otherwise land in TTFT.

## Sampling and reproducibility

Defaults are greedy (`temperature=0`, `top_p=1`, `seed=42`) so token counts
are stable enough to compare machines. That is a documented default, not a
silent rewrite: if you set `temperature: 0.8` in YAML, that is what we send.

Recorded on every run: temperature, top_p, top_k, min_p, repeat_penalty,
seed, max_tokens, stop, quantization, context length, backend type/version,
model id.

Two runs are comparable only if those fields match. `llmbench compare`
aligns by suite name and concurrency; it will not pretend mismatched
sampling is the same experiment.

## Pitfalls (read these)

**Warmup.** The first completion often pays model load and CUDA context.
Leave `warmup ≥ 1`. Two is the default.

**Thermal throttling.** A 30-second suite will not show a laptop that
throttles after two minutes. If you care about sustained decode, raise
`iterations` and watch TTFT drift across trial index (raw JSON).

**Batching.** A server that batches concurrent requests can raise wall
throughput while worsening per-request TTFT. Report both. Do not quote only
the number that flatters the stack.

**Token counts.** We prefer server `usage` / Ollama `eval_count` /
llama.cpp `timings`. Fallback is counting SSE content deltas (a delta is
not always one token) then a chars/4 estimate, labelled in `token_source`.
Never mix `count_chunks` and `server_usage` across a comparison.

**Queueing vs compute.** Concurrent TTFT includes time spent waiting for a
slot. That is still the latency a user sees. If you want unloaded TTFT, read
the `concurrency=1` summary.

**Client perturbation.** Logging, per-token Python, and `nvidia-smi` in the
request loop would move the numbers. The hot path timestamps and parses;
memory sampling is a niced thread; `nvidia-smi` is rate-limited to 1 Hz.

**Network.** "Local" still crosses the kernel. Loopback is not zero. The
overhead probe does not include it.

**Power / clocks.** We do not pin GPU clocks or disable boost. Two runs on
the same box can differ. Say so in the README you paste.

## What this tool does not measure

- Training, fine-tuning, or embeddings
- Paid cloud APIs (non-local URLs are rejected)
- Quality beyond the small greedy probe
- Per-layer GPU kernels
- Energy (joules) in v0.1

## Suggested reporting template

When you paste results somewhere public:

1. Model, quantization, context length, sampling (all of it).
2. Backend and version, server parallel slots / batch flags.
3. Hardware (CPU, GPU, RAM, OS) from `llmbench hardware`.
4. Suite, iterations, warmup, concurrency.
5. TTFT p50/p95, decode tok/s p50, e2e p95, wall throughput at each `C`.
6. Harness overhead floor.
7. Quality probe pass/fail and digest.
8. The word **synthetic** if you used `--backend mock`.
