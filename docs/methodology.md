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
3. `first_token_ns` when that chunk contains the first generated token of
   **any** kind (Ollama `message.thinking`, llama.cpp `delta.reasoning_content`,
   or content).
4. `first_content_ns` when that chunk contains the first *content* token
   (time to answer on a reasoning model).
5. `last_token_ns` on the last generated token of any kind.
6. `end_ns` when the stream terminates.

If several tokens share a TCP segment they share a timestamp. That is honest:
the client cannot have seen them earlier. Decode tok/s may look bursty on
highly buffered servers.

## Metrics

### Time to first token (TTFT)

Elapsed time from `start_ns` to `first_token_ns`, in milliseconds. That first
token is the first *generated* token, including reasoning/thinking tokens.

`ttft_content_ms` is time from `start_ns` to `first_content_ns` — time to the
first answer token after thinking. On a non-reasoning model the two are the
same.

TTFT includes:

- kernel / loopback / HTTP overhead
- server queueing (especially under concurrency > parallel slots)
- prompt prefill
- the first decode step (which may be a thinking token)
- harness overhead (see below)

TTFT is **not** prefill time. Ollama's `prompt_eval_duration` and llama.cpp
`timings.prompt_ms` are stored separately as `server_timings` when the server
sends them.

### Decode tokens / second

Prefer the **server-reported** rate when the server sends completion count and
duration:

- Ollama: `eval_count / eval_duration`
- llama.cpp: `timings.predicted_n / timings.predicted_ms`

Client fallback, over the full generated span (reasoning + content):

```
(output_tokens - 1) / (last_token_ns - first_token_ns)
```

The first token of any kind is attributed to TTFT, not the decode loop. A
one-token completion has no decode rate (`null`, not `0`).

`reasoning_tokens` and `content_tokens` are counted separately. Decode uses
the full `output_tokens` (server `completion_tokens` / `eval_count` when
present), not content-only.

If client and server decode differ by more than 25% relative to the server,
the trial is flagged `decode_client_server_disagree` and the run records a
warning. Summaries use the preferred (server) rate.

### Prefill tokens / second

Prefill tok/s is **uncached** prompt tokens / **uncached** prompt milliseconds.

- llama.cpp: `prompt_n` / `prompt_ms` are uncached; `cache_n` is stored as
  `cached_prompt_tokens` and is not in the rate.
- Ollama: `prompt_eval_count` is all tokens; `prompt_eval_duration` is
  uncached-only. We need `prompt_eval_cached_count` to form the uncached
  remainder. Without it, prefill is `null`.
- vLLM typically has no prompt timings; prefill stays `null` unless a
  per-request metrics object is present. When `usage.prompt_tokens_details.cached_tokens`
  is present, uncached prompt tokens are `prompt_tokens - cached_tokens`.
  A cached/uncached count of `0` is kept (not treated as missing).

If uncached tokens or uncached ms are missing or zero, prefill is `null`.
We do **not** fall through to `prompt_tokens / TTFT` while labelling it
server prefill. That proxy is stored as `prefill_tps_proxy` and summarised
separately. Percentiles never mix the two.

Prefill on prompts shorter than **256 tokens** is omitted
(`prefill_source=not_meaningful_short_prompt`). A 20-token chat prompt's
"prefill tok/s" is dominated by fixed HTTP/kernel overhead, not kernel
compute; quoting ~150 tok/s next to a 3k-token ~2200 tok/s figure is
misleading.

`long_context` trials get a unique prefix nonce so later measured trials
are not prefix-cache hits on the same body. `cache_hit` is true only when
cached tokens exceed the shared chat-template prefix (more than 64 tokens)
or the whole prompt body is cached. A llama.cpp `cache_n` of ~42 is the
template prefix and is recorded as `cached_prompt_fraction`, not a hit.

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

The probe always sends ``thinking=off`` and scores **content** only. If a
reasoning model still burns the 16-token default on thinking, the probe
retries once at 128 tokens.

`--quality-strict` / `quality.strict: true` aborts the run if the probe fails.
The default Paris substring is a smoke test, not a quantization audit.

## Reasoning / thinking models

Ollama streams a `message.thinking` field; llama.cpp streams
`delta.reasoning_content`. Those tokens are generated tokens: they count in
`output_tokens` / decode tok/s and they stamp `first_token_ns`.

A stream that ends with `done: true` / `[DONE]` / `finish_reason` after
spending the whole `max_tokens` budget on thinking (`done_reason=length` or
`finish_reason=length`) is **ok** for speed metrics, flagged
`no_content_reasoning_only`. A TCP/HTTP cut-off with no terminal event is
still `error_kind=truncated`.

`--thinking {default,off,low,medium,high,xhigh}` is recorded on the run
snapshot. `off` sends Ollama `think=false` and llama.cpp / vLLM
`chat_template_kwargs.enable_thinking=false` (llama.cpp also
`reasoning_budget=0`). `default` leaves the server's default.

If `/api/show` capabilities include `thinking` (or llama.cpp
`chat_template_caps` reports reasoning support) and the smallest
`max_tokens` on the run is under 64, the harness warns: thinking will often
eat a 16- or 32-token budget before any content.

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

Warmup is the prefix of one shuffled sequence (`warmup + iterations`);
measured trials continue from that index rather than restarting at 0.
If the sequence wraps past the suite size, wrapping occurrences get a
unique prefix nonce so measured trials never replay warmup prompt bytes
(default `short_chat` is 12 prompts; default `iterations=16` wraps).
`long_context` is always cache-busted; the nonce includes run id, suite,
concurrency, and trial index so a `[1, 4]` sweep is not a prefix-cache hit
on the previous level. Warmup rows are stored and discarded from summaries.

## Sampling and reproducibility

Defaults are greedy (`temperature=0`, `top_p=1`, `seed=42`) so token counts
are stable enough to compare machines. That is a documented default, not a
silent rewrite: if you set `temperature: 0.8` in YAML, that is what we send.

Recorded on every run: requested sampling, `sampling.applied` (what went
on the wire, including vLLM `repetition_penalty` and Ollama `num_ctx`),
`sampling.dropped`, per-trial `max_tokens` and the min–max range,
quantization / context_length with a `*_source` of `user_declared` or
`applied_num_ctx`, backend type/version, model id.

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
overhead probe does not include it. Backend URLs are resolved; every
address must be loopback, RFC1918 (`10/8`, `172.16/12`, `192.168/16`),
or IPv6 ULA. Link-local (`169.254/16`, `fe80::/10`) and cloud metadata
(`169.254.169.254`, `fd00:ec2::254`, `metadata.google.internal`) are
refused. `--allow-remote` is the explicit opt-in for other non-local
hosts; metadata endpoints stay blocked.

**Truncated streams.** An HTTP 200 body that ends without `[DONE]`,
`finish_reason`, or Ollama `done: true` is `ok=false` with
`error_kind=truncated` and is excluded from percentiles.

**Context vs concurrency.** Before a suite wave, the harness reads llama.cpp
`/props` (`n_ctx` **per slot**, `total_slots`) or Ollama `num_ctx` (per
request). A wave is skipped only when **one** prompt + `max_tokens` cannot
fit in that per-slot window. Concurrency above `total_slots` /
`OLLAMA_NUM_PARALLEL` is noted as queueing, not skipped. HTTP/SSE error
bodies (e.g. "Context size has been exceeded.") are copied onto the suite
summary, not only stored as `error_kind=protocol`.

**Power / clocks.** We do not pin GPU clocks or disable boost. Two runs on
the same box can differ. Say so in the README you paste.

**Windows.** `llmbench hardware` reads `ProcessorNameString` from the
registry (then CIM/wmic) so the CPU is "AMD Ryzen 9 7950X3D ..." rather
than the `platform.processor()` family string. CLI list output uses ASCII
`*` so a cp1252 console does not mangle a bullet. Missing `pynvml` is one
note pointing at the optional `[nvidia]` extra.

## What this tool does not measure

- Training, fine-tuning, or embeddings
- Paid cloud APIs (non-local URLs are rejected unless `--allow-remote`)
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
