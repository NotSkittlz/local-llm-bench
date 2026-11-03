# Changelog

## 0.1.0

First public-shaped release of local-llm-bench. Measurement-integrity fixes
landed on PR #1 after adversarial review:

- Warmup is a prefix of one prompt sequence; `long_context` is cache-busted.
- Prefill tok/s is uncached tokens / uncached ms, never mixed with the TTFT proxy.
- Sampling fields are mapped per backend; dropped names are recorded.
- Truncated HTTP 200 streams are errors, not ok trials.
- Local URLs use `ipaddress` + DNS; `--allow-remote` is the opt-in.
- Mock defaults are 40 ms TTFT / 8 ms per token.
- Concurrency charts are one series per suite; `max_tokens` is per-trial.
- `os.nice` is Linux-only; `--quality-strict` exists on the CLI.
- HTML theme a11y, captured stream fixtures, Python 3.13 CI.
- Wrapped suite prompts are cache-busted so default n=16 cannot replay warmup IDs.
- Cache-bust nonce includes concurrency and run id (no `[1, 4]` prefix replay).
- Link-local and cloud metadata URLs (`169.254.169.254`, `fd00:ec2::254`,
  `metadata.google.internal`) are refused.
- vLLM `usage.prompt_tokens_details.cached_tokens` drives uncached prefill;
  a cached count of 0 is not treated as missing.
- Reasoning-model decode: timestamp the first token of any kind, keep
  TTFT-to-content separately, and prefer server `eval_count/eval_duration`
  or `predicted_n/predicted_ms`. Warn when client vs server disagree by >25%.
- A finished length-limited response that is only reasoning is `ok` (flag
  `no_content_reasoning_only`), not `interrupted`. `--thinking` maps to
  Ollama `think` and llama.cpp/vLLM `chat_template_kwargs`.
- Quality probe sends `thinking=off`, scores content only, and retries at
  128 tokens if thinking ate the 16-token budget.
- llama.cpp template-prefix `cache_n` (~42) is a cached fraction, not a
  prompt cache hit.
- Preflight copies HTTP/SSE error text (e.g. llama.cpp "Context size has
  been exceeded.") onto the suite summary.
- Windows: CPU name from ProcessorNameString; ASCII-safe `llmbench suites`;
  pynvml-missing note mentions the optional `[nvidia]` extra once.
- Prefill tok/s is omitted for prompts under 256 tokens (fixed overhead
  dominates; not comparable to long-context prefill).
- llama.cpp `/props` `n_ctx` is per slot: skip only when one request cannot
  fit; concurrency above `total_slots` is queueing, not a skip. Same
  per-request semantics for Ollama `num_ctx` / `OLLAMA_NUM_PARALLEL`.
- Rich console output escapes interpolated strings so notes like
  `[nvidia]` are not swallowed as markup.
- Thinking-budget warning ignores the quality probe (always
  `thinking=off`); the speed-run minimum (long_context = 32) is used.
- Report cosmetics: `--md` path on the Wrote line; llama.cpp version from
  `/props` `build_info` or `/version`; `model.context_length` from server
  n_ctx / num_ctx; decode mismatch flag suppressed below 16 output tokens.
- README notes that speculative decoding makes decode tok/s prompt-dependent.
- CLI help tests collapse whitespace so 80-column Rich wrapping cannot hide
  flags on CI.
- Ollama effective `context_length` prefers `/api/ps`, then Modelfile
  `num_ctx`, then `OLLAMA_CONTEXT_LENGTH` on loopback, then the 4096
  default. Architecture max is stored as `context_length_max`.
- Summary/HTML notes that llama.cpp `long_context` at `C>1` can show
  lower per-request decode because slots share the GPU during prefill.
- Ollama does not assume a 4096 context default: re-read `/api/ps` after
  the model loads, and warn instead of skipping when the runtime window
  is still unknown.
- Queueing notes cite `/props total_slots` (no nested parens) and appear
  once per suite summary.
- Suite footer notes are prefixed with `suite [backend]`.
- Notes bimodal llama.cpp prompt-processing times (p90/p50) caused by RAM
  prompt-cache eviction.
- Notes when Ollama `OLLAMA_NUM_PARALLEL` is unknown at `C>1`. Renders
  missing `context_length_max` as `unknown`.
