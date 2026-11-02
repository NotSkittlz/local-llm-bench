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
