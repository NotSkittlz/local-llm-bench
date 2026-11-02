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
