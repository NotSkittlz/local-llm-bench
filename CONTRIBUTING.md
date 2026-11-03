# Contributing

Thanks for looking. local-llm-bench is a small tool; keep it that way.

## Setup

Python 3.11+. From the repo root:

```bash
python -m pip install -e ".[dev]"
# or
uv pip install -e ".[dev]"
```

## Checks

```bash
ruff check src tests
ruff format src tests
pytest
```

CI runs the same three commands on GitHub-hosted runners (Python 3.11, 3.12, 3.13).

## Measurement rules

These are not style nits:

1. Intervals use `time.perf_counter_ns`. Do not subtract `time.time()`.
2. Timestamp inbound bytes **before** parsing. Do not log on the hot path.
3. Do not sample GPU/RSS inside `complete()`. That is `MemorySampler`'s job.
4. Do not silently change sampling, quantization, or max tokens on a speed
   run. Record what you sent. Quality checks are a separate request.
5. Failed trials are errors with `ok: false`. Never coerce missing data to 0
   tok/s.
6. Local endpoints only. If you add a backend, it must refuse non-local URLs.

## Layout

| Path | Role |
| --- | --- |
| `src/llmbench/backends/` | Adapters (mock, OpenAI-compatible, Ollama) |
| `src/llmbench/runner.py` | Warmup, concurrency, summaries |
| `src/llmbench/report.py` | HTML / markdown |
| `docs/methodology.md` | Metric definitions and pitfalls |
| `tests/test_adversarial.py` | Fault injection |

## Pull requests

- Small, reviewable diffs.
- Tests for the behaviour you changed, including a failure path.
- Do not commit GPU result files that look like product claims. Sample
  fixtures must set `synthetic: true` / `data_lineage: synthetic_mock`.
- Conventional commit messages (`feat:`, `fix:`, `test:`, `docs:`).

## License

MIT. By contributing you agree your changes ship under the same license.
