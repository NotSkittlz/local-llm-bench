> **Synthetic data** from the mock backend. Do not cite as hardware performance.

# local-llm-bench report — mock-7b

- Backend: `mock` mock-0.1
- Sampling: T=0.0 top_p=1.0 max_tokens=32-256 seed=42
- Hardware: Intel(R) Xeon(R) Processor · no GPU

| Suite | Conc | n | TTFT p50 (ms) | TTFT p95 (ms) | Decode p50 (tok/s) | E2E p50 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short_chat | 1 | 8 | 40.16 | 40.21 | 122.1 | 294.0 |
| short_chat | 4 | 8 | 40.30 | 40.32 | 121.6 | 295.2 |
| code | 1 | 8 | 40.15 | 40.33 | 122.3 | 293.7 |
| code | 4 | 8 | 40.22 | 40.26 | 121.5 | 295.4 |
| long_context | 1 | 8 | 40.26 | 40.32 | 122.3 | 293.8 |
| long_context | 4 | 8 | 40.31 | 40.32 | 121.6 | 295.2 |

Harness overhead (zero-delay mock): TTFT floor **0.71 µs**.
