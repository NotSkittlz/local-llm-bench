from __future__ import annotations

from llmbench.config import BenchConfig
from llmbench.workloads import iter_prompts, load_suite


def test_suites_are_nonempty() -> None:
    cfg = BenchConfig()
    assert len(load_suite("short_chat", cfg)) >= 8
    assert len(load_suite("code", cfg)) >= 4
    long_p = load_suite("long_context", cfg)[0]
    assert len(long_p.text) >= cfg.long_context_chars


def test_seed_reproducible_order() -> None:
    a = [p.prompt_id for p in iter_prompts(BenchConfig(seed=3), "short_chat", 20)]
    b = [p.prompt_id for p in iter_prompts(BenchConfig(seed=3), "short_chat", 20)]
    c = [p.prompt_id for p in iter_prompts(BenchConfig(seed=4), "short_chat", 20)]
    assert a == b
    assert a != c


def test_long_context_is_seedable() -> None:
    cfg = BenchConfig(seed=1, long_context_chars=4000)
    x = load_suite("long_context", cfg)[0].text
    y = load_suite("long_context", BenchConfig(seed=1, long_context_chars=4000))[0].text
    z = load_suite("long_context", BenchConfig(seed=2, long_context_chars=4000))[0].text
    assert x == y
    assert x != z
