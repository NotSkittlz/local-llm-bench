from __future__ import annotations

from llmbench.config import BenchConfig
from llmbench.workloads import iter_prompts, load_suite, split_suite


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


def test_warmup_is_not_replayed_as_measured() -> None:
    """B1: default seed=42 warmup=2 iterations=8 must not reuse the warmup prefix."""

    cfg = BenchConfig(seed=42, warmup=2, iterations=8)
    warm, measured = split_suite(cfg, "short_chat", 2, 8)
    assert [p.prompt_id for p in warm] != [p.prompt_id for p in measured[:2]]
    warm_ids = [p.prompt_id for p in warm]
    measured_ids = [p.prompt_id for p in measured]
    # First measured trial is index 2 of the same shuffle, not index 0 again.
    full = iter_prompts(cfg, "short_chat", 10)
    assert [p.prompt_id for p in full[:2]] == warm_ids
    assert [p.prompt_id for p in full[2:]] == measured_ids
    assert set(warm_ids).isdisjoint(measured_ids)


def test_long_context_trials_are_cache_busted() -> None:
    cfg = BenchConfig(seed=42, warmup=2, iterations=4, long_context_chars=4000)
    warm, measured = split_suite(cfg, "long_context", 2, 4)
    texts = [p.text for p in warm + measured]
    assert len(set(texts)) == 6
    ids = [p.prompt_id for p in warm + measured]
    assert len(set(ids)) == 6
    for i, prompt in enumerate(warm + measured):
        assert f"trial={i}" in prompt.text
        assert prompt.text.startswith("[cache-bust")
        # Nonce is a prefix so the long body is not a prefix-cache hit.
        assert prompt.text.index("You are given") > 10
