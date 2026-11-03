from __future__ import annotations

from llmbench.schema import SCHEMA_VERSION, RunResult, SamplingParams, Trial


def _minimal_run() -> dict:
    return {
        "run_id": "abc",
        "created_at": "2026-01-01T00:00:00+00:00",
        "config": {
            "seed": 42,
            "warmup": 1,
            "iterations": 2,
            "concurrency": [1],
            "timeout_s": 30,
            "suites": ["short_chat"],
        },
        "sampling": SamplingParams().model_dump(),
        "model": {"name": "mock-7b"},
        "backend": {"type": "mock"},
        "hardware": {
            "os": "Linux",
            "os_release": "1",
            "arch": "x86_64",
        },
        "harness": {"version": "0.1.0"},
    }


def test_schema_version_stable() -> None:
    assert SCHEMA_VERSION == "1.0.0"
    run = RunResult.model_validate(_minimal_run())
    assert run.schema_version == "1.0.0"
    dumped = run.model_dump()
    assert dumped["schema_version"] == "1.0.0"
    assert dumped["synthetic"] is False


def test_trial_roundtrip() -> None:
    trial = Trial(
        trial_id="t1",
        suite="short_chat",
        prompt_id="p1",
        concurrency=1,
        ok=True,
        prompt_chars=12,
        start_ns=0,
        end_ns=10,
    )
    again = Trial.model_validate(trial.model_dump())
    assert again.trial_id == "t1"
    assert again.warmup is False
