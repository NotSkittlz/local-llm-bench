from __future__ import annotations

from pathlib import Path

import pytest

from llmbench.config import apply_cli_overrides, config_from_mapping, load_config
from llmbench.exceptions import ConfigError


def test_defaults_are_local_mock() -> None:
    cfg = load_config(None)
    assert cfg.backend.type == "mock"
    assert cfg.sampling.temperature == 0.0
    assert cfg.seed == 42
    assert cfg.model.name == "mock-7b"


def test_rejects_cloud_url() -> None:
    with pytest.raises(ConfigError, match="non-local"):
        config_from_mapping(
            {
                "backend": {
                    "type": "openai_compat",
                    "base_url": "https://api.openai.com/v1",
                    "model": "gpt",
                }
            }
        )


def test_accepts_loopback() -> None:
    cfg = config_from_mapping(
        {
            "backend": {
                "type": "ollama",
                "base_url": "http://127.0.0.1:11434",
                "model": "llama",
            }
        }
    )
    assert cfg.backend.base_url == "http://127.0.0.1:11434"


def test_unknown_suite() -> None:
    with pytest.raises(ConfigError, match="unknown suite"):
        config_from_mapping({"suites": ["not_a_suite"]})


def test_yaml_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "seed: 7\niterations: 3\nbackend:\n  type: mock\n  model: mock-x\n",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.seed == 7
    assert cfg.iterations == 3
    assert cfg.sampling.seed == 7


def test_cli_overrides() -> None:
    cfg = load_config(None)
    cfg = apply_cli_overrides(
        cfg,
        {"iterations": 4, "backend.model": "other", "sampling.temperature": 0.8},
    )
    assert cfg.iterations == 4
    assert cfg.backend.model == "other"
    assert cfg.model.name == "other"
    assert cfg.sampling.temperature == 0.8
