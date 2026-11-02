from __future__ import annotations

import socket
from pathlib import Path

import pytest

from llmbench.config import (
    _assert_local_url,
    apply_cli_overrides,
    config_from_mapping,
    load_config,
)
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


def _fake_addr(ip: str) -> list:
    family = socket.AF_INET6 if ":" in ip else socket.AF_INET
    return [(family, socket.SOCK_STREAM, 6, "", (ip, 0))]


LOCAL_URL_CASES = [
    ("http://127.0.0.1:8080", True),
    ("http://127.0.0.1", True),
    ("http://[::1]:8080", True),
    ("http://0.0.0.0:8080", True),
    ("http://10.0.0.1:11434", True),
    ("http://192.168.1.1", True),
    ("http://172.16.0.1", True),
    ("http://172.31.255.255", True),
    ("http://169.254.1.1", True),
    ("http://[fd12:3456:789a::1]", True),
    ("http://[fe80::1]", True),
    ("http://[::ffff:10.0.0.1]", True),
    ("http://172.15.0.1", False),
    ("http://172.32.0.1", False),
    ("http://172.217.14.206", False),
    ("http://8.8.8.8", False),
    ("https://api.openai.com/v1", False),
    ("http://[::ffff:172.217.14.206]", False),
    ("ftp://127.0.0.1", False),
]


HOSTNAME_CASES = [
    ("http://localhost:8080", ["127.0.0.1"], True),
    ("http://10.attacker.com", ["8.8.8.8"], False),
    ("http://172.evil.com", ["172.217.14.206"], False),
    ("http://home.local", ["192.168.0.5"], True),
    ("http://mixed.example", ["10.0.0.1", "8.8.8.8"], False),
    ("http://loop.example", ["::1"], True),
]


@pytest.mark.parametrize("url,ok", LOCAL_URL_CASES)
def test_local_url_ip_literals(url: str, ok: bool) -> None:
    if ok:
        _assert_local_url(url)
    else:
        with pytest.raises(ConfigError):
            _assert_local_url(url)


@pytest.mark.parametrize("url,ips,ok", HOSTNAME_CASES)
def test_local_url_resolved_hostnames(
    url: str, ips: list[str], ok: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_getaddrinfo(host: str, port: object, *args: object, **kwargs: object):
        infos: list = []
        for ip in ips:
            infos.extend(_fake_addr(ip))
        return infos

    monkeypatch.setattr("socket.getaddrinfo", fake_getaddrinfo)
    if ok:
        _assert_local_url(url)
    else:
        with pytest.raises(ConfigError, match="non-local"):
            _assert_local_url(url)


def test_prefix_match_does_not_allow_public_172() -> None:
    with pytest.raises(ConfigError, match="non-local"):
        config_from_mapping(
            {
                "backend": {
                    "type": "openai_compat",
                    "base_url": "http://172.217.14.206:80",
                    "model": "x",
                }
            }
        )


def test_allow_remote_opt_in() -> None:
    cfg = config_from_mapping(
        {
            "backend": {
                "type": "openai_compat",
                "base_url": "https://api.openai.com/v1",
                "model": "gpt",
                "allow_remote": True,
            }
        }
    )
    assert cfg.backend.allow_remote is True


def test_unresolved_hostname_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(host: str, port: object, *args: object, **kwargs: object):
        raise socket.gaierror("Name or service not known")

    monkeypatch.setattr("socket.getaddrinfo", boom)
    with pytest.raises(ConfigError, match="cannot resolve"):
        _assert_local_url("http://no-such-host.invalid")


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
