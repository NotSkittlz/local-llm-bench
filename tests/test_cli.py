from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from llmbench.cli import app

runner = CliRunner()


def test_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.stdout
    assert "compare" in result.stdout
    assert "report" in result.stdout


def test_run_mock_writes_artifacts(tmp_path: Path) -> None:
    json_path = tmp_path / "run.json"
    html_path = tmp_path / "run.html"
    result = runner.invoke(
        app,
        [
            "run",
            "--backend",
            "mock",
            "--iterations",
            "2",
            "--warmup",
            "0",
            "--suites",
            "short_chat",
            "--json",
            str(json_path),
            "--html",
            str(html_path),
            "--csv",
            str(tmp_path / "run.csv"),
            "--label",
            "cli-mock",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json_path.is_file()
    text = json_path.read_text(encoding="utf-8")
    assert '"synthetic": true' in text
    assert '"data_lineage": "synthetic_mock"' in text
    html = html_path.read_text(encoding="utf-8")
    assert "Synthetic data" in html
    assert "cli-mock" in html
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    decode = payload["summaries"][0]["decode_tps"]["p50"]
    ttft = payload["summaries"][0]["ttft_ms"]["p50"]
    assert decode is not None and decode < 10_000
    assert ttft is not None and ttft > 5.0


def test_refuses_public_172_and_attacker_hostname() -> None:
    for url in ("http://172.217.14.206", "http://10.attacker.com"):
        result = runner.invoke(
            app,
            ["run", "--backend", "openai_compat", "--base-url", url, "--model", "x"],
        )
        assert result.exit_code == 1
        blob = (result.output + (result.stderr or "")).lower()
        assert "non-local" in blob or "cannot resolve" in blob


def test_refuses_cloud_url() -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "--backend",
            "openai_compat",
            "--base-url",
            "https://api.openai.com/v1",
            "--model",
            "gpt",
        ],
    )
    assert result.exit_code == 1
    assert "non-local" in result.output.lower() or "non-local" in (result.stderr or "").lower()


def test_suites_and_version() -> None:
    s = runner.invoke(app, ["suites"])
    assert s.exit_code == 0
    assert "short_chat" in s.stdout
    v = runner.invoke(app, ["--version"])
    assert v.exit_code == 0
    assert "0.1.0" in v.stdout


def test_report_and_compare(tmp_path: Path) -> None:
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    for path, label in ((a, "A"), (b, "B")):
        r = runner.invoke(
            app,
            [
                "run",
                "--backend",
                "mock",
                "--iterations",
                "2",
                "--warmup",
                "0",
                "--json",
                str(path),
                "--html",
                str(path.with_suffix(".html")),
                "--csv",
                str(path.with_suffix(".csv")),
                "--label",
                label,
                "--no-quality",
            ],
        )
        assert r.exit_code == 0, r.output
    html_out = tmp_path / "cmp.html"
    cmp = runner.invoke(app, ["compare", str(a), str(b), "--html", str(html_out)])
    assert cmp.exit_code == 0, cmp.output
    assert html_out.is_file()
    assert "A" in html_out.read_text(encoding="utf-8")
    rpt = runner.invoke(
        app,
        ["report", str(a), "--html", str(tmp_path / "r.html"), "--md", str(tmp_path / "r.md")],
    )
    assert rpt.exit_code == 0, rpt.output
    assert "short_chat" in (tmp_path / "r.md").read_text(encoding="utf-8")
