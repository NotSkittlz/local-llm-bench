"""Command-line interface for local-llm-bench."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from llmbench import __version__
from llmbench.config import (
    BUILTIN_SUITES,
    THINKING_MODES,
    apply_cli_overrides,
    load_config,
)
from llmbench.exceptions import LLMBenchError
from llmbench.export import read_json, write_csv, write_json
from llmbench.hardware import probe_hardware
from llmbench.overhead import measure_harness_overhead
from llmbench.report import render_markdown, write_html, write_markdown
from llmbench.runner import run_benchmark
from llmbench.schema import RunResult

app = typer.Typer(
    name="llmbench",
    no_args_is_help=True,
    rich_markup_mode="rich",
    help="Benchmark local LLM inference on your own hardware.",
)
console = Console()
err_console = Console(stderr=True)


def _print_dim(msg: str) -> None:
    console.print(escape(msg), style="dim")


def _print_error(msg: str) -> None:
    err_console.print(f"[red]Error:[/] {escape(str(msg))}")


def _configure_stdio() -> None:
    """Prefer UTF-8; fall back to ASCII-safe glyphs on legacy Windows consoles."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if not callable(reconfigure):
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError, AttributeError):
            continue


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"local-llm-bench {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    _configure_stdio()
    return None


@app.command()
def run(
    config: Path | None = typer.Option(None, "--config", "-c", help="YAML config file."),
    backend: str | None = typer.Option(
        None,
        "--backend",
        help="mock | ollama | llamacpp | vllm | openai_compat",
    ),
    base_url: str | None = typer.Option(None, "--base-url", help="Local server URL."),
    model: str | None = typer.Option(None, "--model", "-m"),
    quantization: str | None = typer.Option(None, "--quantization"),
    context_length: int | None = typer.Option(None, "--context-length"),
    suites: str | None = typer.Option(None, "--suites", help="Comma-separated suite names."),
    iterations: int | None = typer.Option(None, "--iterations", "-n"),
    warmup: int | None = typer.Option(None, "--warmup"),
    concurrency: str | None = typer.Option(None, "--concurrency", help="e.g. 1 or 1,2,4"),
    seed: int | None = typer.Option(None, "--seed"),
    timeout: float | None = typer.Option(None, "--timeout", help="Per-request timeout in seconds."),
    temperature: float | None = typer.Option(None, "--temperature"),
    json_out: Path | None = typer.Option(None, "--json", help="Write JSON results."),
    csv_out: Path | None = typer.Option(None, "--csv"),
    html_out: Path | None = typer.Option(None, "--html"),
    md_out: Path | None = typer.Option(None, "--md"),
    label: str | None = typer.Option(None, "--label"),
    quality: bool | None = typer.Option(None, "--quality/--no-quality"),
    quality_strict: bool = typer.Option(
        False,
        "--quality-strict",
        help="Abort the run if the greedy quality probe fails.",
    ),
    allow_remote: bool = typer.Option(
        False,
        "--allow-remote",
        help="Permit a non-local backend URL (prompts will leave this machine).",
    ),
    thinking: str | None = typer.Option(
        None,
        "--thinking",
        help="Reasoning control: default | off | low | medium | high | xhigh.",
    ),
) -> None:
    """Run a benchmark and write JSON/CSV/HTML."""

    try:
        cfg = load_config(config)
        overrides: dict = {}
        if backend:
            overrides["backend.type"] = backend
        if base_url:
            overrides["backend.base_url"] = base_url
        if model:
            overrides["backend.model"] = model
        if quantization:
            overrides["model.quantization"] = quantization
        if context_length is not None:
            overrides["model.context_length"] = context_length
        if suites:
            overrides["suites"] = [s.strip() for s in suites.split(",") if s.strip()]
        if iterations is not None:
            overrides["iterations"] = iterations
        if warmup is not None:
            overrides["warmup"] = warmup
        if concurrency:
            overrides["concurrency"] = [int(x) for x in concurrency.split(",") if x.strip()]
        if seed is not None:
            overrides["seed"] = seed
        if timeout is not None:
            overrides["timeout_s"] = timeout
        if temperature is not None:
            overrides["sampling.temperature"] = temperature
        if label:
            overrides["output.label"] = label
        if quality is not None:
            overrides["quality.enabled"] = quality
        if quality_strict:
            overrides["quality.strict"] = True
        if allow_remote:
            overrides["backend.allow_remote"] = True
        if thinking is not None:
            mode = thinking.strip().lower()
            if mode not in THINKING_MODES:
                allowed = ", ".join(THINKING_MODES)
                _print_error(f"--thinking must be one of: {allowed}")
                raise typer.Exit(2)
            overrides["thinking"] = mode
        cfg = apply_cli_overrides(cfg, overrides)

        json_path = json_out or (
            Path(cfg.output.json_path) if cfg.output.json_path else Path("results/run.json")
        )
        csv_path = csv_out or (
            Path(cfg.output.csv_path) if cfg.output.csv_path else json_path.with_suffix(".csv")
        )
        html_path = html_out or (
            Path(cfg.output.html_path) if cfg.output.html_path else json_path.with_suffix(".html")
        )
        md_path = md_out or (Path(cfg.output.markdown_path) if cfg.output.markdown_path else None)

        console.print(
            Panel.fit(
                Text.from_markup(
                    f"[bold]local-llm-bench[/] {escape(__version__)}\n"
                    f"backend [cyan]{escape(cfg.backend.type)}[/]  "
                    f"model [cyan]{escape(cfg.model.name)}[/]  "
                    f"seed [cyan]{cfg.seed}[/]"
                ),
                border_style="dim",
            )
        )

        def progress(msg: str) -> None:
            _print_dim(msg)

        result = asyncio.run(run_benchmark(cfg, progress=progress))
        write_json(result, json_path)
        write_csv(result, csv_path)
        write_html([result], html_path)
        if md_path:
            write_markdown([result], md_path)
        _print_summary(result)
        console.print(
            f"\nWrote [bold]{escape(str(json_path))}[/], "
            f"{escape(str(csv_path))}, {escape(str(html_path))}"
        )
        if result.synthetic:
            console.print(
                "[yellow]Synthetic mock timings — not hardware inference. Do not cite as performance.[/]"
            )
    except LLMBenchError as exc:
        _print_error(exc)
        raise typer.Exit(1) from exc


@app.command()
def report(
    results: list[Path] = typer.Argument(..., help="One or more JSON result files."),
    html_out: Path | None = typer.Option(None, "--html", "-o"),
    md_out: Path | None = typer.Option(None, "--md"),
    stdout_md: bool = typer.Option(False, "--stdout", help="Print markdown to stdout."),
) -> None:
    """Render a self-contained HTML report (and optional markdown tables)."""

    try:
        runs = [_load(p) for p in results]
        html_path = html_out or Path("results/report.html")
        write_html(runs, html_path)
        md = render_markdown(runs)
        if md_out:
            Path(md_out).parent.mkdir(parents=True, exist_ok=True)
            Path(md_out).write_text(md, encoding="utf-8")
        if stdout_md:
            console.print(md, markup=False)
        console.print(f"Wrote [bold]{escape(str(html_path))}[/]")
    except LLMBenchError as exc:
        _print_error(exc)
        raise typer.Exit(1) from exc


@app.command()
def compare(
    results: list[Path] = typer.Argument(..., help="JSON result files to compare."),
    html_out: Path | None = typer.Option(None, "--html", "-o"),
    md_out: Path | None = typer.Option(None, "--md"),
) -> None:
    """Compare runs across hardware or backends; markdown tables for READMEs."""

    if len(results) < 2:
        _print_error("compare needs at least two result files.")
        raise typer.Exit(2)
    try:
        runs = [_load(p) for p in results]
        md = render_markdown(runs)
        console.print(md, markup=False)
        if html_out:
            write_html(runs, html_out)
            console.print(f"Wrote [bold]{escape(str(html_out))}[/]")
        if md_out:
            Path(md_out).write_text(md, encoding="utf-8")
    except LLMBenchError as exc:
        _print_error(exc)
        raise typer.Exit(1) from exc


@app.command()
def overhead() -> None:
    """Measure client harness overhead against the zero-delay mock backend."""

    try:
        result = asyncio.run(measure_harness_overhead())
    except LLMBenchError as exc:
        _print_error(exc)
        raise typer.Exit(1) from exc
    table = Table(title="Harness overhead (in-process mock, zero simulated compute)")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("TTFT floor", f"{result.zero_delay_ttft_ms:.4f} ms")
    table.add_row("E2E floor", f"{result.zero_delay_e2e_ms:.4f} ms")
    table.add_row("perf_counter_ns", f"{result.timestamp_cost_ns:.0f} ns/call")
    table.add_row("Probe tokens", str(result.tokens_in_probe))
    console.print(table)
    for note in result.notes:
        _print_dim(note)


@app.command()
def hardware() -> None:
    """Dump detected CPU, RAM, GPU, and OS metadata."""

    info = probe_hardware()
    table = Table(title="Hardware probe")
    table.add_column("Key")
    table.add_column("Value")
    table.add_row("OS", f"{info.os} {info.os_release}")
    table.add_row("Arch", info.arch)
    table.add_row("CPU", info.cpu_name or "-")
    table.add_row("Cores", f"{info.cpu_cores_physical}/{info.cpu_cores_logical} phys/log")
    table.add_row("RAM", f"{info.ram_total_mib:.0f} MiB" if info.ram_total_mib else "-")
    if info.gpus:
        for gpu in info.gpus:
            table.add_row(
                f"GPU {gpu.index}",
                f"{gpu.name}  {gpu.vram_total_mib or '-'} MiB ({gpu.vram_source})",
            )
    else:
        table.add_row("GPU", "none detected")
    console.print(table)
    for note in info.notes:
        _print_dim(note)


@app.command("suites")
def suites_cmd() -> None:
    """List built-in prompt suites."""

    for name in BUILTIN_SUITES:
        console.print(f"* [bold]{escape(name)}[/]")
    console.print("[dim]Configure via YAML or --suites short_chat,code,long_context[/]")


def _load(path: Path) -> RunResult:
    return read_json(path)


def _print_summary(result: RunResult) -> None:
    table = Table(title="Measured summary (warmup excluded)")
    table.add_column("Suite")
    table.add_column("Conc", justify="right")
    table.add_column("n", justify="right")
    table.add_column("TTFT p50", justify="right")
    table.add_column("Decode p50", justify="right")
    table.add_column("E2E p95", justify="right")
    for s in result.summaries:
        table.add_row(
            s.name,
            str(s.concurrency),
            str(s.n_ok),
            f"{s.ttft_ms.p50:.2f} ms" if s.ttft_ms.p50 is not None else "-",
            f"{s.decode_tps.p50:.1f} tok/s" if s.decode_tps.p50 is not None else "-",
            f"{s.e2e_ms.p95:.1f} ms" if s.e2e_ms.p95 is not None else "-",
        )
    console.print(table)
    if result.quality and result.quality.enabled:
        flag = "[green]pass[/]" if result.quality.ok else "[red]fail[/]"
        console.print(f"Quality probe: {flag}")
    if result.harness.overhead:
        floor = result.harness.overhead.zero_delay_ttft_ms
        if floor < 0.01:
            console.print(f"Harness TTFT floor: {floor * 1000:.2f} µs")
        else:
            console.print(f"Harness TTFT floor: {floor:.3f} ms")
