"""HTML and Markdown reports from one or more ``RunResult`` documents."""

from __future__ import annotations

import html
from importlib import resources
from pathlib import Path

from jinja2 import Template

from llmbench import __version__
from llmbench.charts import grouped_bar_chart
from llmbench.exceptions import ReportError
from llmbench.schema import SCHEMA_VERSION, HarnessOverhead, RunResult, SuiteSummary


def render_html(runs: list[RunResult]) -> str:
    if not runs:
        raise ReportError("no runs to report")
    primary = runs[0]
    synthetic = all(r.synthetic for r in runs)
    tiles = _hero_tiles(primary) if len(runs) == 1 else _compare_tiles(runs)
    warnings: list[str] = []
    for r in runs:
        warnings.extend(r.warnings)
    heading = "Run report" if len(runs) == 1 else f"Comparison · {len(runs)} runs"
    subtitle = _subtitle(runs)
    tmpl = Template(_load_template(), autoescape=True)
    return tmpl.render(
        title=f"local-llm-bench · {heading}",
        heading=heading,
        subtitle=subtitle,
        version=__version__,
        schema_version=SCHEMA_VERSION,
        synthetic=synthetic,
        warnings=warnings,
        tiles=tiles,
        hardware_html=_hardware_section(runs),
        quality_html=_quality_section(runs),
        charts_html=_charts_section(runs),
        tables_html=_tables_section(runs),
    )


def render_markdown(runs: list[RunResult]) -> str:
    if not runs:
        raise ReportError("no runs to report")
    lines: list[str] = []
    if all(r.synthetic for r in runs):
        lines.append(
            "> **Synthetic data** from the mock backend. Do not cite as hardware performance.\n"
        )
    if len(runs) == 1:
        lines.extend(_markdown_single(runs[0]))
    else:
        lines.extend(_markdown_compare(runs))
    return "\n".join(lines).rstrip() + "\n"


def write_html(runs: list[RunResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(runs), encoding="utf-8")


def write_markdown(runs: list[RunResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(runs), encoding="utf-8")


def _load_template() -> str:
    return (
        resources.files("llmbench").joinpath("templates/report.html.j2").read_text(encoding="utf-8")
    )


def _subtitle(runs: list[RunResult]) -> str:
    bits = []
    for r in runs:
        label = r.label or r.model.name
        bits.append(f"{label} · {r.backend.type} · {r.created_at}")
    return "  ·  ".join(bits)


def _hero_tiles(run: RunResult) -> list[dict[str, str]]:
    s = _primary_summary(run)
    overhead = run.harness.overhead
    mem = run.memory
    return [
        {
            "label": "TTFT p50",
            "value": _n(s.ttft_ms.p50 if s else None),
            "unit": "ms",
        },
        {
            "label": "TTFT p95",
            "value": _n(s.ttft_ms.p95 if s else None),
            "unit": "ms",
        },
        {
            "label": "Decode p50",
            "value": _n(s.decode_tps.p50 if s else None),
            "unit": "tok/s",
        },
        {
            "label": "E2E p50",
            "value": _n(s.e2e_ms.p50 if s else None),
            "unit": "ms",
        },
        {
            "label": "VRAM peak",
            "value": _n(mem.gpu_vram_peak_mib if mem else None),
            "unit": "MiB" if mem and mem.gpu_vram_peak_mib is not None else "",
        },
        _harness_tile(overhead),
    ]


def _harness_tile(overhead: HarnessOverhead | None) -> dict[str, str]:
    if overhead is None:
        return {"label": "Harness TTFT floor", "value": "—", "unit": ""}
    ms = overhead.zero_delay_ttft_ms
    if ms < 0.01:
        return {"label": "Harness TTFT floor", "value": f"{ms * 1000:.2f}", "unit": "µs"}
    return {"label": "Harness TTFT floor", "value": _n(ms), "unit": "ms"}


def _compare_tiles(runs: list[RunResult]) -> list[dict[str, str]]:
    return [
        {"label": "Runs", "value": str(len(runs)), "unit": ""},
        {
            "label": "Lineage",
            "value": "synthetic" if all(r.synthetic for r in runs) else "mixed/measured",
            "unit": "",
        },
    ]


def _primary_summary(run: RunResult) -> SuiteSummary | None:
    for s in run.summaries:
        if s.concurrency == 1:
            return s
    return run.summaries[0] if run.summaries else None


def _hardware_section(runs: list[RunResult]) -> str:
    blocks = ["<h2>Hardware and model</h2>", '<div class="grid-2">']
    for run in runs:
        label = html.escape(run.label or run.model.name)
        hw = run.hardware
        gpu = hw.gpus[0].name if hw.gpus else "none detected"
        rows = [
            ("Run", label),
            ("CPU", hw.cpu_name or "unknown"),
            ("RAM", f"{hw.ram_total_mib:.0f} MiB" if hw.ram_total_mib else "—"),
            ("GPU", gpu),
            ("OS", f"{hw.os} {hw.arch}"),
            ("Backend", f"{run.backend.type} {run.backend.version or ''}".strip()),
            ("Model", run.model.name),
            (
                "Quantization",
                _declared(run.model.quantization, run.model.quantization_source),
            ),
            (
                "Context",
                _declared(run.model.context_length, run.model.context_length_source),
            ),
            (
                "Sampling",
                f"T={run.sampling.temperature} top_p={run.sampling.top_p} "
                f"max_tokens={_max_tokens_label(run.sampling)} seed={run.sampling.seed}"
                + (
                    f" (dropped: {', '.join(run.sampling.dropped)})" if run.sampling.dropped else ""
                ),
            ),
        ]
        blocks.append("<dl class='kv'>")
        for k, v in rows:
            blocks.append(f"<div><dt>{html.escape(k)}</dt><dd>{html.escape(str(v))}</dd></div>")
        blocks.append("</dl>")
    blocks.append("</div>")
    return "\n".join(blocks)


def _quality_section(runs: list[RunResult]) -> str:
    if not any(r.quality and r.quality.enabled for r in runs):
        return ""
    rows = [
        "<h2>Quality probe</h2>",
        "<table>",
        "<thead><tr>",
        "<th>Run</th><th>OK</th><th>Contains</th><th>SHA-256</th><th>Output (trim)</th>",
        "</tr></thead><tbody>",
    ]
    for run in runs:
        q = run.quality
        if not q or not q.enabled:
            continue
        ok = "yes" if q.ok else "no"
        klass = "ok" if q.ok else "bad"
        contains = "—" if q.contains_hit is None else ("yes" if q.contains_hit else "no")
        digest = (q.output_sha256 or "—")[:16]
        out = html.escape((q.output_text or "")[:80])
        rows.append(
            f"<tr><td>{html.escape(run.label or run.model.name)}</td>"
            f"<td class='{klass}'>{ok}</td><td>{contains}</td>"
            f"<td class='num'>{html.escape(digest)}</td><td>{out}</td></tr>"
        )
    rows.append("</tbody></table>")
    rows.append(
        "<p class='sub'>Greedy (T=0) request, separate from speed trials. "
        "A failed probe means the speed numbers may describe a different quality level.</p>"
    )
    return "\n".join(rows)


def _charts_section(runs: list[RunResult]) -> str:
    parts = ["<h2>Metrics</h2>"]
    keys = _aligned_keys(runs)
    labels = [f"{name} c={conc}" for name, conc in keys]
    if len(runs) == 1:
        run = runs[0]
        lookup = {(s.name, s.concurrency): s for s in run.summaries}
        ttft = [lookup[k].ttft_ms.p50 if k in lookup else None for k in keys]
        decode = [lookup[k].decode_tps.p50 if k in lookup else None for k in keys]
        e2e = [lookup[k].e2e_ms.p95 if k in lookup else None for k in keys]
        parts.append(
            grouped_bar_chart(
                title="TTFT p50", categories=labels, series=[("p50", ttft)], unit="ms"
            )
        )
        parts.append(
            grouped_bar_chart(
                title="Decode throughput p50",
                categories=labels,
                series=[("p50", decode)],
                unit="tok/s",
            )
        )
        parts.append(
            grouped_bar_chart(
                title="End-to-end p95", categories=labels, series=[("p95", e2e)], unit="ms"
            )
        )
        concs = sorted({s.concurrency for s in run.summaries})
        if len(concs) > 1:
            suite_names: list[str] = []
            seen: set[str] = set()
            for s in run.summaries:
                if s.name not in seen:
                    seen.add(s.name)
                    suite_names.append(s.name)
            clabels = [str(c) for c in concs]
            series = []
            for suite in suite_names:
                tput = []
                for c in concs:
                    match = next(
                        (
                            row
                            for row in run.summaries
                            if row.concurrency == c and row.name == suite
                        ),
                        None,
                    )
                    tput.append(match.throughput_output_tps if match else None)
                series.append((suite, tput))
            parts.append(
                grouped_bar_chart(
                    title="Output tok/s vs concurrency (wall-clock, per suite)",
                    categories=clabels,
                    series=series,
                    unit="tok/s",
                )
            )
    else:
        series_ttft = []
        series_dec = []
        for run in runs:
            lookup = {(s.name, s.concurrency): s for s in run.summaries}
            name = run.label or run.model.name
            series_ttft.append(
                (name, [lookup[k].ttft_ms.p50 if k in lookup else None for k in keys])
            )
            series_dec.append(
                (name, [lookup[k].decode_tps.p50 if k in lookup else None for k in keys])
            )
        parts.append(
            grouped_bar_chart(
                title="TTFT p50 by run", categories=labels, series=series_ttft, unit="ms"
            )
        )
        parts.append(
            grouped_bar_chart(
                title="Decode tok/s p50 by run",
                categories=labels,
                series=series_dec,
                unit="tok/s",
            )
        )
    return "\n".join(parts)


def _tables_section(runs: list[RunResult]) -> str:
    parts = ["<h2>Summaries</h2>", "<table>", "<thead><tr>"]
    headers = [
        "Run",
        "Suite",
        "Conc",
        "n",
        "TTFT p50",
        "TTFT p95",
        "Decode p50",
        "Prefill p50 (server)",
        "Prefill proxy p50",
        "E2E p50",
        "E2E p95",
        "Tok/s wall",
    ]
    parts.append(
        "".join(
            f"<th class='num'>{h}</th>" if i > 2 else f"<th>{h}</th>" for i, h in enumerate(headers)
        )
    )
    parts.append("</tr></thead><tbody>")
    for run in runs:
        for s in run.summaries:
            parts.append("<tr>")
            cells = [
                html.escape(run.label or run.model.name),
                html.escape(s.name),
                str(s.concurrency),
                str(s.n_ok),
                _n(s.ttft_ms.p50),
                _n(s.ttft_ms.p95),
                _n(s.decode_tps.p50),
                _n(s.prefill_tps.p50),
                _n(s.prefill_tps_proxy.p50 if s.prefill_tps_proxy else None),
                _n(s.e2e_ms.p50),
                _n(s.e2e_ms.p95),
                _n(s.throughput_output_tps),
            ]
            for i, c in enumerate(cells):
                klass = "num" if i > 2 else ""
                parts.append(f"<td class='{klass}'>{c}</td>")
            parts.append("</tr>")
    parts.append("</tbody></table>")
    note_bits = []
    for run in runs:
        for s in run.summaries:
            for n in s.notes:
                if n not in note_bits:
                    note_bits.append(n)
    parts.append(
        "<p class='sub'>Prefill (server) is uncached prompt tokens / uncached prompt ms. "
        "Prefill proxy is prompt_tokens/TTFT and is never mixed into the server column.</p>"
    )
    if note_bits:
        parts.append("<ul class='notes'>")
        for n in note_bits:
            parts.append(f"<li>{html.escape(n)}</li>")
        parts.append("</ul>")

    if len(runs) > 1:
        parts.append("<h3>Deltas vs first run (p50 decode tok/s, TTFT)</h3>")
        parts.append(_delta_table_html(runs))

    run = runs[0]
    if run.harness.overhead:
        o = run.harness.overhead
        parts.append("<h2>Harness overhead</h2>")
        parts.append(
            f"<p>Zero-delay mock TTFT floor: <strong>{_fmt_overhead(o.zero_delay_ttft_ms)}</strong> · "
            f"e2e {_fmt_overhead(o.zero_delay_e2e_ms)} · "
            f"perf_counter_ns ≈ {o.timestamp_cost_ns:.0f} ns/call "
            f"({o.iterations} iterations, {o.tokens_in_probe} tokens).</p>"
        )

    parts.append("<h2>Trials</h2><details><summary>Raw measured trials</summary><table>")
    parts.append(
        "<thead><tr><th>Suite</th><th>Prompt</th><th>Conc</th><th>OK</th>"
        "<th class='num'>TTFT ms</th><th class='num'>Decode tok/s</th>"
        "<th class='num'>Out tok</th></tr></thead><tbody>"
    )
    for t in run.measured_trials()[:200]:
        klass = "ok" if t.ok else "bad"
        parts.append(
            f"<tr><td>{html.escape(t.suite)}</td><td>{html.escape(t.prompt_id)}</td>"
            f"<td>{t.concurrency}</td><td class='{klass}'>{'yes' if t.ok else 'no'}</td>"
            f"<td class='num'>{_n(t.ttft_ms)}</td><td class='num'>{_n(t.decode_tps)}</td>"
            f"<td class='num'>{t.output_tokens or 0}</td></tr>"
        )
    parts.append("</tbody></table></details>")
    return "\n".join(parts)


def _delta_table_html(runs: list[RunResult]) -> str:
    base = runs[0]
    keys = _aligned_keys(runs)
    parts = ["<table><thead><tr><th>Suite</th><th>Conc</th>"]
    for run in runs[1:]:
        name = html.escape(run.label or run.model.name)
        parts.append(
            f"<th class='num'>{name} Δ TTFT p50</th><th class='num'>{name} Δ decode p50</th>"
        )
    parts.append("</tr></thead><tbody>")
    bmap = {(s.name, s.concurrency): s for s in base.summaries}
    for key in keys:
        parts.append(f"<tr><td>{html.escape(key[0])}</td><td>{key[1]}</td>")
        b = bmap.get(key)
        for run in runs[1:]:
            omap = {(s.name, s.concurrency): s for s in run.summaries}
            o = omap.get(key)
            parts.append(
                f"<td class='num'>{_delta(b.ttft_ms.p50 if b else None, o.ttft_ms.p50 if o else None)}</td>"
            )
            parts.append(
                f"<td class='num'>{_delta(b.decode_tps.p50 if b else None, o.decode_tps.p50 if o else None)}</td>"
            )
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def _aligned_keys(runs: list[RunResult]) -> list[tuple[str, int]]:
    seen: list[tuple[str, int]] = []
    have: set[tuple[str, int]] = set()
    for run in runs:
        for s in run.summaries:
            key = (s.name, s.concurrency)
            if key not in have:
                have.add(key)
                seen.append(key)
    return seen


def _markdown_single(run: RunResult) -> list[str]:
    lines = [
        f"# local-llm-bench report — {run.model.name}",
        "",
        f"- Backend: `{run.backend.type}` {run.backend.version or ''}".rstrip(),
        f"- Sampling: T={run.sampling.temperature} top_p={run.sampling.top_p} "
        f"max_tokens={_max_tokens_label(run.sampling)} seed={run.sampling.seed}",
        f"- Hardware: {run.hardware.cpu_name or 'CPU?'} · "
        f"{(run.hardware.gpus[0].name if run.hardware.gpus else 'no GPU')}",
        "",
        "| Suite | Conc | n | TTFT p50 (ms) | TTFT p95 (ms) | Decode p50 (tok/s) | E2E p50 (ms) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in run.summaries:
        lines.append(
            f"| {s.name} | {s.concurrency} | {s.n_ok} | {_n(s.ttft_ms.p50)} | "
            f"{_n(s.ttft_ms.p95)} | {_n(s.decode_tps.p50)} | {_n(s.e2e_ms.p50)} |"
        )
    if run.harness.overhead:
        o = run.harness.overhead
        lines += [
            "",
            f"Harness overhead (zero-delay mock): TTFT floor **{o.zero_delay_ttft_ms:.3f} ms**.",
        ]
    return lines


def _markdown_compare(runs: list[RunResult]) -> list[str]:
    names = [r.label or r.model.name for r in runs]
    lines = [
        "# local-llm-bench comparison",
        "",
        "| Suite | Conc | metric | " + " | ".join(names) + " |",
        "| --- | ---: | --- | " + " | ".join("---:" for _ in runs) + " |",
    ]
    keys = _aligned_keys(runs)
    metrics = [
        ("TTFT p50 ms", lambda s: s.ttft_ms.p50),
        ("Decode p50 tok/s", lambda s: s.decode_tps.p50),
        ("E2E p50 ms", lambda s: s.e2e_ms.p50),
    ]
    for key in keys:
        for label, fn in metrics:
            vals = []
            for run in runs:
                lookup = {(s.name, s.concurrency): s for s in run.summaries}
                s = lookup.get(key)
                vals.append(_n(fn(s) if s else None))
            lines.append(f"| {key[0]} | {key[1]} | {label} | " + " | ".join(vals) + " |")
    return lines


def _max_tokens_label(sampling: object) -> str:
    lo = getattr(sampling, "max_tokens_min", None)
    hi = getattr(sampling, "max_tokens_max", None)
    default = getattr(sampling, "max_tokens", None)
    if isinstance(lo, int) and isinstance(hi, int) and lo != hi:
        return f"{lo}-{hi}"
    if default is not None:
        return str(default)
    return "—"


def _declared(value: object, source: str | None) -> str:
    if value is None:
        return "—"
    if source == "applied_num_ctx":
        return f"{value} (sent as num_ctx)"
    if source == "user_declared":
        return f"{value} (user-declared, not applied)"
    return str(value)


def _fmt_overhead(ms: float) -> str:
    if ms < 0.01:
        return f"{ms * 1000:.2f} µs"
    return f"{ms:.3f} ms"


def _n(value: float | None) -> str:
    if value is None:
        return "—"
    av = abs(value)
    if av == 0:
        return "0"
    if av < 0.01:
        return f"{value:.4f}"
    if av >= 100:
        return f"{value:.1f}"
    if av >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _delta(base: float | None, other: float | None) -> str:
    if base is None or other is None or base == 0:
        return "—"
    pct = (other - base) / base * 100.0
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"
