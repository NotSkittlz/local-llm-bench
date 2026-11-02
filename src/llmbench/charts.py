"""Inline SVG charts. Colors come from CSS variables so dark/light both work."""

from __future__ import annotations

import html
from collections.abc import Sequence


def bar_chart(
    *,
    title: str,
    labels: Sequence[str],
    values: Sequence[float | None],
    unit: str,
    width: int = 640,
    height: int = 260,
) -> str:
    series = [("value", list(values))]
    return grouped_bar_chart(
        title=title,
        categories=labels,
        series=series,
        unit=unit,
        width=width,
        height=height,
    )


def grouped_bar_chart(
    *,
    title: str,
    categories: Sequence[str],
    series: Sequence[tuple[str, Sequence[float | None]]],
    unit: str,
    width: int = 720,
    height: int = 280,
) -> str:
    if not categories:
        return ""
    left, right, top, bottom = 56, 16, 36, 48
    plot_w = width - left - right
    plot_h = height - top - bottom
    n_cat = len(categories)
    n_ser = max(1, len(series))
    group_w = plot_w / n_cat
    bar_w = min(36.0, group_w / (n_ser + 0.6))
    ymax = 0.0
    for _, vals in series:
        for v in vals:
            if v is not None and v > ymax:
                ymax = v
    ymax = ymax * 1.15 if ymax > 0 else 1.0
    ticks = 4
    parts: list[str] = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{html.escape(title)}">',
        f'<text class="chart-title" x="{left}" y="18">{html.escape(title)}</text>',
        f'<text class="chart-unit" x="{left}" y="32">{html.escape(unit)}</text>',
    ]
    for i in range(ticks + 1):
        y = top + plot_h - (plot_h * i / ticks)
        val = ymax * i / ticks
        parts.append(
            f'<line class="grid" x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}" />'
        )
        parts.append(
            f'<text class="tick" x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">'
            f"{_fmt(val)}</text>"
        )
    for ci, label in enumerate(categories):
        gx = left + ci * group_w
        cx = gx + group_w / 2
        parts.append(
            f'<text class="tick cat" x="{cx:.1f}" y="{height - 14}" text-anchor="middle">'
            f"{html.escape(_clip(label, 18))}</text>"
        )
        for si, (sname, vals) in enumerate(series):
            v = vals[ci] if ci < len(vals) else None
            if v is None:
                continue
            h = (v / ymax) * plot_h
            x = gx + (group_w - n_ser * bar_w) / 2 + si * bar_w
            y = top + plot_h - h
            parts.append(
                f'<rect class="bar bar-{si}" x="{x:.1f}" y="{y:.1f}" '
                f'width="{bar_w:.1f}" height="{h:.1f}">'
                f"<title>{html.escape(sname)} · {html.escape(label)}: {_fmt(v)} {html.escape(unit)}</title>"
                f"</rect>"
            )
    if n_ser > 1:
        lx = left
        for si, (sname, _) in enumerate(series):
            parts.append(
                f'<rect class="bar bar-{si}" x="{lx}" y="{height - 8}" width="8" height="8" />'
            )
            parts.append(
                f'<text class="legend" x="{lx + 12}" y="{height - 1}">{html.escape(sname)}</text>'
            )
            lx += 12 + min(140, 8 * len(sname) + 24)
    parts.append("</svg>")
    return "\n".join(parts)


def _fmt(v: float) -> str:
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


def _clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"
