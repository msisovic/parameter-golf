#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot transition_monitor JSON lines from a train_gpt log."
    )
    parser.add_argument("logfile", type=Path, help="Path to training log file")
    parser.add_argument(
        "--label",
        default="seq_len_curriculum",
        help="Transition label to plot, default: seq_len_curriculum",
    )
    parser.add_argument(
        "--event-step",
        type=int,
        default=None,
        help="Specific transition event_step to plot; defaults to the latest matching event",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output files; defaults to analysis/<log-stem>",
    )
    return parser.parse_args()


def load_transition_rows(logfile: Path, label: str) -> list[dict]:
    rows: list[dict] = []
    prefix = "transition_monitor:"
    with logfile.open("r", encoding="utf-8") as f:
        for line in f:
            if prefix not in line:
                continue
            _, payload = line.split(prefix, 1)
            payload = payload.strip()
            if not payload.startswith("{"):
                continue
            try:
                record = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if record.get("label") == label:
                rows.append(record)
    return rows


def select_event_step(rows: list[dict], event_step: int | None) -> int:
    event_steps = sorted({int(row["event_step"]) for row in rows})
    if not event_steps:
        raise ValueError("No matching transition_monitor rows found")
    if event_step is None:
        return event_steps[-1]
    if event_step not in event_steps:
        raise ValueError(
            f"event_step={event_step} not found; available values: {event_steps}"
        )
    return event_step


def write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "event",
        "label",
        "event_step",
        "phase",
        "offset",
        "step",
        "frac",
        "lr_scale",
        "seq_len",
        "looping",
        "loss",
        "step_ms",
        "pre_loss_avg",
        "pre_step_ms_avg",
        "loss_delta_vs_pre_avg",
        "step_ms_delta_vs_pre_avg",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _nice_min_max(values: Iterable[float]) -> tuple[float, float]:
    vals = list(values)
    low = min(vals)
    high = max(vals)
    if math.isclose(low, high):
        pad = max(abs(low) * 0.05, 1e-6)
        return low - pad, high + pad
    pad = (high - low) * 0.08
    return low - pad, high + pad


def _polyline_points(
    rows: list[dict],
    field: str,
    x_map,
    y_min: float,
    y_max: float,
    plot_left: float,
    plot_top: float,
    plot_width: float,
    plot_height: float,
) -> str:
    points = []
    denom = max(y_max - y_min, 1e-12)
    for row in rows:
        x = plot_left + x_map(row["offset"]) * plot_width
        y = plot_top + (1.0 - (float(row[field]) - y_min) / denom) * plot_height
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def render_svg(rows: list[dict], label: str, event_step: int, path: Path) -> None:
    width = 1200
    height = 720
    margin_left = 80
    margin_right = 40
    margin_top = 70
    margin_bottom = 60
    panel_gap = 34
    panel_height = (height - margin_top - margin_bottom - panel_gap) / 2
    plot_width = width - margin_left - margin_right
    plot_left = margin_left
    loss_top = margin_top
    step_top = margin_top + panel_height + panel_gap

    offsets = [int(row["offset"]) for row in rows]
    min_offset = min(offsets)
    max_offset = max(offsets)
    span = max(max_offset - min_offset, 1)

    def x_map(offset: int) -> float:
        return (offset - min_offset) / span

    loss_min, loss_max = _nice_min_max(float(row["loss"]) for row in rows)
    step_ms_min, step_ms_max = _nice_min_max(float(row["step_ms"]) for row in rows)
    loss_points = _polyline_points(
        rows,
        "loss",
        x_map,
        loss_min,
        loss_max,
        plot_left,
        loss_top,
        plot_width,
        panel_height,
    )
    step_points = _polyline_points(
        rows,
        "step_ms",
        x_map,
        step_ms_min,
        step_ms_max,
        plot_left,
        step_top,
        plot_width,
        panel_height,
    )
    pre_count = sum(1 for row in rows if row["phase"] == "pre")
    bump_x = plot_left + x_map(0) * plot_width
    pre_bg_width = max(bump_x - plot_left, 0.0)
    post_bg_width = max(plot_left + plot_width - bump_x, 0.0)

    def axis_y(top: float, panel_h: float) -> float:
        return top + panel_h

    def tick_lines(top: float, panel_h: float, y_min: float, y_max: float) -> list[str]:
        lines = []
        for i in range(5):
            frac = i / 4 if 4 else 0.0
            y = top + panel_h * frac
            value = y_max - (y_max - y_min) * frac
            lines.append(
                f'<line x1="{plot_left:.1f}" y1="{y:.1f}" x2="{plot_left + plot_width:.1f}" y2="{y:.1f}" '
                f'stroke="#d9dde3" stroke-width="1" />'
            )
            lines.append(
                f'<text x="{plot_left - 10:.1f}" y="{y + 4:.1f}" text-anchor="end" '
                f'font-size="12" fill="#334155">{value:.3f}</text>'
            )
        return lines

    x_tick_lines = []
    x_tick_labels = []
    for offset in sorted(set(offsets)):
        if len(offsets) > 12 and offset not in (min_offset, 0, max_offset):
            if offset % max(1, math.ceil(span / 8)) != 0:
                continue
        x = plot_left + x_map(offset) * plot_width
        x_tick_lines.append(
            f'<line x1="{x:.1f}" y1="{loss_top:.1f}" x2="{x:.1f}" y2="{step_top + panel_height:.1f}" '
            f'stroke="#eef2f7" stroke-width="1" />'
        )
        x_tick_labels.append(
            f'<text x="{x:.1f}" y="{height - 20:.1f}" text-anchor="middle" '
            f'font-size="12" fill="#334155">{offset}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff" />
<text x="{plot_left:.1f}" y="32" font-size="24" font-family="sans-serif" fill="#0f172a">
  {label} transition at step {event_step}
</text>
<text x="{plot_left:.1f}" y="52" font-size="13" font-family="sans-serif" fill="#475569">
  pre samples: {pre_count} | post samples: {len(rows) - pre_count} | offset 0 = transition step
</text>
<rect x="{plot_left:.1f}" y="{loss_top:.1f}" width="{pre_bg_width:.1f}" height="{panel_height:.1f}" fill="#f8fafc" />
<rect x="{bump_x:.1f}" y="{loss_top:.1f}" width="{post_bg_width:.1f}" height="{panel_height:.1f}" fill="#fff7ed" />
<rect x="{plot_left:.1f}" y="{step_top:.1f}" width="{pre_bg_width:.1f}" height="{panel_height:.1f}" fill="#f8fafc" />
<rect x="{bump_x:.1f}" y="{step_top:.1f}" width="{post_bg_width:.1f}" height="{panel_height:.1f}" fill="#fff7ed" />
{''.join(x_tick_lines)}
{''.join(tick_lines(loss_top, panel_height, loss_min, loss_max))}
{''.join(tick_lines(step_top, panel_height, step_ms_min, step_ms_max))}
<line x1="{plot_left:.1f}" y1="{axis_y(loss_top, panel_height):.1f}" x2="{plot_left + plot_width:.1f}" y2="{axis_y(loss_top, panel_height):.1f}" stroke="#94a3b8" stroke-width="1.5" />
<line x1="{plot_left:.1f}" y1="{axis_y(step_top, panel_height):.1f}" x2="{plot_left + plot_width:.1f}" y2="{axis_y(step_top, panel_height):.1f}" stroke="#94a3b8" stroke-width="1.5" />
<line x1="{plot_left:.1f}" y1="{loss_top:.1f}" x2="{plot_left:.1f}" y2="{axis_y(loss_top, panel_height):.1f}" stroke="#94a3b8" stroke-width="1.5" />
<line x1="{plot_left:.1f}" y1="{step_top:.1f}" x2="{plot_left:.1f}" y2="{axis_y(step_top, panel_height):.1f}" stroke="#94a3b8" stroke-width="1.5" />
<line x1="{bump_x:.1f}" y1="{loss_top:.1f}" x2="{bump_x:.1f}" y2="{step_top + panel_height:.1f}" stroke="#ea580c" stroke-width="2" stroke-dasharray="6 4" />
<polyline fill="none" stroke="#2563eb" stroke-width="2.5" points="{loss_points}" />
<polyline fill="none" stroke="#059669" stroke-width="2.5" points="{step_points}" />
<text x="{plot_left:.1f}" y="{loss_top - 10:.1f}" font-size="16" font-family="sans-serif" fill="#0f172a">Loss</text>
<text x="{plot_left:.1f}" y="{step_top - 10:.1f}" font-size="16" font-family="sans-serif" fill="#0f172a">Step ms</text>
<text x="{plot_left + plot_width:.1f}" y="{loss_top - 10:.1f}" text-anchor="end" font-size="12" font-family="sans-serif" fill="#2563eb">blue: loss</text>
<text x="{plot_left + plot_width:.1f}" y="{step_top - 10:.1f}" text-anchor="end" font-size="12" font-family="sans-serif" fill="#059669">green: step_ms</text>
{''.join(x_tick_labels)}
<text x="{plot_left + plot_width / 2:.1f}" y="{height - 40:.1f}" text-anchor="middle" font-size="13" font-family="sans-serif" fill="#334155">
  sample offset relative to transition
</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = load_transition_rows(args.logfile, args.label)
    event_step = select_event_step(rows, args.event_step)
    rows = [row for row in rows if int(row["event_step"]) == event_step]
    rows.sort(key=lambda row: (int(row["offset"]), int(row["step"])))

    output_dir = args.output_dir or Path("analysis") / args.logfile.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.label}_step{event_step}"
    csv_path = output_dir / f"{stem}.csv"
    svg_path = output_dir / f"{stem}.svg"

    write_csv(rows, csv_path)
    render_svg(rows, args.label, event_step, svg_path)

    print(f"wrote {csv_path}")
    print(f"wrote {svg_path}")


if __name__ == "__main__":
    main()
