#!/usr/bin/env python3
import argparse
import math
from pathlib import Path


COLORS = (
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
)


def parse_kv(parts):
    out = {}
    for part in parts:
        if ":" not in part:
            continue
        key, val = part.split(":", 1)
        out[key] = val
    return out


def to_float(row, key, default=math.nan):
    try:
        return float(row.get(key, default))
    except ValueError:
        return default


def to_int(row, key, default=0):
    try:
        return int(row.get(key, default))
    except ValueError:
        return default


def read_counterfactuals(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if not parts or parts[0] != "bump_counterfactual":
            continue
        row = parse_kv(parts[1:])
        step = to_int(row, "step", -1)
        if step < 0:
            continue
        off = to_float(row, "loop_off_loss")
        on = to_float(row, "loop_on_loss")
        p_loop = to_float(row, "p_loop")
        policy = to_float(row, "policy_loss")
        if math.isnan(policy) and not math.isnan(off) and not math.isnan(on) and not math.isnan(p_loop):
            policy = (1.0 - p_loop) * off + p_loop * on
        rows.append(
            {
                "step": step,
                "p_loop": p_loop,
                "policy_loss": policy,
                "loop_off_loss": off,
                "loop_on_loss": on,
                "delta_loss": to_float(row, "delta_loss"),
            }
        )
    return rows


def metric_points(rows, metric):
    return [
        (row["step"], row[metric])
        for row in rows
        if metric in row and not math.isnan(row[metric])
    ]


def write_svg(path, series, title, ylabel, markers):
    width, height = 1040, 600
    margin = {"left": 82, "right": 28, "top": 54, "bottom": 66}
    points = [pt for s in series for pt in s["points"]]
    if not points:
        return
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    ypad = (ymax - ymin) * 0.08 or 1.0
    ymin -= ypad
    ymax += ypad
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]

    def sx(x):
        return margin["left"] + (x - xmin) / max(xmax - xmin, 1) * plot_w

    def sy(y):
        return margin["top"] + (ymax - y) / max(ymax - ymin, 1e-12) * plot_h

    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2:.1f}" y="30" text-anchor="middle" font-family="sans-serif" font-size="20" font-weight="700">{title}</text>',
        f'<rect x="{margin["left"]}" y="{margin["top"]}" width="{plot_w}" height="{plot_h}" fill="#fafafa" stroke="#cccccc"/>',
    ]
    for i in range(6):
        yv = ymin + (ymax - ymin) * i / 5
        y = sy(yv)
        elems.append(f'<line x1="{margin["left"]}" x2="{width - margin["right"]}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e5e5e5"/>')
        elems.append(f'<text x="{margin["left"] - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="monospace" font-size="12" fill="#333">{yv:.3f}</text>')
    tick_start = int(math.ceil(xmin / 100.0) * 100)
    for xv in range(tick_start, int(xmax) + 1, 100):
        x = sx(xv)
        elems.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{margin["top"]}" y2="{height - margin["bottom"]}" stroke="#eeeeee"/>')
        elems.append(f'<text x="{x:.1f}" y="{height - margin["bottom"] + 24}" text-anchor="middle" font-family="monospace" font-size="12" fill="#333">{xv}</text>')
    for marker in markers:
        xv = marker["step"]
        if xmin <= xv <= xmax:
            x = sx(xv)
            elems.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{margin["top"]}" y2="{height - margin["bottom"]}" stroke="{marker.get("color", "#111")}" stroke-width="1.4" stroke-dasharray="5 5"/>')
            elems.append(f'<text x="{x + 6:.1f}" y="{margin["top"] + 16 + 16 * marker.get("lane", 0)}" font-family="sans-serif" font-size="12" fill="{marker.get("color", "#111")}">{marker["label"]}</text>')
    for idx, s in enumerate(series):
        color = s.get("color", COLORS[idx % len(COLORS)])
        pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in s["points"])
        elems.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.3" points="{pts}"/>')
    lx, ly = width - margin["right"] - 260, margin["top"] + 18
    for idx, s in enumerate(series):
        color = s.get("color", COLORS[idx % len(COLORS)])
        y = ly + idx * 22
        elems.append(f'<line x1="{lx}" x2="{lx + 28}" y1="{y}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        elems.append(f'<text x="{lx + 36}" y="{y + 4}" font-family="sans-serif" font-size="13" fill="#222">{s["name"]}</text>')
    elems.append(f'<text x="{width / 2:.1f}" y="{height - 18}" text-anchor="middle" font-family="sans-serif" font-size="14">step</text>')
    elems.append(f'<text transform="translate(20 {height / 2:.1f}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="14">{ylabel}</text>')
    elems.append("</svg>")
    Path(path).write_text("\n".join(elems), encoding="utf-8")


def summarize(name, rows):
    policy = metric_points(rows, "policy_loss")
    delta = metric_points(rows, "delta_loss")
    if not policy:
        print(f"{name}: no policy_loss rows")
        return
    vals = [y for _, y in policy]
    pvals = [row["p_loop"] for row in rows if not math.isnan(row["p_loop"])]
    print(
        f"{name}: rows={len(rows)} steps={policy[0][0]}..{policy[-1][0]} "
        f"policy_min={min(vals):.6f} policy_max={max(vals):.6f} "
        f"p_loop={min(pvals):.3f}..{max(pvals):.3f}"
    )
    if delta:
        dvals = [y for _, y in delta]
        print(f"  delta_min={min(dvals):.6f} delta_max={max(dvals):.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="analysis/bump_obs_overlay")
    parser.add_argument(
        "runs",
        nargs="+",
        help="Run spec as label=path.log, or plain path.log",
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for idx, spec in enumerate(args.runs):
        if "=" in spec:
            label, path = spec.split("=", 1)
        else:
            path = spec
            label = Path(path).stem
        rows = read_counterfactuals(path)
        runs.append({"label": label, "path": path, "rows": rows, "color": COLORS[idx % len(COLORS)]})
        summarize(label, rows)
    markers = [
        {"step": 2000, "label": "ramp start", "color": "#777", "lane": 0},
        {"step": 2200, "label": "hard switch / midpoint", "color": "#111", "lane": 1},
        {"step": 2400, "label": "ramp end", "color": "#777", "lane": 2},
    ]
    for metric, title, ylabel in (
        ("policy_loss", "Policy Loss Overlay", "loss"),
        ("loop_on_loss", "Loop-On Loss Overlay", "loss"),
        ("delta_loss", "Loop-On Minus Loop-Off Overlay", "loss delta"),
        ("p_loop", "Loop Probability Overlay", "p_loop"),
    ):
        series = []
        for run in runs:
            pts = metric_points(run["rows"], metric)
            if pts:
                series.append({"name": run["label"], "points": pts, "color": run["color"]})
        write_svg(out_dir / f"{metric}_overlay.svg", series, title, ylabel, markers)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
