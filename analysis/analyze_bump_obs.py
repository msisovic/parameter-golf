#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


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
        val = row.get(key, default)
        if isinstance(val, str) and val.endswith("ms"):
            val = val[:-2]
        return float(val)
    except ValueError:
        return default


def to_int(row, key, default=0):
    try:
        return int(row.get(key, default))
    except ValueError:
        return default


def read_rows(path):
    probes = []
    counterfactuals = []
    layer_stats = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        tag = parts[0]
        row = parse_kv(parts[1:])
        if tag == "bump_probe":
            probes.append(row)
        elif tag == "bump_counterfactual":
            counterfactuals.append(row)
        elif tag == "bump_layer_stats":
            layer_stats.append(row)
    return probes, counterfactuals, layer_stats


def mean(vals):
    vals = [v for v in vals if not math.isnan(v)]
    return sum(vals) / len(vals) if vals else math.nan


def minmax(vals):
    vals = [v for v in vals if not math.isnan(v)]
    return (min(vals), max(vals)) if vals else (math.nan, math.nan)


def phase_order(rows):
    phases = []
    for row in rows:
        phase = row.get("phase", "")
        if phase and phase not in phases:
            phases.append(phase)
    return phases


def summarize_by_phase(rows, metrics):
    for phase in phase_order(rows):
        subset = [row for row in rows if row.get("phase") == phase]
        print(f"\n{phase} n={len(subset)}")
        for metric in metrics:
            vals = [to_float(row, metric) for row in subset]
            lo, hi = minmax(vals)
            print(f"  {metric}: mean={mean(vals):.8f} min={lo:.8f} max={hi:.8f}")


def summarize_counterfactual_transition(rows):
    if not rows:
        return
    print("\ntransition_summary")
    by_phase = {phase: [row for row in rows if row.get("phase") == phase] for phase in phase_order(rows)}
    for phase, subset in by_phase.items():
        if not subset:
            continue
        first = subset[0]
        last = subset[-1]
        deltas = [to_float(row, "delta_loss") for row in subset]
        policies = [to_float(row, "policy_loss") for row in subset]
        print(
            f"  {phase}: steps={to_int(first, 'step')}..{to_int(last, 'step')} "
            f"policy_mean={mean(policies):.8f} delta_mean={mean(deltas):.8f} "
            f"first_delta={to_float(first, 'delta_loss'):.8f} "
            f"last_delta={to_float(last, 'delta_loss'):.8f}"
        )
    post = by_phase.get("post_early", [])
    if post:
        first = post[0]
        print(
            "  hard_switch_first_step: "
            f"step={to_int(first, 'step')} "
            f"loop_off={to_float(first, 'loop_off_loss'):.8f} "
            f"loop_on={to_float(first, 'loop_on_loss'):.8f} "
            f"delta={to_float(first, 'delta_loss'):.8f}"
        )
        recovered = next((row for row in post if to_float(row, "delta_loss") <= 0.0), None)
        if recovered is not None:
            print(
                "  recovery_crossing: "
                f"step={to_int(recovered, 'step')} "
                f"delta={to_float(recovered, 'delta_loss'):.8f}"
            )
        else:
            print("  recovery_crossing: none in post_early")


def summarize_layers(rows):
    if not rows:
        return
    phases = []
    for row in rows:
        phase = row.get("phase", "")
        if phase and phase not in phases:
            phases.append(phase)
    for phase in phases:
        subset = [row for row in rows if row.get("phase") == phase]
        layers = sorted({to_int(row, "layer") for row in subset})
        print(f"\nlayer_stats {phase}")
        print("  layer looped grad_norm param_norm grad_param_ratio")
        for layer in layers:
            layer_rows = [row for row in subset if to_int(row, "layer") == layer]
            looped = max(to_int(row, "looped") for row in layer_rows)
            grad_norm = mean([to_float(row, "grad_norm") for row in layer_rows])
            param_norm = mean([to_float(row, "param_norm") for row in layer_rows])
            ratio = mean([to_float(row, "grad_param_ratio") for row in layer_rows])
            print(f"  {layer:>5} {looped:>6} {grad_norm:.8e} {param_norm:.8e} {ratio:.8e}")


def avg_nested(x):
    if isinstance(x, (int, float)):
        return float(x)
    vals = []
    stack = [x]
    while stack:
        cur = stack.pop()
        if isinstance(cur, list):
            stack.extend(cur)
        else:
            vals.append(float(cur))
    return sum(vals) / len(vals) if vals else math.nan


def read_scalar_records(path):
    records = []
    if path is None:
        return records
    scalar_path = Path(path)
    if not scalar_path.exists():
        raise FileNotFoundError(scalar_path)
    with scalar_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            loop_extra = row.get("loop_extra")
            if loop_extra:
                row["_summary"] = loop_extra.get("summaries", {})
                attn = loop_extra.get("attn_scale", [])
                mlp = loop_extra.get("mlp_scale", [])
                resid = loop_extra.get("resid_mix", [])
                per_call = []
                for pass_idx, pass_rows in enumerate(attn):
                    for local_idx, attn_vec in enumerate(pass_rows):
                        resid_mix = resid[pass_idx][local_idx]
                        per_call.append(
                            {
                                "extra_pass": pass_idx,
                                "loop_local": local_idx,
                                "attn_mean": avg_nested(attn_vec),
                                "mlp_mean": avg_nested(mlp[pass_idx][local_idx]),
                                "resid_current_mean": avg_nested(resid_mix[0]),
                                "resid_x0_mean": avg_nested(resid_mix[1]),
                            }
                        )
                row["_per_call"] = per_call
            records.append(row)
    return records


def nearest_record(records, step):
    if not records:
        return None
    return min(records, key=lambda row: abs(int(row.get("step", 0)) - step))


def summarize_scalar_records(records):
    if not records:
        return
    print(f"\nloop_scalar_records: n={len(records)} steps={records[0]['step']}..{records[-1]['step']}")
    interesting = []
    for step in (2100, 2199, 2200, 2201, 2210, 2319, records[-1]["step"]):
        row = nearest_record(records, step)
        if row is not None and row not in interesting:
            interesting.append(row)
    for row in interesting:
        summary = row.get("_summary", {})
        if not summary:
            continue
        print(
            f"\nscalar step={row['step']} reason={row.get('reason')} "
            f"looping_active={int(bool(row.get('looping_active')))}"
        )
        for name in ("attn_scale", "mlp_scale", "resid_mix"):
            s = summary.get(name, {})
            print(
                f"  {name}: mean={s.get('mean', math.nan):.8f} "
                f"min={s.get('min', math.nan):.8f} max={s.get('max', math.nan):.8f} "
                f"norm={s.get('norm', math.nan):.8f}"
            )
        per_call = row.get("_per_call", [])
        if per_call:
            print("  per_call extra_pass local attn_mean mlp_mean resid_current resid_x0")
            for call in per_call:
                print(
                    f"    {call['extra_pass']:>2} {call['loop_local']:>5} "
                    f"{call['attn_mean']:.8f} {call['mlp_mean']:.8f} "
                    f"{call['resid_current_mean']:.8f} {call['resid_x0_mean']:.8f}"
                )


def write_svg(path, series, title, ylabel, markers=()):
    points = [pt for item in series for pt in item["points"]]
    if not points:
        return
    width, height = 1040, 600
    margin = {"left": 82, "right": 28, "top": 54, "bottom": 66}
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

    colors = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf")
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
    for idx, marker in enumerate(markers):
        xv = marker[0]
        if xmin <= xv <= xmax:
            x = sx(xv)
            elems.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{margin["top"]}" y2="{height - margin["bottom"]}" stroke="#111" stroke-width="1.4" stroke-dasharray="5 5"/>')
            elems.append(f'<text x="{x + 6:.1f}" y="{margin["top"] + 16 + 16 * idx}" font-family="sans-serif" font-size="12" fill="#111">{marker[1]}</text>')
    lx, ly = width - margin["right"] - 260, margin["top"] + 18
    for idx, item in enumerate(series):
        color = item.get("color", colors[idx % len(colors)])
        pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in item["points"])
        elems.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.3" points="{pts}"/>')
        y = ly + idx * 22
        elems.append(f'<line x1="{lx}" x2="{lx + 28}" y1="{y}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        elems.append(f'<text x="{lx + 36}" y="{y + 4}" font-family="sans-serif" font-size="13" fill="#222">{item["name"]}</text>')
    elems.append(f'<text x="{width / 2:.1f}" y="{height - 18}" text-anchor="middle" font-family="sans-serif" font-size="14">step</text>')
    elems.append(f'<text transform="translate(20 {height / 2:.1f}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="14">{ylabel}</text>')
    elems.append("</svg>")
    Path(path).write_text("\n".join(elems), encoding="utf-8")


def maybe_plot_svg(path, probes):
    out_dir = Path(path).with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)
    markers = ((2200, "loop on"),)
    if probes:
        write_svg(
            out_dir / "bump_probe_loss.svg",
            [{"name": "probe loss", "points": [(to_int(row, "step"), to_float(row, "loss")) for row in probes]}],
            "Bump Probe Loss",
            "loss",
            markers,
        )


def maybe_plot(path, probes):
    maybe_plot_svg(path, probes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logfile")
    parser.add_argument("--scalars-jsonl", default="")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    probes, counterfactuals, layer_stats = read_rows(args.logfile)
    print(f"bump_probe rows: {len(probes)}")
    summarize_by_phase(probes, ("loss", "bpb", "eval_time"))
    print(f"\nbump_counterfactual rows: {len(counterfactuals)}")
    summarize_by_phase(
        counterfactuals,
        (
            "p_loop",
            "policy_loss",
            "loop_off_loss",
            "loop_on_loss",
            "delta_loss",
            "policy_bpb",
            "loop_off_bpb",
            "loop_on_bpb",
            "delta_bpb",
        ),
    )
    summarize_counterfactual_transition(counterfactuals)
    print(f"\nbump_layer_stats rows: {len(layer_stats)}")
    summarize_layers(layer_stats)
    scalar_records = read_scalar_records(args.scalars_jsonl) if args.scalars_jsonl else []
    summarize_scalar_records(scalar_records)
    if args.plot:
        maybe_plot(args.logfile, probes)


if __name__ == "__main__":
    main()
