#!/usr/bin/env python3
import argparse
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


def summarize_by_phase(rows, metrics):
    phases = []
    for row in rows:
        phase = row.get("phase", "")
        if phase and phase not in phases:
            phases.append(phase)
    for phase in phases:
        subset = [row for row in rows if row.get("phase") == phase]
        print(f"\n{phase} n={len(subset)}")
        for metric in metrics:
            print(f"  {metric}: {mean([to_float(row, metric) for row in subset]):.8f}")


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


def maybe_plot(path, probes, counterfactuals):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    out_dir = Path(path).with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)
    if probes:
        steps = [to_int(row, "step") for row in probes]
        losses = [to_float(row, "loss") for row in probes]
        plt.figure()
        plt.plot(steps, losses, marker="o")
        plt.xlabel("step")
        plt.ylabel("fixed probe loss")
        plt.title("Bump Probe")
        plt.tight_layout()
        plt.savefig(out_dir / "bump_probe_loss.png", dpi=160)
        plt.close()
    if counterfactuals:
        steps = [to_int(row, "step") for row in counterfactuals]
        off = [to_float(row, "loop_off_loss") for row in counterfactuals]
        on = [to_float(row, "loop_on_loss") for row in counterfactuals]
        delta = [to_float(row, "delta_loss") for row in counterfactuals]
        plt.figure()
        plt.plot(steps, off, marker="o", label="loop off")
        plt.plot(steps, on, marker="o", label="loop on")
        plt.xlabel("step")
        plt.ylabel("fixed probe loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "bump_counterfactual_loss.png", dpi=160)
        plt.close()
        plt.figure()
        plt.plot(steps, delta, marker="o")
        plt.xlabel("step")
        plt.ylabel("loop_on - loop_off loss")
        plt.title("Bump Counterfactual Delta")
        plt.tight_layout()
        plt.savefig(out_dir / "bump_counterfactual_delta.png", dpi=160)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logfile")
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
    print(f"\nbump_layer_stats rows: {len(layer_stats)}")
    summarize_layers(layer_stats)
    if args.plot:
        maybe_plot(args.logfile, probes, counterfactuals)


if __name__ == "__main__":
    main()
