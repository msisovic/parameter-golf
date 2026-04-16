#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_gpt


SLICE_GROUPS = {
    "docs_gt_2048_position_buckets": [
        ("1-256", 2049, 1, 256),
        ("257-512", 2049, 257, 512),
        ("513-1024", 2049, 513, 1024),
        ("1025-2048", 2049, 1025, 2048),
        ("2049-4096", 2049, 2049, 4096),
        ("4097+", 2049, 4097, None),
    ],
    "docs_gt_2048_excluding_prefix256": [
        ("257-512", 2049, 257, 512),
        ("513-1024", 2049, 513, 1024),
        ("1025-2048", 2049, 1025, 2048),
        ("2049-4096", 2049, 2049, 4096),
        ("4097+", 2049, 4097, None),
    ],
    "docs_gt_4096_position_buckets": [
        ("1-256", 4097, 1, 256),
        ("257-512", 4097, 257, 512),
        ("513-1024", 4097, 513, 1024),
        ("1025-2048", 4097, 1025, 2048),
        ("2049-4096", 4097, 2049, 4096),
        ("4097-8192", 4097, 4097, 8192),
        ("8193+", 4097, 8193, None),
    ],
    "docs_gt_4096_excluding_prefix256": [
        ("257-512", 4097, 257, 512),
        ("513-1024", 4097, 513, 1024),
        ("1025-2048", 4097, 1025, 2048),
        ("2049-4096", 4097, 2049, 4096),
        ("4097-8192", 4097, 4097, 8192),
        ("8193+", 4097, 8193, None),
    ],
}


def bucket_mask(values: torch.Tensor, lower: int, upper: int | None) -> torch.Tensor:
    mask = values >= lower
    if upper is not None:
        mask &= values <= upper
    return mask


def main() -> None:
    run_id = os.environ.get("RUN_ID", "sp8192_deeper_doc_eval")
    output_dir = Path(os.environ.get("ANALYSIS_OUTPUT_DIR", f"./analysis/{run_id}"))
    output_dir.mkdir(parents=True, exist_ok=True)

    h = train_gpt.Hyperparameters()
    h.run_id = run_id
    h.analysis_output_dir = str(output_dir)
    h.logfile = None
    h.rank = 0
    h.world_size = 1
    h.local_rank = 0
    h.distributed = False
    h.is_main_process = True
    train_gpt.set_logging_hparams(h)

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    val_data = train_gpt.ValidationData(h, device)
    if train_gpt.BOS_ID is None:
        train_gpt.BOS_ID = 1
    _docs = train_gpt._find_docs(val_data.val_tokens)
    val_data.doc_start_indices = torch.tensor(
        [s for s, _ in _docs], dtype=torch.int64
    )
    val_data.doc_lengths = torch.tensor(
        [l for _, l in _docs], dtype=torch.int64
    )
    eval_model = train_gpt.deserialize(h, device)
    if h.num_loops > 0:
        eval_model.looping_active = True
    eval_model.eval()
    logits_fn = torch.compile(eval_model.forward_logits, dynamic=False, fullgraph=True)

    seq_len = h.eval_seq_len
    stride = h.eval_stride if 0 < h.eval_stride < seq_len else seq_len
    context_size = max(seq_len - stride, 0)
    batch_seqs = int(os.environ.get("DOC_AWARE_ANALYSIS_BATCH_SEQS", "32"))

    accum = {
        group_name: {
            label: {"loss_sum": 0.0, "token_count": 0.0, "byte_count": 0.0}
            for label, _, _, _ in specs
        }
        for group_name, specs in SLICE_GROUPS.items()
    }

    pending: list[tuple[int, int, int, int, int]] = []

    def flush() -> None:
        nonlocal pending
        if not pending:
            return
        bsz = len(pending)
        x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
        y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
        score_from = torch.zeros(bsz, dtype=torch.int64, device=device)
        score_to = torch.zeros(bsz, dtype=torch.int64, device=device)
        window_start_pos = torch.zeros(bsz, dtype=torch.int64, device=device)
        doc_scored_len = torch.zeros(bsz, dtype=torch.int64, device=device)
        for i, (token_start, wlen, s, ws, doc_idx) in enumerate(pending):
            chunk = val_data.val_tokens[token_start:token_start + wlen + 1].to(
                device=device, dtype=torch.int64, non_blocking=True
            )
            x_batch[i, :wlen] = chunk[:-1]
            y_batch[i, :wlen] = chunk[1:]
            score_from[i] = s
            score_to[i] = wlen
            window_start_pos[i] = ws
            doc_scored_len[i] = int(val_data.doc_lengths[doc_idx].item()) - 1

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = logits_fn(x_batch)
        nll = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            y_batch.reshape(-1),
            reduction="none",
        ).reshape(bsz, seq_len)

        for i, (_, _, _, ws, _) in enumerate(pending):
            s = int(score_from[i].item())
            e = int(score_to[i].item())
            if e <= s:
                continue
            scored_nll = nll[i, s:e].to(torch.float64)
            tgt = y_batch[i, s:e]
            prev = x_batch[i, s:e]
            tb = val_data.base_bytes_lut[tgt].to(torch.float64)
            tb += (val_data.has_leading_space_lut[tgt] &
                   ~val_data.is_boundary_token_lut[prev]).to(torch.float64)
            positions = torch.arange(ws + s + 1, ws + e + 1, device=device, dtype=torch.int64)

            scored_len = int(doc_scored_len[i].item())
            for group_name, specs in SLICE_GROUPS.items():
                for label, min_doc_scored_len, lower, upper in specs:
                    if scored_len < min_doc_scored_len:
                        continue
                    mask = bucket_mask(positions, lower, upper)
                    if bool(mask.any().item()):
                        accum[group_name][label]["loss_sum"] += float(scored_nll[mask].sum().item())
                        accum[group_name][label]["token_count"] += float(mask.sum().item())
                        accum[group_name][label]["byte_count"] += float(tb[mask].sum().item())
        pending = []
    import time as _time
    _log_every = int(os.environ.get("ANALYSIS_LOG_EVERY_DOCS", "1000"))
    _t_start = _time.perf_counter()
    with torch.inference_mode():
        total_docs = int(val_data.doc_lengths.numel())
        for doc_idx in range(total_docs):
            if _log_every > 0 and doc_idx > 0 and doc_idx % _log_every == 0:
                _elapsed = _time.perf_counter() - _t_start
                _eta = _elapsed * (total_docs - doc_idx) / doc_idx
                _total_tokens = sum(
                    v["token_count"]
                    for group in accum.values()
                    for v in group.values()
                )
                _total_loss = sum(
                    v["loss_sum"]
                    for group in accum.values()
                    for v in group.values()
                )
                _run_loss = _total_loss / _total_tokens if _total_tokens > 0 else 0.0
                train_gpt.log(
                    f"analysis_progress: doc {doc_idx}/{total_docs} "
                    f"elapsed:{_elapsed:.0f}s eta:{_eta:.0f}s "
                    f"running_loss(any-bucket):{_run_loss:.4f}"
                )
            doc_len = int(val_data.doc_lengths[doc_idx].item())
            scored_len = doc_len - 1
            if scored_len <= 0:
                continue
            doc_token_start = int(val_data.doc_start_indices[doc_idx].item())

            pending.append((doc_token_start, min(seq_len, scored_len), 0, 0, doc_idx))
            if len(pending) >= batch_seqs:
                flush()

            if scored_len <= seq_len or stride >= seq_len:
                continue

            for ws in range(stride, scored_len, stride):
                if ws + context_size >= scored_len:
                    break
                wlen = min(seq_len, scored_len - ws)
                s = min(context_size, wlen)
                pending.append((doc_token_start + ws, wlen, s, ws, doc_idx))
                if len(pending) >= batch_seqs:
                    flush()
        flush()

    result: dict[str, object] = {}
    for group_name, rows in accum.items():
        out_rows = []
        total_tokens = sum(v["token_count"] for v in rows.values())
        for label, stats in rows.items():
            tokens = stats["token_count"]
            bytes_ = stats["byte_count"]
            loss_sum = stats["loss_sum"]
            out_rows.append({
                "label": label,
                "token_count": int(tokens),
                "token_fraction_within_group": (tokens / total_tokens) if total_tokens > 0 else 0.0,
                "val_loss": (loss_sum / tokens) if tokens > 0 else float("nan"),
                "val_bpb": ((loss_sum / math.log(2.0)) / bytes_) if bytes_ > 0 else float("nan"),
                "loss_sum": loss_sum,
                "byte_count": bytes_,
            })
        result[group_name] = out_rows

    with (output_dir / "deeper_doc_eval.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")

    for group_name, rows in result.items():
        with (output_dir / f"{group_name}.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    for group_name, rows in result.items():
        train_gpt.log(group_name)
        train_gpt.log("bucket | tokens | tok_% | loss | bpb")
        train_gpt.log("-------+--------+-------+------+------")
        for row in rows:
            train_gpt.log(
                f"{row['label']} | {row['token_count']} | "
                f"{100.0 * row['token_fraction_within_group']:.2f} | "
                f"{row['val_loss']:.4f} | {row['val_bpb']:.4f}"
            )


if __name__ == "__main__":
    main()
