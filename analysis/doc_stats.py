#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np
import sentencepiece as spm


SHARD_HEADER_INTS = 256
SHARD_MAGIC = 20240520
SHARD_VERSION = 1
THRESHOLDS = (1024, 2048, 4096, 8192)


def load_data_shard(path: Path) -> np.ndarray:
    header = np.fromfile(path, dtype="<i4", count=SHARD_HEADER_INTS)
    if header.size != SHARD_HEADER_INTS or int(header[0]) != SHARD_MAGIC or int(header[1]) != SHARD_VERSION:
        raise ValueError(f"unexpected shard header for {path}")
    num_tokens = int(header[2])
    offset = SHARD_HEADER_INTS * np.dtype("<i4").itemsize
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=offset)
    if tokens.size != num_tokens:
        raise ValueError(f"short read for {path}")
    return tokens


def load_tokens(pattern: str) -> np.ndarray:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"no files match {pattern}")
    arrays = [load_data_shard(path) for path in files]
    return arrays[0] if len(arrays) == 1 else np.concatenate(arrays)


def percentile(lengths: np.ndarray, q: float) -> int:
    return int(np.quantile(lengths, q, method="linear"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute exact validation document statistics from BOS-delimited shards.")
    parser.add_argument("--val-pattern", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    bos_id = int(sp.bos_id())
    if bos_id < 0:
        raise ValueError("tokenizer has no BOS id")

    tokens = load_tokens(args.val_pattern)
    doc_starts = np.flatnonzero(tokens == bos_id)
    if doc_starts.size == 0:
        raise ValueError("no BOS tokens found; cannot recover document boundaries")
    if int(doc_starts[0]) != 0:
        raise ValueError(f"expected first token to be BOS, got first BOS at index {int(doc_starts[0])}")

    doc_lengths = np.diff(np.append(doc_starts, tokens.size)).astype(np.int64, copy=False)
    if np.any(doc_lengths <= 0):
        raise ValueError("found non-positive document length")

    num_docs = int(doc_lengths.size)
    total_tokens = int(doc_lengths.sum())
    scored_tokens_doc_aware = int((doc_lengths - 1).sum())

    summary = {
        "num_docs": num_docs,
        "total_tokens_including_bos": total_tokens,
        "total_scored_tokens_doc_aware": scored_tokens_doc_aware,
        "mean_tokens": float(doc_lengths.mean()),
        "median_tokens": float(np.median(doc_lengths)),
        "p75_tokens": percentile(doc_lengths, 0.75),
        "p90_tokens": percentile(doc_lengths, 0.90),
        "p95_tokens": percentile(doc_lengths, 0.95),
        "p99_tokens": percentile(doc_lengths, 0.99),
        "max_tokens": int(doc_lengths.max()),
    }

    doc_fraction_rows: list[dict[str, float | int]] = []
    token_mass_rows: list[dict[str, float | int]] = []
    for threshold in THRESHOLDS:
        docs_gt = int((doc_lengths > threshold).sum())
        tokens_beyond = int(np.maximum(doc_lengths - (threshold + 1), 0).sum())
        doc_fraction_rows.append(
            {
                "threshold": threshold,
                "docs_longer_count": docs_gt,
                "docs_longer_fraction": docs_gt / num_docs,
            }
        )
        token_mass_rows.append(
            {
                "threshold": threshold,
                "tokens_beyond_count": tokens_beyond,
                "tokens_beyond_fraction_of_scored_doc_aware_tokens": tokens_beyond / scored_tokens_doc_aware,
            }
        )

    summary["doc_length_thresholds"] = doc_fraction_rows
    summary["within_doc_token_mass_thresholds"] = token_mass_rows

    with (out_dir / "val_doc_stats.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    with (out_dir / "val_doc_lengths.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["doc_idx", "start_token_idx", "doc_tokens_including_bos", "scored_tokens_doc_aware"])
        for doc_idx, (start, length) in enumerate(zip(doc_starts.tolist(), doc_lengths.tolist(), strict=True)):
            writer.writerow([doc_idx, start, length, length - 1])

    lines = [
        f"num_docs={num_docs}",
        f"total_tokens_including_bos={total_tokens}",
        f"total_scored_tokens_doc_aware={scored_tokens_doc_aware}",
        f"mean={summary['mean_tokens']:.6f}",
        f"median={summary['median_tokens']:.1f}",
        f"p75={summary['p75_tokens']}",
        f"p90={summary['p90_tokens']}",
        f"p95={summary['p95_tokens']}",
        f"p99={summary['p99_tokens']}",
        f"max={summary['max_tokens']}",
    ]
    for row in doc_fraction_rows:
        threshold = int(row["threshold"])
        lines.append(
            f"docs_gt_{threshold}={int(row['docs_longer_count'])} ({float(row['docs_longer_fraction']):.8f})"
        )
    for row in token_mass_rows:
        threshold = int(row["threshold"])
        lines.append(
            f"scored_tokens_pos_gt_{threshold}={int(row['tokens_beyond_count'])} "
            f"({float(row['tokens_beyond_fraction_of_scored_doc_aware_tokens']):.8f})"
        )
    (out_dir / "val_doc_stats.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
