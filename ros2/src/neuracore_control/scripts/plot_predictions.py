#!/usr/bin/env python3
"""Plot policy chunks from the inference node's predictions CSV.

CSV schema (one row per executed action):
    t, chunk_id, chunk_idx,
    obs_<arm joint x7>, obs_grip,
    out_<arm joint x7>, out_grip

`obs_*` is constant within a chunk (snapshot at predict time), `out_*` is
the per-step raw policy output. Plot per joint: input dot, then 100 output
points, then the next input dot, then the next 100 outputs, ...

Usage:
    python3 plot_predictions.py <predictions.csv> [--chunks N] [--start K]
                                                  [--out preds.png] [--show]
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load(path: Path):
    with path.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [[float(x) if x else float("nan") for x in r] for r in reader]
    if not rows:
        sys.exit(f"no data rows in {path}")
    return header, np.array(rows, dtype=np.float64)


def joint_columns(header):
    obs_cols = [(h.removeprefix("obs_"), i) for i, h in enumerate(header) if h.startswith("obs_")]
    out_cols = [(h.removeprefix("out_"), i) for i, h in enumerate(header) if h.startswith("out_")]
    obs_map = dict(obs_cols)
    out_map = dict(out_cols)
    names = [n for n, _ in obs_cols]
    if [n for n, _ in out_cols] != names:
        sys.exit(f"obs/out column names disagree: {[n for n,_ in obs_cols]} vs {[n for n,_ in out_cols]}")
    return names, obs_map, out_map


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", type=Path, help="predictions CSV from inference_node")
    ap.add_argument("--chunks", type=int, default=2, help="number of chunks to plot (default 2)")
    ap.add_argument("--start", type=int, default=0, help="first chunk_id to include (default 0)")
    ap.add_argument("--out", type=Path, default=None, help="output PNG (default <csv>.png)")
    ap.add_argument("--show", action="store_true", help="display interactively")
    args = ap.parse_args()

    header, data = load(args.csv)

    if any(h.startswith("target_") or h.startswith("current_") for h in header):
        sys.exit(
            f"{args.csv} uses the OLD CSV schema (target_*/current_*). "
            "The plotter now expects obs_*/out_* + chunk_id/chunk_idx — "
            "regenerate the CSV with the updated inference_node."
        )

    for required in ("chunk_id", "chunk_idx"):
        if required not in header:
            sys.exit(f"{args.csv} missing column '{required}'")

    names, obs_map, out_map = joint_columns(header)
    chunk_id_col = header.index("chunk_id")
    chunk_idx_col = header.index("chunk_idx")

    chunk_ids_present = np.unique(data[:, chunk_id_col]).astype(int)
    selected = chunk_ids_present[
        (chunk_ids_present >= args.start)
        & (chunk_ids_present < args.start + args.chunks)
    ]
    if len(selected) == 0:
        sys.exit(
            f"no chunks in range [{args.start}, {args.start + args.chunks}). "
            f"Available: {chunk_ids_present.min()}..{chunk_ids_present.max()}"
        )

    n = len(names)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(11, 2.4 * rows), sharex=True)
    axes = np.atleast_1d(axes).flatten()

    chunk_len = 100  # default x spacing per chunk; actual size taken from data

    for ax, name in zip(axes, names):
        obs_col = obs_map[name]
        out_col = out_map[name]
        for k, cid in enumerate(selected):
            mask = data[:, chunk_id_col] == cid
            chunk_rows = data[mask]
            order = np.argsort(chunk_rows[:, chunk_idx_col])
            chunk_rows = chunk_rows[order]
            n_steps = len(chunk_rows)

            x_in = k * (chunk_len + 1)
            x_out = np.arange(x_in + 1, x_in + 1 + n_steps)

            obs_value = chunk_rows[0, obs_col]
            out_values = chunk_rows[:, out_col]

            ax.scatter(
                [x_in], [obs_value],
                color="tab:blue", marker="o", s=40, zorder=3,
                label="input obs" if k == 0 else None,
            )
            ax.plot(
                x_out, out_values,
                color="tab:orange", linewidth=1.2,
                label="policy chunk output" if k == 0 else None,
            )
            ax.axvline(x_in, color="gray", alpha=0.25, linewidth=0.8)

        ax.set_title(name, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        ax.set_ylabel("[0,1]" if name.endswith("grip") else "rad", fontsize=8)

    for ax in axes[n:]:
        ax.axis("off")

    axes[0].legend(fontsize=8, loc="best")
    for ax in axes[-cols:]:
        ax.set_xlabel("step (input + 100 outputs per chunk)", fontsize=8)

    fig.suptitle(
        f"{args.csv.name} — chunks {selected[0]}..{selected[-1]} "
        f"({len(selected)} of {len(chunk_ids_present)} total)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out = args.out or args.csv.with_suffix(".png")
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
