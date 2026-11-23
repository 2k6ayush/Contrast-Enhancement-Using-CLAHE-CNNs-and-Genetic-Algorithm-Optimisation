"""Experiment 1 runner: Histogram Equalization and CLAHE baseline on LOL.

Usage (from the repository root):
    python experiment1_traditional.py --root dataset/LOL --split eval15
    python experiment1_traditional.py --niqe          # also compute NIQE (needs pyiqa)

Outputs:
    results/traditional_metrics.csv       per-image metrics
    results/traditional_summary.csv       mean per method (goes in the results table)
    results/figures/traditional_*.png     visual comparisons and error maps
    results/traditional_outputs/<method>/ enhanced images
"""
import argparse
import os
import time

import pandas as pd

from dataset import load_lol_pairs, save_image
from traditional import apply_clahe, apply_histogram_equalization
from traditional.metrics import evaluate_all
from visualization import show_comparison, show_error_maps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset/LOL")
    ap.add_argument("--split", default="eval15")
    ap.add_argument("--clip", type=float, default=2.0)
    ap.add_argument("--grid", type=int, default=8)
    ap.add_argument("--niqe", action="store_true", help="also compute NIQE")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--figures", type=int, default=5, help="how many sample figures to save")
    args = ap.parse_args()

    pairs = load_lol_pairs(args.root, args.split, limit=args.limit)
    methods = {
        "Low-light (no enhancement)": lambda im: im,
        "HE": apply_histogram_equalization,
        f"CLAHE (clip={args.clip}, grid={args.grid})": lambda im: apply_clahe(im, args.clip, args.grid),
    }

    rows = []
    for idx, (name, low, high) in enumerate(pairs):
        outputs = {}
        for method, fn in methods.items():
            start = time.perf_counter()
            out = fn(low)
            elapsed_ms = (time.perf_counter() - start) * 1000
            outputs[method] = out
            row = {"image": name, "method": method, "Time (ms)": elapsed_ms}
            row.update(evaluate_all(low, out, high, include_niqe=args.niqe))
            rows.append(row)
            if method != "Low-light (no enhancement)":
                save_image(os.path.join("results", "traditional_outputs", method.split(" ")[0], name), out)
        if idx < args.figures:
            names = list(outputs)
            show_comparison(
                [low, outputs[names[1]], outputs[names[2]], high],
                ["Low-light", "HE", "CLAHE", "Ground truth"],
                save_path=f"results/figures/traditional_{os.path.splitext(name)[0]}.png",
            )
            show_error_maps(
                {"HE": outputs[names[1]], "CLAHE": outputs[names[2]]},
                high,
                save_path=f"results/figures/traditional_error_{os.path.splitext(name)[0]}.png",
            )

    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/traditional_metrics.csv", index=False)
    summary = df.drop(columns="image").groupby("method").mean().round(4)
    summary.to_csv("results/traditional_summary.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main()
