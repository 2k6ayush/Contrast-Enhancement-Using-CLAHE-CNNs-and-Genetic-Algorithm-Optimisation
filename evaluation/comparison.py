"""
evaluation/comparison.py — Method Comparison and Analysis
==========================================================
Member 3 owns this file.

OVERVIEW
--------
Loads the experiment results from results/*.csv and performs statistical
synthesis, generates publication-ready Markdown tables, and produces
high-resolution comparative visualizations saved to results/figures/.

INPUT CSVs (from results/):
---------------------------
  - results/comparison_table.csv    : Aggregated metrics across enhancement paradigms
  - results/ablation.csv            : Per-image metrics for Ablation Configs A, B, C, D
  - results/hyperparameter_tuning.csv: Fixed CLAHE and GA sweeps on validation split
  - results/metrics.csv             : Per-image metrics for Experiments 1–3

OUTPUT VISUALIZATIONS (saved to results/figures/):
--------------------------------------------------
  1. results/figures/comparison_metrics_barchart.png
     - Side-by-side grouped bar chart with error bars (PSNR, SSIM, Entropy, CII, EPI)
  2. results/figures/ablation_boxplots.png
     - Box plots showing SSIM and PSNR distributions across test images per ablation stage
  3. results/figures/clahe_tuning_heatmap.png
     - 2D Heatmap of Validation SSIM across Clip Limits vs Tile Grid Sizes
  4. results/figures/runtime_tradeoff.png
     - Quality (SSIM) vs Latency (time in ms) Pareto trade-off plot

USAGE:
------
    python evaluation/comparison.py
    python evaluation/comparison.py --results-dir results/ --figures-dir results/figures/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ===========================================================================
# DATA LOADING
# ===========================================================================

def load_results(results_dir: Path) -> dict[str, Optional[pd.DataFrame]]:
    """
    Load all available experiment result CSV files from *results_dir*.

    Parameters
    ----------
    results_dir : Path — directory containing the CSV files

    Returns
    -------
    dict mapping filename stem to DataFrame (or None if file is missing)
    """
    files = {
        "comparison": results_dir / "comparison_table.csv",
        "ablation": results_dir / "ablation.csv",
        "hyperparameter_tuning": results_dir / "hyperparameter_tuning.csv",
        "metrics": results_dir / "metrics.csv",
    }

    data: dict[str, Optional[pd.DataFrame]] = {}
    for key, path in files.items():
        if path.exists():
            try:
                data[key] = pd.read_csv(path)
            except Exception as exc:
                print(f"[WARN] Error reading {path.name}: {exc}")
                data[key] = None
        else:
            data[key] = None

    return data


# ===========================================================================
# MARKDOWN REPORT GENERATORS
# ===========================================================================

def generate_comparison_markdown(df_cmp: pd.DataFrame) -> str:
    """
    Generate a clean GitHub-flavored Markdown table from comparison_table.csv.

    Parameters
    ----------
    df_cmp : pd.DataFrame

    Returns
    -------
    str — formatted markdown
    """
    lines = [
        "### 📊 Overall Method Performance Comparison",
        "",
        "| Paradigm | Method | PSNR (dB) | SSIM | Entropy (bits) | CII | EPI | Time (ms) |",
        "|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for _, row in df_cmp.iterrows():
        approach = str(row.get("approach", ""))
        method = str(row.get("method", ""))

        def _fmt(val_m, val_s, prec=2):
            if pd.isna(val_m):
                return "N/A"
            if pd.isna(val_s):
                return f"{val_m:.{prec}f}"
            return f"{val_m:.{prec}f} ± {val_s:.{prec}f}"

        psnr_s = _fmt(row.get("psnr_mean"), row.get("psnr_std"), prec=2)
        ssim_s = _fmt(row.get("ssim_mean"), row.get("ssim_std"), prec=4)
        ent_s = _fmt(row.get("entropy_mean"), row.get("entropy_std"), prec=3)
        cii_s = _fmt(row.get("cii_mean"), row.get("cii_std"), prec=2)
        epi_s = _fmt(row.get("epi_mean"), row.get("epi_std"), prec=4)
        t_s = _fmt(row.get("time_ms_mean"), row.get("time_ms_std"), prec=1)

        lines.append(
            f"| {approach} | **{method}** | {psnr_s} | {ssim_s} | {ent_s} | {cii_s} | {epi_s} | {t_s} |"
        )

    return "\n".join(lines)


def generate_ablation_markdown(df_abl: pd.DataFrame) -> str:
    """
    Generate an ablation study summary Markdown table from ablation.csv.

    Parameters
    ----------
    df_abl : pd.DataFrame

    Returns
    -------
    str — formatted markdown
    """
    lines = [
        "### 🔬 Component Ablation Study (LOL Test Split: 15 images)",
        "",
        "| Config | Component Pipeline | PSNR (dB) | SSIM | Entropy (bits) | CII | EPI | Time (ms) |",
        "|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    grouped = df_abl.groupby(["config", "config_label"])
    for (cfg, label), group in grouped:
        psnr_m, psnr_s = group["psnr"].mean(), group["psnr"].std()
        ssim_m, ssim_s = group["ssim"].mean(), group["ssim"].std()
        ent_m, ent_s = group["entropy"].mean(), group["entropy"].std()
        cii_m, cii_s = group["contrast_improvement_index"].mean(), group["contrast_improvement_index"].std()
        epi_m, epi_s = group["edge_preservation"].mean(), group["edge_preservation"].std()
        t_m, t_s = group["time_ms"].mean(), group["time_ms"].std()

        lines.append(
            f"| **{cfg}** | {label} | "
            f"{psnr_m:.2f} ± {psnr_s:.2f} | "
            f"{ssim_m:.4f} ± {ssim_s:.4f} | "
            f"{ent_m:.3f} ± {ent_s:.3f} | "
            f"{cii_m:.2f} ± {cii_s:.2f} | "
            f"{epi_m:.4f} ± {epi_s:.4f} | "
            f"{t_m:.1f} ± {t_s:.1f} |"
        )

    return "\n".join(lines)


# ===========================================================================
# VISUALIZATION / PLOTTING
# ===========================================================================

def plot_comparison_metrics(df_cmp: pd.DataFrame, out_path: Path) -> None:
    """
    Create a multi-metric bar chart comparing approaches with error bars.

    Parameters
    ----------
    df_cmp   : pd.DataFrame — from comparison_table.csv
    out_path : Path — destination PNG path
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=150)

    methods = df_cmp["method"].tolist()
    x = np.arange(len(methods))
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    # 1. PSNR
    ax1 = axes[0]
    psnr_m = df_cmp["psnr_mean"].values
    psnr_s = df_cmp["psnr_std"].fillna(0).values
    bars1 = ax1.bar(x, psnr_m, yerr=psnr_s, capsize=5, color=colors[:len(methods)], alpha=0.85, edgecolor="black")
    ax1.set_title("Peak Signal-to-Noise Ratio (PSNR)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("PSNR (dB) — Higher is better", fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=15, ha="right", fontsize=9)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, yval / 2.0, f"{yval:.2f}", ha="center", va="center", color="white", fontweight="bold", fontsize=9)

    # 2. SSIM
    ax2 = axes[1]
    ssim_m = df_cmp["ssim_mean"].values
    ssim_s = df_cmp["ssim_std"].fillna(0).values
    bars2 = ax2.bar(x, ssim_m, yerr=ssim_s, capsize=5, color=colors[:len(methods)], alpha=0.85, edgecolor="black")
    ax2.set_title("Structural Similarity Index (SSIM)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("SSIM [0, 1] — Higher is better", fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=15, ha="right", fontsize=9)
    ax2.set_ylim(0, 0.70)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, yval / 2.0, f"{yval:.4f}", ha="center", va="center", color="white", fontweight="bold", fontsize=9)

    # 3. Entropy
    ax3 = axes[2]
    ent_m = df_cmp["entropy_mean"].values
    ent_s = df_cmp["entropy_std"].fillna(0).values
    bars3 = ax3.bar(x, ent_m, yerr=ent_s, capsize=5, color=colors[:len(methods)], alpha=0.85, edgecolor="black")
    ax3.set_title("Shannon Entropy (Information Richness)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Entropy (bits) — Higher is better", fontsize=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods, rotation=15, ha="right", fontsize=9)
    ax3.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars3:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2.0, yval / 2.0, f"{yval:.2f}", ha="center", va="center", color="white", fontweight="bold", fontsize=9)

    fig.suptitle("Performance Comparison Across Enhancement Methods on LOL Test Split", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] Comparison bar chart → {out_path}")


def plot_ablation_distributions(df_abl: pd.DataFrame, out_path: Path) -> None:
    """
    Generate box plots showing distribution of SSIM and PSNR for each ablation configuration.

    Parameters
    ----------
    df_abl   : pd.DataFrame — from ablation.csv
    out_path : Path — destination PNG path
    """
    configs = sorted(df_abl["config"].unique())
    labels = [
        f"{cfg}: {df_abl[df_abl['config'] == cfg]['config_label'].iloc[0]}"
        for cfg in configs
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    # SSIM Boxplot
    ssim_data = [df_abl[df_abl["config"] == cfg]["ssim"].values for cfg in configs]
    bplot1 = ax1.boxplot(ssim_data, patch_artist=True, tick_labels=labels)
    colors = ["#E69F00", "#56B4E9", "#009E73", "#D55E00"]
    for patch, color in zip(bplot1["boxes"], colors[:len(configs)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax1.set_title("SSIM Distribution Across Test Images", fontsize=11, fontweight="bold")
    ax1.set_ylabel("SSIM", fontsize=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.tick_params(axis="x", rotation=15)

    # PSNR Boxplot
    psnr_data = [df_abl[df_abl["config"] == cfg]["psnr"].values for cfg in configs]
    bplot2 = ax2.boxplot(psnr_data, patch_artist=True, tick_labels=labels)
    for patch, color in zip(bplot2["boxes"], colors[:len(configs)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_title("PSNR Distribution Across Test Images", fontsize=11, fontweight="bold")
    ax2.set_ylabel("PSNR (dB)", fontsize=10)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.tick_params(axis="x", rotation=15)

    fig.suptitle("Ablation Study: Component Impact on LOL Test Images", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] Ablation boxplots → {out_path}")


def plot_clahe_heatmap(df_tune: pd.DataFrame, out_path: Path) -> None:
    """
    Generate a 2D parameter heatmap of the fixed CLAHE sweep (clip_limit vs tile_grid_size).

    Parameters
    ----------
    df_tune  : pd.DataFrame — from hyperparameter_tuning.csv
    out_path : Path — destination PNG path
    """
    clahe_data = df_tune[df_tune["method"] == "clahe_fixed"]
    if clahe_data.empty:
        print("  [SKIP] No fixed CLAHE sweep data found in tuning CSV.")
        return

    pivot = clahe_data.pivot(index="clip_limit", columns="grid_size", values="ssim_mean")
    pivot = pivot.sort_index(ascending=False)  # High clip limits at top

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels([f"{int(c)}x{int(c)}" for c in pivot.columns])
    ax.set_yticklabels([f"{r:.1f}" for r in pivot.index])

    ax.set_xlabel("Tile Grid Size", fontsize=11, fontweight="bold")
    ax.set_ylabel("Clip Limit", fontsize=11, fontweight="bold")
    ax.set_title("Fixed CLAHE Validation SSIM Landscape\n(Clip Limit vs Tile Grid Size)", fontsize=12, fontweight="bold")

    # Add text annotations on each cell
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            text_color = "black" if val > 0.45 else "white"
            ax.text(j, i, f"{val:.4f}", ha="center", va="center", color=text_color, fontweight="bold", fontsize=9)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Validation Mean SSIM", fontsize=10)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] CLAHE parameter heatmap → {out_path}")


def plot_runtime_tradeoff(df_cmp: pd.DataFrame, out_path: Path) -> None:
    """
    Create a Quality vs Latency (SSIM vs Time) trade-off scatter plot.

    Parameters
    ----------
    df_cmp   : pd.DataFrame — from comparison_table.csv
    out_path : Path — destination PNG path
    """
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    colors = {"Traditional": "#4C72B0", "ML/DL": "#55A868", "Proposed": "#C44E52"}

    for _, row in df_cmp.iterrows():
        t = row["time_ms_mean"]
        s = row["ssim_mean"]
        app = row.get("approach", "Other")
        m = row.get("method", "")
        c = colors.get(app, "#333333")

        ax.scatter(t, s, color=c, s=120, edgecolors="black", zorder=5)
        # Offset annotation
        offset_x = 1.15
        ax.annotate(
            f"{m}\n({t:.1f} ms, SSIM {s:.4f})",
            (t, s),
            textcoords="offset points",
            xytext=(10, -5 if "Proposed" not in m else 10),
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=c, alpha=0.15, edgecolor=c),
        )

    ax.set_xscale("log")
    ax.set_xlabel("Execution Time per Image (ms, log scale)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean SSIM (Structural Quality)", fontsize=11, fontweight="bold")
    ax.set_title("Enhancement Quality vs. Latency Trade-off", fontsize=12, fontweight="bold")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] Runtime trade-off plot → {out_path}")


# ===========================================================================
# MAIN ENTRYPOINT
# ===========================================================================

def compare_methods(
    results_dir: Optional[Path] = None,
    figures_dir: Optional[Path] = None,
) -> None:
    """
    Synthesize all experiment results, display Markdown tables, and render figures.

    Parameters
    ----------
    results_dir : Path | None — directory with results CSVs (default: results/)
    figures_dir : Path | None — output directory for plots (default: results/figures/)
    """
    if results_dir is None:
        results_dir = PROJECT_ROOT / "results"
    if figures_dir is None:
        figures_dir = results_dir / "figures"

    print("=" * 65)
    print("  IMAGE ENHANCEMENT: EXPERIMENT COMPARISON & SYNTHESIS")
    print("=" * 65)
    print(f"  Reading CSVs from    : {results_dir}")
    print(f"  Generating plots to : {figures_dir}\n")

    results = load_results(results_dir)

    # 1. Comparison Table
    df_cmp = results.get("comparison")
    if df_cmp is not None and not df_cmp.empty:
        print("\n" + generate_comparison_markdown(df_cmp))
        plot_comparison_metrics(df_cmp, figures_dir / "comparison_metrics_barchart.png")
        plot_runtime_tradeoff(df_cmp, figures_dir / "runtime_tradeoff.png")
    else:
        print("[INFO] comparison_table.csv not found or empty.")

    # 2. Ablation Study
    df_abl = results.get("ablation")
    if df_abl is not None and not df_abl.empty:
        print("\n" + generate_ablation_markdown(df_abl))
        plot_ablation_distributions(df_abl, figures_dir / "ablation_boxplots.png")
    else:
        print("[INFO] ablation.csv not found or empty.")

    # 3. Hyperparameter Tuning
    df_tune = results.get("hyperparameter_tuning")
    if df_tune is not None and not df_tune.empty:
        plot_clahe_heatmap(df_tune, figures_dir / "clahe_tuning_heatmap.png")
    else:
        print("[INFO] hyperparameter_tuning.csv not found or empty.")

    print("\n" + "=" * 65)
    print("  [COMPLETE] Comparison tables and figures generated.")
    print("=" * 65)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate method comparison tables and figures.")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--figures-dir", type=Path, default=PROJECT_ROOT / "results" / "figures")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compare_methods(args.results_dir, args.figures_dir)
