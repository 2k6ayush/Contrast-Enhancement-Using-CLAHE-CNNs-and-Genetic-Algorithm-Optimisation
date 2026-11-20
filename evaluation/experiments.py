"""
evaluation/experiments.py — Experiment Runner
==============================================
Member 3 owns this file.

OVERVIEW
--------
Runs all three experiments comparing image enhancement methods on the LOL
dataset (low-light pairs). Results are written to results/metrics.csv.

EXPERIMENTS
-----------
  Exp 1 — Traditional: Histogram Equalisation and default CLAHE.
  Exp 2 — CNN only:    build_model() + enhance_with_cnn(). Skipped if the
                        CNN is a stub or weights are unavailable.
  Exp 3 — Hybrid GA:  enhance_hybrid(), run in two modes:
            (a) hybrid_ref   — reference-based GA fitness (uses ground truth)
            (b) hybrid_noref — no-reference GA fitness (no ground truth given)

DATASET
-------
  dataset/our485/low  + dataset/our485/high  → training + validation
  dataset/eval15/low  + dataset/eval15/high  → test set (never used for tuning)

  Validation split (15% of our485, fixed seed) is carved here but NOT used
  for tuning — it is available via --split val for future ablation runs.

USAGE
-----
  python evaluation/experiments.py --exp 1 --max-images 3
  python evaluation/experiments.py --exp all --split test
  python evaluation/experiments.py --exp 2 --model-path models/cnn_weights.pth
  python evaluation/experiments.py --exp 3 --max-images 10 --split val

SAFE RE-RUN
-----------
  Results are written to results/metrics.csv.  On re-run, any existing rows
  for the same (experiment, method) pair are replaced, so you can re-run a
  single experiment without duplicating results.

METRICS (per image)
-------------------
  PSNR                      — Peak Signal-to-Noise Ratio (dB), reference-based
  SSIM                      — Structural Similarity Index [0,1], reference-based
  entropy                   — Shannon entropy of grayscale histogram (bits)
  NIQE                      — Natural Image Quality Evaluator (NaN if stub)
  contrast_improvement_index— std(enhanced) / std(original), > 1 means more contrast
  edge_preservation         — Pearson corr of Sobel gradients vs ground truth
  time_ms                   — wall-clock time for the enhancement (ms)
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import warnings
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

# Add project root to path so this script can be run from any directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── try importing tqdm; fall back gracefully if not installed ────────────────
try:
    from tqdm import tqdm as _tqdm
    def tqdm(iterable, **kwargs):
        return _tqdm(iterable, **kwargs)
except ImportError:
    def tqdm(iterable, desc="", **kw):
        print(f"[{desc}] running (tqdm not installed, no progress bar)…")
        return iterable

# ── metric and pipeline imports from our compat layer ───────────────────────
from optimization.compat import (
    apply_histogram_equalization,
    apply_clahe,
    calculate_psnr,
    calculate_ssim,
    calculate_entropy,
    calculate_niqe,
    calculate_edge_preservation,
    build_model,
    enhance_with_cnn,
    using_stubs,
)
from optimization.hybrid import enhance_hybrid


# ===========================================================================
# CONSTANTS
# ===========================================================================

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
RESULTS_DIR      = PROJECT_ROOT / "results"
METRICS_CSV      = RESULTS_DIR / "metrics.csv"

# Column order for the CSV
CSV_COLUMNS = [
    "experiment",
    "method",
    "image",
    "psnr",
    "ssim",
    "entropy",
    "niqe",
    "contrast_improvement_index",
    "edge_preservation",
    "time_ms",
]


# ===========================================================================
# DATASET LOADING
# ===========================================================================

def _list_images(folder: Path) -> list[str]:
    """
    Return a sorted list of image filenames in *folder*.

    Rules:
      - Only files with extensions in IMAGE_EXTENSIONS (.png, .jpg, .jpeg)
      - Skip any file whose name starts with "." (macOS junk like ._file.png)

    Parameters
    ----------
    folder : Path — directory to scan

    Returns
    -------
    list[str] — sorted filenames (names only, not full paths)

    Raises
    ------
    FileNotFoundError — if the folder does not exist
    RuntimeError      — if the folder exists but contains no valid images
    """
    if not folder.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {folder}\n"
            "Make sure you have downloaded the LOL dataset and placed it at "
            f"'{folder.parent.parent}'. Never generate synthetic data as a fallback."
        )

    files = sorted([
        f.name for f in folder.iterdir()
        if f.is_file()
        and not f.name.startswith(".")
        and f.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if not files:
        raise RuntimeError(
            f"No valid images found in {folder}. "
            "Expected .png/.jpg/.jpeg files (skipping hidden/dot files)."
        )

    return files


def load_paired_split(
    data_root: Path,
    subset: str,
    max_images: Optional[int] = None,
) -> tuple[list[Path], list[Path]]:
    """
    Load paired (low, high) image paths from a dataset subset.

    Parameters
    ----------
    data_root  : Path         — root of the dataset (e.g. Path("dataset"))
    subset     : str          — sub-folder name, e.g. "our485" or "eval15"
    max_images : int | None   — if set, return only the first N pairs

    Returns
    -------
    low_paths, high_paths : list[Path], list[Path]
        Corresponding low-light and normal-light image paths, same order.

    Raises
    ------
    FileNotFoundError — if either low/ or high/ folder is missing
    RuntimeError      — if filenames don't match between low and high
    """
    low_dir  = data_root / subset / "low"
    high_dir = data_root / subset / "high"

    low_names  = _list_images(low_dir)
    high_names = _list_images(high_dir)

    # Stem-based matching (tolerates different extensions for the same image)
    low_stems  = {Path(n).stem: n for n in low_names}
    high_stems = {Path(n).stem: n for n in high_names}

    common = sorted(set(low_stems) & set(high_stems))
    only_low  = set(low_stems) - set(high_stems)
    only_high = set(high_stems) - set(low_stems)

    if only_low or only_high:
        msg = f"Filename mismatch in {subset}:"
        if only_low:
            msg += f"\n  In low/ but NOT high/: {sorted(only_low)[:10]}"
        if only_high:
            msg += f"\n  In high/ but NOT low/: {sorted(only_high)[:10]}"
        raise RuntimeError(msg)

    low_paths  = [low_dir  / low_stems[s]  for s in common]
    high_paths = [high_dir / high_stems[s] for s in common]

    if max_images is not None:
        low_paths  = low_paths[:max_images]
        high_paths = high_paths[:max_images]

    return low_paths, high_paths


def make_val_split(
    low_paths: list[Path],
    high_paths: list[Path],
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[tuple[list[Path], list[Path]], tuple[list[Path], list[Path]]]:
    """
    Carve a fixed validation split from our485.

    Uses a fixed seed so the split is always the same regardless of run order.
    The validation images are NEVER used for GA tuning — they are reserved for
    ablation / hyper-parameter selection after training.

    Parameters
    ----------
    low_paths, high_paths : list[Path] — the full our485 paired paths
    val_fraction          : float      — fraction to use as validation (default 0.15)
    seed                  : int        — random seed (default 42)

    Returns
    -------
    (train_low, train_high), (val_low, val_high)
    """
    n     = len(low_paths)
    n_val = max(1, int(n * val_fraction))

    rng     = np.random.default_rng(seed=seed)
    indices = rng.permutation(n)

    val_idx   = sorted(indices[:n_val].tolist())
    train_idx = sorted(indices[n_val:].tolist())

    train_low  = [low_paths[i]  for i in train_idx]
    train_high = [high_paths[i] for i in train_idx]
    val_low    = [low_paths[i]  for i in val_idx]
    val_high   = [high_paths[i] for i in val_idx]

    return (train_low, train_high), (val_low, val_high)


def load_image(path: Path) -> np.ndarray:
    """
    Load an image from disk as RGB uint8.

    Parameters
    ----------
    path : Path — image file path

    Returns
    -------
    np.ndarray — RGB uint8 HxWx3

    Raises
    ------
    RuntimeError — if cv2 fails to read the file
    """
    raw = cv2.imread(str(path))
    if raw is None:
        raise RuntimeError(f"cv2.imread failed to read: {path}")
    return cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)


# ===========================================================================
# METRIC EVALUATION INFRASTRUCTURE
# ===========================================================================

def evaluate_method(
    method_fn: Callable[[np.ndarray], np.ndarray],
    low_paths: list[Path],
    gt_paths:  list[Path],
    desc: str = "evaluating",
) -> list[dict]:
    """
    Run *method_fn* on each low-light image, compare to ground truth,
    and return a list of per-image metric dicts.

    Parameters
    ----------
    method_fn  : Callable[[np.ndarray], np.ndarray]
                 Takes a low-light RGB uint8 image, returns enhanced RGB uint8.
    low_paths  : list[Path]  — low-light image paths
    gt_paths   : list[Path]  — paired ground-truth (normal-light) image paths
    desc       : str         — label for the tqdm progress bar

    Returns
    -------
    list[dict] — one dict per image with keys:
        image, psnr, ssim, entropy, niqe, contrast_improvement_index,
        edge_preservation, time_ms
    """
    rows = []

    for low_p, gt_p in tqdm(zip(low_paths, gt_paths), total=len(low_paths), desc=desc):
        low_img = load_image(low_p)
        gt_img  = load_image(gt_p)

        # ── Run the enhancement method and time it ───────────────────────────
        t0  = time.perf_counter()
        out = method_fn(low_img)
        t1  = time.perf_counter()
        time_ms = (t1 - t0) * 1000.0

        # Defensive: ensure output is uint8
        if out.dtype != np.uint8:
            out = np.clip(out, 0, 255).astype(np.uint8)

        # ── Compute metrics ──────────────────────────────────────────────────
        psnr_val = calculate_psnr(out, gt_img)
        ssim_val = calculate_ssim(out, gt_img)
        entr_val = calculate_entropy(out)

        # NIQE may be a stub returning NaN — that's acceptable; we keep NaN.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")     # suppress stub warning in CSV loop
            niqe_val = calculate_niqe(out)

        # Contrast Improvement Index: std(enhanced) / std(original)
        # > 1.0 means enhanced image has more contrast than input
        std_orig = float(low_img.std()) + 1e-8   # avoid division by zero
        std_enh  = float(out.std())
        cii      = std_enh / std_orig

        # Edge preservation against ground truth
        epi_val = calculate_edge_preservation(out, gt_img)

        rows.append({
            "image":                    low_p.name,
            "psnr":                     round(float(psnr_val), 4),
            "ssim":                     round(float(ssim_val), 4),
            "entropy":                  round(float(entr_val), 4),
            "niqe":                     round(float(niqe_val), 4) if not np.isnan(niqe_val) else float("nan"),
            "contrast_improvement_index": round(cii, 4),
            "edge_preservation":        round(float(epi_val), 4),
            "time_ms":                  round(time_ms, 2),
        })

    return rows


# ===========================================================================
# CSV PERSISTENCE (safe re-run)
# ===========================================================================

def _load_csv(path: Path) -> list[dict]:
    """Load existing metrics CSV into a list of dicts. Returns [] if missing."""
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _save_csv(path: Path, rows: list[dict]) -> None:
    """Write all rows to CSV, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def upsert_results(
    csv_path: Path,
    experiment: str,
    method: str,
    new_rows: list[dict],
) -> None:
    """
    Insert or replace rows for (experiment, method) in the CSV.

    Safe to re-run: existing rows for the same experiment+method are removed
    before inserting the new ones, so there are no duplicates.

    Parameters
    ----------
    csv_path   : Path       — path to results/metrics.csv
    experiment : str        — experiment label (e.g. "exp1")
    method     : str        — method label (e.g. "histogram_eq")
    new_rows   : list[dict] — per-image metric dicts (without experiment/method keys)
    """
    existing = _load_csv(csv_path)

    # Drop any old rows for this experiment+method
    kept = [
        r for r in existing
        if not (r.get("experiment") == experiment and r.get("method") == method)
    ]

    # Inject experiment and method keys into each new row
    tagged = []
    for row in new_rows:
        full = {"experiment": experiment, "method": method}
        full.update(row)
        # Ensure all CSV columns present (fill missing with empty string)
        for col in CSV_COLUMNS:
            full.setdefault(col, "")
        tagged.append({col: full[col] for col in CSV_COLUMNS})

    _save_csv(csv_path, kept + tagged)
    print(f"  → Saved {len(tagged)} rows to {csv_path}  "
          f"(experiment={experiment}, method={method})")


# ===========================================================================
# EXPERIMENT 1 — Traditional Methods
# ===========================================================================

def run_experiment_1(
    low_paths: list[Path],
    gt_paths:  list[Path],
    split_label: str,
) -> None:
    """
    Experiment 1: Histogram Equalisation and default CLAHE.

    Both methods come from optimization/compat.py and may be stubs —
    stubs print their own warnings, but we always run and record results.

    Parameters
    ----------
    low_paths   : list[Path] — low-light image paths
    gt_paths    : list[Path] — ground-truth image paths
    split_label : str        — "val" or "test", used in experiment label
    """
    print("\n" + "=" * 60)
    print(f"  EXPERIMENT 1 — Traditional Methods  [{split_label}]")
    print("=" * 60)

    experiment_tag = f"exp1_{split_label}"

    # ── Histogram Equalisation ───────────────────────────────────────────────
    print("\n[1a] Histogram Equalisation")
    he_rows = evaluate_method(
        method_fn=lambda img: apply_histogram_equalization(img),
        low_paths=low_paths,
        gt_paths=gt_paths,
        desc="HE",
    )
    upsert_results(METRICS_CSV, experiment_tag, "histogram_eq", he_rows)

    # ── Default CLAHE (clip=2.0, grid=8) ────────────────────────────────────
    print("\n[1b] CLAHE (clip=2.0, grid=8)")
    clahe_rows = evaluate_method(
        method_fn=lambda img: apply_clahe(img, clip_limit=2.0, grid_size=8),
        low_paths=low_paths,
        gt_paths=gt_paths,
        desc="CLAHE",
    )
    upsert_results(METRICS_CSV, experiment_tag, "clahe_default", clahe_rows)

    # ── Summary ─────────────────────────────────────────────────────────────
    _print_summary("Histogram EQ", he_rows)
    _print_summary("CLAHE default", clahe_rows)


# ===========================================================================
# EXPERIMENT 2 — CNN Only
# ===========================================================================

def run_experiment_2(
    low_paths:   list[Path],
    gt_paths:    list[Path],
    split_label: str,
    model_path:  Optional[Path] = None,
) -> None:
    """
    Experiment 2: CNN-only enhancement.

    Skips entirely (with a clear warning) if:
      - The CNN module is a stub (build_model is not real), OR
      - No --model-path is provided, OR
      - The weights file does not exist.

    We must NOT run the stub CNN and record it as "real CNN results".

    Parameters
    ----------
    low_paths   : list[Path]        — low-light image paths
    gt_paths    : list[Path]        — ground-truth image paths
    split_label : str               — "val" or "test"
    model_path  : Path | None       — path to trained CNN weights (.pth / .pt)
    """
    print("\n" + "=" * 60)
    print(f"  EXPERIMENT 2 — CNN Only  [{split_label}]")
    print("=" * 60)

    experiment_tag = f"exp2_{split_label}"

    # ── Guard: refuse to run stub CNN as if it's real ───────────────────────
    stubs = using_stubs()
    if stubs.get("build_model", True) or stubs.get("enhance_with_cnn", True):
        print(
            "\n[SKIP] CNN module is not available (using stubs).\n"
            "       Experiment 2 will be run once Member 2 merges cnn/ into the branch.\n"
            "       Do NOT record stub CNN results as real CNN results."
        )
        return

    if model_path is None:
        print(
            "\n[SKIP] --model-path not provided. "
            "Cannot load CNN weights. Experiment 2 skipped."
        )
        return

    if not model_path.exists():
        print(
            f"\n[SKIP] Weights file not found: {model_path}\n"
            "       Experiment 2 skipped."
        )
        return

    # ── Load model ───────────────────────────────────────────────────────────
    print(f"\n[2] Loading CNN model from {model_path} …")
    try:
        import torch  # type: ignore
        model = build_model()
        state = torch.load(str(model_path), map_location="cpu")
        model.load_state_dict(state)
        model.eval()
    except Exception as exc:
        print(f"\n[SKIP] Failed to load CNN weights: {exc}\n"
              "       Experiment 2 skipped.")
        return

    # Model size in MB
    size_mb = model_path.stat().st_size / (1024 ** 2)
    print(f"  Model size: {size_mb:.2f} MB")

    # ── Evaluate ─────────────────────────────────────────────────────────────
    cnn_rows = evaluate_method(
        method_fn=lambda img: enhance_with_cnn(img, model),
        low_paths=low_paths,
        gt_paths=gt_paths,
        desc="CNN",
    )
    upsert_results(METRICS_CSV, experiment_tag, "cnn_only", cnn_rows)
    _print_summary("CNN only", cnn_rows)


# ===========================================================================
# EXPERIMENT 3 — Hybrid GA-CLAHE + CNN
# ===========================================================================

def run_experiment_3(
    low_paths:   list[Path],
    gt_paths:    list[Path],
    split_label: str,
    model_path:  Optional[Path] = None,
    ga_seed:     int = 42,
) -> None:
    """
    Experiment 3: Hybrid GA-CLAHE → CNN enhancement.

    Two runs:
      (a) hybrid_ref   — GA fitness uses ground truth (reference-based)
      (b) hybrid_noref — GA fitness ignores ground truth (no-reference)

    Both record per-image time including the full GA optimisation.

    Parameters
    ----------
    low_paths   : list[Path]        — low-light image paths
    gt_paths    : list[Path]        — ground-truth image paths
    split_label : str               — "val" or "test"
    model_path  : Path | None       — optional CNN weights for Stage 2
    ga_seed     : int               — random seed forwarded to the GA
    """
    print("\n" + "=" * 60)
    print(f"  EXPERIMENT 3 — Hybrid GA-CLAHE + CNN  [{split_label}]")
    print("=" * 60)

    experiment_tag = f"exp3_{split_label}"

    # ── Load CNN model if available ──────────────────────────────────────────
    model = None
    stubs = using_stubs()
    if (
        not stubs.get("build_model", True)
        and not stubs.get("enhance_with_cnn", True)
        and model_path is not None
        and model_path.exists()
    ):
        try:
            import torch  # type: ignore
            model = build_model()
            state = torch.load(str(model_path), map_location="cpu")
            model.load_state_dict(state)
            model.eval()
            print(f"  CNN loaded from {model_path}  "
                  f"({model_path.stat().st_size / 1024**2:.2f} MB)")
        except Exception as exc:
            print(f"  [WARN] Could not load CNN weights: {exc}. "
                  "Running hybrid with model=None (GA-CLAHE only).")
            model = None
    else:
        if stubs.get("build_model", True):
            print("  [INFO] CNN is a stub — running hybrid as GA-CLAHE only.")
        else:
            print("  [INFO] No --model-path provided — running hybrid as GA-CLAHE only.")

    # ── 3a: Reference-based GA fitness ──────────────────────────────────────
    print("\n[3a] Hybrid — reference-based GA fitness (ground truth used)")

    def _hybrid_ref(low_img: np.ndarray, gt_img: np.ndarray) -> np.ndarray:
        result = enhance_hybrid(
            low_img, model,
            ground_truth=gt_img,
            seed=ga_seed,
        )
        return result["output"]

    # evaluate_method takes method_fn(low) -> enhanced, so we wrap using closure
    ref_rows = _evaluate_paired(
        method_fn_paired=_hybrid_ref,
        low_paths=low_paths,
        gt_paths=gt_paths,
        desc="Hybrid-ref",
    )
    upsert_results(METRICS_CSV, experiment_tag, "hybrid_ref", ref_rows)

    # ── 3b: No-reference GA fitness ─────────────────────────────────────────
    print("\n[3b] Hybrid — no-reference GA fitness (ground truth NOT used in GA)")

    def _hybrid_noref(low_img: np.ndarray, _gt_img: np.ndarray) -> np.ndarray:
        result = enhance_hybrid(
            low_img, model,
            ground_truth=None,     # no GT → no-reference fitness
            seed=ga_seed,
        )
        return result["output"]

    noref_rows = _evaluate_paired(
        method_fn_paired=_hybrid_noref,
        low_paths=low_paths,
        gt_paths=gt_paths,
        desc="Hybrid-noref",
    )
    upsert_results(METRICS_CSV, experiment_tag, "hybrid_noref", noref_rows)

    _print_summary("Hybrid (ref fitness)",   ref_rows)
    _print_summary("Hybrid (noref fitness)", noref_rows)


def _evaluate_paired(
    method_fn_paired: Callable[[np.ndarray, np.ndarray], np.ndarray],
    low_paths: list[Path],
    gt_paths:  list[Path],
    desc: str = "evaluating",
) -> list[dict]:
    """
    Like evaluate_method() but the method function takes BOTH low and gt
    as arguments (needed for the reference-based hybrid variant).

    The evaluation metrics are still computed against ground truth.
    """
    rows = []

    for low_p, gt_p in tqdm(zip(low_paths, gt_paths), total=len(low_paths), desc=desc):
        low_img = load_image(low_p)
        gt_img  = load_image(gt_p)

        t0  = time.perf_counter()
        out = method_fn_paired(low_img, gt_img)
        t1  = time.perf_counter()
        time_ms = (t1 - t0) * 1000.0

        if out.dtype != np.uint8:
            out = np.clip(out, 0, 255).astype(np.uint8)

        psnr_val = calculate_psnr(out, gt_img)
        ssim_val = calculate_ssim(out, gt_img)
        entr_val = calculate_entropy(out)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            niqe_val = calculate_niqe(out)

        std_orig = float(low_img.std()) + 1e-8
        std_enh  = float(out.std())
        cii      = std_enh / std_orig

        epi_val = calculate_edge_preservation(out, gt_img)

        rows.append({
            "image":                    low_p.name,
            "psnr":                     round(float(psnr_val), 4),
            "ssim":                     round(float(ssim_val), 4),
            "entropy":                  round(float(entr_val), 4),
            "niqe":                     round(float(niqe_val), 4) if not np.isnan(niqe_val) else float("nan"),
            "contrast_improvement_index": round(cii, 4),
            "edge_preservation":        round(float(epi_val), 4),
            "time_ms":                  round(time_ms, 2),
        })

    return rows


# ===========================================================================
# HELPERS
# ===========================================================================

def _print_summary(label: str, rows: list[dict]) -> None:
    """Print mean ± std for key metrics across all evaluated images."""
    if not rows:
        print(f"  [{label}] No results to summarise.")
        return

    def _mean(key):
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float)) and not np.isnan(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    print(f"\n  ── {label} summary ({len(rows)} images) ──")
    print(f"     PSNR  : {_mean('psnr'):.2f} dB")
    print(f"     SSIM  : {_mean('ssim'):.4f}")
    print(f"     Entropy: {_mean('entropy'):.3f} bits")
    print(f"     CII   : {_mean('contrast_improvement_index'):.4f}")
    print(f"     EPI   : {_mean('edge_preservation'):.4f}")
    print(f"     t/img : {_mean('time_ms'):.1f} ms")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run image enhancement experiments on LOL dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--exp",
        choices=["1", "2", "3", "all"],
        default="all",
        help="Which experiment(s) to run.",
    )
    parser.add_argument(
        "--max-images", "-n",
        type=int,
        default=None,
        dest="max_images",
        help="Limit evaluation to first N images (useful for quick testing).",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
        help="Dataset split to evaluate on. 'test' = eval15, 'val' = 15%% of our485.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        dest="model_path",
        help="Path to trained CNN weights (.pth or .pt). Required for Exp 2 and Exp 3 CNN stage.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("dataset"),
        dest="data_root",
        help="Root directory of the dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the GA and validation split.",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    print("=" * 60)
    print("  Image Enhancement Experiments")
    print("=" * 60)
    print(f"  Experiment  : {args.exp}")
    print(f"  Split       : {args.split}")
    print(f"  Max images  : {args.max_images or 'all'}")
    print(f"  Data root   : {args.data_root}")
    print(f"  Model path  : {args.model_path or 'None'}")
    print(f"  Seed        : {args.seed}")

    # ── Warn about active stubs ──────────────────────────────────────────────
    stubs = using_stubs()
    active = [k for k, v in stubs.items() if v]
    if active:
        warnings.warn(
            f"\n[COMPAT] {len(active)} stub(s) active: {active}\n"
            "  Results may differ once real teammate modules are merged.",
            UserWarning,
            stacklevel=1,
        )

    # ── Load images for the requested split ──────────────────────────────────
    data_root = args.data_root.resolve()

    if args.split == "test":
        print(f"\n[DATA] Loading eval15 test set from {data_root}/eval15 …")
        low_paths, gt_paths = load_paired_split(
            data_root, "eval15", max_images=args.max_images
        )
        split_label = "test"
    else:
        # Validation split carved from our485
        print(f"\n[DATA] Loading our485 from {data_root}/our485 and carving val split …")
        all_low, all_gt = load_paired_split(data_root, "our485")
        _, (val_low, val_gt) = make_val_split(all_low, all_gt, seed=args.seed)
        low_paths  = val_low[:args.max_images]  if args.max_images else val_low
        gt_paths   = val_gt[:args.max_images]   if args.max_images else val_gt
        split_label = "val"

    print(f"  → {len(low_paths)} image pairs loaded.")

    # ── Dispatch to experiments ───────────────────────────────────────────────
    run_1 = args.exp in ("1", "all")
    run_2 = args.exp in ("2", "all")
    run_3 = args.exp in ("3", "all")

    if run_1:
        run_experiment_1(low_paths, gt_paths, split_label)
    if run_2:
        run_experiment_2(low_paths, gt_paths, split_label, args.model_path)
    if run_3:
        run_experiment_3(low_paths, gt_paths, split_label, args.model_path, args.seed)

    # ── Final CSV summary ────────────────────────────────────────────────────
    if METRICS_CSV.exists():
        print(f"\n[DONE] Results written to {METRICS_CSV}")
        all_rows = _load_csv(METRICS_CSV)
        # Print unique experiment+method combos and their row counts
        combos: dict[tuple, int] = {}
        for r in all_rows:
            k = (r.get("experiment", ""), r.get("method", ""))
            combos[k] = combos.get(k, 0) + 1
        print("\n  CSV contents:")
        for (exp, meth), count in sorted(combos.items()):
            print(f"    {exp:20s}  {meth:20s}  {count} rows")

    print("\n[ALL EXPERIMENTS COMPLETE]")


if __name__ == "__main__":
    main()
