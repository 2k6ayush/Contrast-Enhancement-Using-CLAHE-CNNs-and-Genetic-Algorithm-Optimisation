"""
evaluation/experiments.py — Experiment Runner
==============================================
Member 3 owns this file.

OVERVIEW
--------
Runs six experiments comparing image enhancement methods on the LOL
dataset (low-light pairs). Results are written to several CSV files in results/.

EXPERIMENTS
-----------
  Exp 1 — Traditional: Histogram Equalisation and default CLAHE.
  Exp 2 — CNN only:    build_model() + enhance_with_cnn(). Skipped if the
                        CNN is a stub or weights are unavailable.
  Exp 3 — Hybrid GA:  enhance_hybrid(), run in two modes:
            (a) hybrid_ref   — reference-based GA fitness (uses ground truth)
            (b) hybrid_noref — no-reference GA fitness (no ground truth given)
  Exp 4 — Hyperparameter tuning (validation split ONLY):
            (a) Fixed-param CLAHE sweep (clip × grid combinations)
            (b) GA parameter sweep (pop_size × gens × crossover × mutation)
  Exp 5 — Ablation study (test split): A=CLAHE, B=CNN, C=GA-CLAHE, D=hybrid
  Exp 6 — Performance comparison: aggregate table (Traditional, CNN, Hybrid)

DATASET
-------
  dataset/our485/low  + dataset/our485/high  → training + validation
  dataset/eval15/low  + dataset/eval15/high  → test set (never used for tuning)

  Validation split (15% of our485, fixed seed) is carved here but NOT used
  for GA tuning — it is available via --split val for hyperparameter selection.

USAGE
-----
  python evaluation/experiments.py --exp 1 --max-images 3
  python evaluation/experiments.py --exp all --split test
  python evaluation/experiments.py --exp 2 --model-path models/cnn_weights.pth
  python evaluation/experiments.py --exp 3 --max-images 10 --split val
  python evaluation/experiments.py --exp 4 --max-images 3 --n-configs 2
  python evaluation/experiments.py --exp 5
  python evaluation/experiments.py --exp 6

SAFE RE-RUN
-----------
  Results are written to results/metrics.csv (Exp 1-3), results/hyperparameter_tuning.csv
  (Exp 4), results/ablation.csv (Exp 5), results/comparison_table.csv (Exp 6).
  On re-run, existing rows for the same key are replaced.

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
from optimization.hybrid import enhance_hybrid, enhance_ga_clahe
from optimization.genetic_algorithm import optimize_clahe


# ===========================================================================
# CONSTANTS
# ===========================================================================

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
RESULTS_DIR      = PROJECT_ROOT / "results"
METRICS_CSV      = RESULTS_DIR / "metrics.csv"
HPARAM_CSV       = RESULTS_DIR / "hyperparameter_tuning.csv"
ABLATION_CSV     = RESULTS_DIR / "ablation.csv"
COMPARISON_CSV   = RESULTS_DIR / "comparison_table.csv"

# Column order for the per-image metrics CSV (Exp 1–3)
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

# Column order for hyperparameter tuning CSV (Exp 4)
HPARAM_COLUMNS = [
    "experiment",
    "method",
    "config_id",
    # CLAHE fixed-sweep params
    "clip_limit",
    "grid_size",
    # GA sweep params
    "pop_size",
    "generations",
    "crossover_rate",
    "mutation_rate",
    # CNN learning-rate placeholder (Member 2 fills this in)
    "cnn_learning_rate",
    "seed",
    # Aggregated metrics (mean over images)
    "psnr_mean",
    "psnr_std",
    "ssim_mean",
    "ssim_std",
    "time_ms_mean",
    "time_ms_std",
    "n_images",
]

# Column order for ablation CSV (Exp 5)
ABLATION_COLUMNS = [
    "config",
    "config_label",
    "image",
    "psnr",
    "ssim",
    "entropy",
    "niqe",
    "contrast_improvement_index",
    "edge_preservation",
    "time_ms",
]

# Column order for comparison table CSV (Exp 6)
COMPARISON_COLUMNS = [
    "approach",
    "method",
    "psnr_mean",
    "psnr_std",
    "ssim_mean",
    "ssim_std",
    "entropy_mean",
    "entropy_std",
    "niqe_mean",
    "niqe_std",
    "cii_mean",
    "cii_std",
    "epi_mean",
    "epi_std",
    "time_ms_mean",
    "time_ms_std",
    "model_size_mb",
    "n_images",
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


def _save_csv_generic(
    path: Path,
    rows: list[dict],
    columns: list[str],
) -> None:
    """
    Write rows to a CSV with custom columns, creating parent dirs as needed.

    Used for hyperparameter_tuning.csv, ablation.csv, comparison_table.csv.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            # Ensure all columns present (fill missing with "")
            safe = {col: row.get(col, "") for col in columns}
            writer.writerow(safe)


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
# EXPERIMENT 4 — Hyperparameter Tuning (VALIDATION SPLIT ONLY)
# ===========================================================================

# ── 4a: Fixed-param CLAHE sweep grid ────────────────────────────────────────
CLAHE_CLIP_VALUES = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
CLAHE_GRID_VALUES = [4, 8, 12, 16]

# ── 4b: GA parameter sweep grid ────────────────────────────────────────────
GA_POP_SIZES       = [20, 30, 50]
GA_GENERATIONS     = [20, 50]
GA_CROSSOVER_RATES = [0.7, 0.8, 0.9]
GA_MUTATION_RATES  = [0.01, 0.05, 0.1]


def _evaluate_clahe_config(
    clip_limit: float,
    grid_size: int,
    low_paths: list[Path],
    gt_paths:  list[Path],
) -> dict:
    """
    Evaluate one CLAHE configuration on a set of images.

    Returns a dict with aggregated metrics (mean/std of PSNR, SSIM, time).
    """
    psnrs, ssims, times = [], [], []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for low_p, gt_p in zip(low_paths, gt_paths):
            low_img = load_image(low_p)
            gt_img  = load_image(gt_p)

            t0  = time.perf_counter()
            out = apply_clahe(low_img, clip_limit=clip_limit, grid_size=grid_size)
            t1  = time.perf_counter()

            psnrs.append(calculate_psnr(out, gt_img))
            ssims.append(calculate_ssim(out, gt_img))
            times.append((t1 - t0) * 1000.0)

    return {
        "psnr_mean":    round(float(np.mean(psnrs)), 4),
        "psnr_std":     round(float(np.std(psnrs)), 4),
        "ssim_mean":    round(float(np.mean(ssims)), 4),
        "ssim_std":     round(float(np.std(ssims)), 4),
        "time_ms_mean": round(float(np.mean(times)), 2),
        "time_ms_std":  round(float(np.std(times)), 2),
        "n_images":     len(psnrs),
    }


def _evaluate_ga_config(
    pop_size: int,
    generations: int,
    crossover_rate: float,
    mutation_rate: float,
    seed: int,
    low_paths: list[Path],
    gt_paths:  list[Path],
) -> dict:
    """
    Evaluate one GA configuration (using no-reference fitness) on a set of images.

    Returns a dict with aggregated metrics (mean/std of PSNR, SSIM, time).
    """
    psnrs, ssims, times = [], [], []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for low_p, gt_p in zip(low_paths, gt_paths):
            low_img = load_image(low_p)
            gt_img  = load_image(gt_p)

            t0 = time.perf_counter()
            ga_result = optimize_clahe(
                low_img,
                ground_truth=None,  # no-reference — we evaluate against GT afterwards
                pop_size=pop_size,
                generations=generations,
                crossover_rate=crossover_rate,
                mutation_rate=mutation_rate,
                seed=seed,
            )
            t1 = time.perf_counter()

            out = ga_result["enhanced"]
            psnrs.append(calculate_psnr(out, gt_img))
            ssims.append(calculate_ssim(out, gt_img))
            times.append((t1 - t0) * 1000.0)

    return {
        "psnr_mean":    round(float(np.mean(psnrs)), 4),
        "psnr_std":     round(float(np.std(psnrs)), 4),
        "ssim_mean":    round(float(np.mean(ssims)), 4),
        "ssim_std":     round(float(np.std(ssims)), 4),
        "time_ms_mean": round(float(np.mean(times)), 2),
        "time_ms_std":  round(float(np.std(times)), 2),
        "n_images":     len(psnrs),
    }


def run_experiment_4(
    data_root:  Path,
    max_images: Optional[int] = None,
    n_configs:  int = 12,
    seed:       int = 42,
) -> None:
    """
    Experiment 4: Hyperparameter tuning — VALIDATION split ONLY.

    4a — Fixed-param CLAHE sweep:
         clip_limit ∈ {1,2,3,4,6,8} × grid_size ∈ {4,8,12,16} = 24 configs.
    4b — GA parameter sweep:
         Randomly sample `n_configs` combinations from the full grid of
         pop_size × generations × crossover_rate × mutation_rate.
         Each config is run with 3 seeds (seed, seed+1, seed+2) and results
         are averaged to report mean and std.

    Results are saved to results/hyperparameter_tuning.csv.

    NOTE: CNN learning rate is Member 2's responsibility. A placeholder column
    'cnn_learning_rate' is included in the CSV schema so Member 2 can fill it in.

    Parameters
    ----------
    data_root  : Path         — dataset root directory
    max_images : int | None   — limit val images (for quick testing)
    n_configs  : int          — number of random GA configs to sample (default 12)
    seed       : int          — base seed for reproducibility
    """
    print("\n" + "=" * 60)
    print("  EXPERIMENT 4 — Hyperparameter Tuning  [val split only]")
    print("=" * 60)

    # ── Load validation split from our485 ────────────────────────────────────
    all_low, all_gt = load_paired_split(data_root, "our485")
    _, (val_low, val_gt) = make_val_split(all_low, all_gt, seed=seed)

    if max_images:
        val_low = val_low[:max_images]
        val_gt  = val_gt[:max_images]

    print(f"  Validation images: {len(val_low)}")

    all_rows: list[dict] = []

    # ── 4a: Fixed-param CLAHE sweep ──────────────────────────────────────────
    print(f"\n[4a] CLAHE sweep: {len(CLAHE_CLIP_VALUES)} clips × "
          f"{len(CLAHE_GRID_VALUES)} grids = "
          f"{len(CLAHE_CLIP_VALUES) * len(CLAHE_GRID_VALUES)} configs")

    config_id = 0
    clahe_configs = [
        (clip, grid)
        for clip in CLAHE_CLIP_VALUES
        for grid in CLAHE_GRID_VALUES
    ]

    for clip, grid in tqdm(clahe_configs, desc="CLAHE sweep"):
        config_id += 1
        metrics = _evaluate_clahe_config(clip, grid, val_low, val_gt)

        row = {
            "experiment":       "exp4_clahe_sweep",
            "method":           "clahe_fixed",
            "config_id":        config_id,
            "clip_limit":       clip,
            "grid_size":        grid,
            "pop_size":         "",
            "generations":      "",
            "crossover_rate":   "",
            "mutation_rate":    "",
            "cnn_learning_rate": "",   # placeholder for Member 2
            "seed":             "",
        }
        row.update(metrics)
        all_rows.append(row)

    # Find best CLAHE config
    best_clahe = max(all_rows, key=lambda r: float(r.get("ssim_mean", 0)))
    print(f"\n  Best CLAHE config: clip={best_clahe['clip_limit']}, "
          f"grid={best_clahe['grid_size']}  →  "
          f"SSIM={best_clahe['ssim_mean']}, PSNR={best_clahe['psnr_mean']} dB")

    # ── 4b: GA parameter sweep ───────────────────────────────────────────────
    # Build full grid, then sample n_configs randomly
    full_ga_grid = [
        (pop, gen, cx, mut)
        for pop in GA_POP_SIZES
        for gen in GA_GENERATIONS
        for cx  in GA_CROSSOVER_RATES
        for mut in GA_MUTATION_RATES
    ]
    total_combos = len(full_ga_grid)

    rng = np.random.default_rng(seed=seed)
    n_to_sample = min(n_configs, total_combos)
    chosen_indices = rng.choice(total_combos, size=n_to_sample, replace=False)
    chosen_configs = [full_ga_grid[i] for i in sorted(chosen_indices)]

    ga_seeds = [seed, seed + 1, seed + 2]  # 3 seeds per config

    print(f"\n[4b] GA sweep: {total_combos} possible combos → sampling {n_to_sample} configs")
    print(f"     Each config runs with {len(ga_seeds)} seeds, avg over {len(val_low)} images")

    ga_rows: list[dict] = []
    for pop, gen, cx, mut in tqdm(chosen_configs, desc="GA sweep"):
        config_id += 1
        # Run each config with multiple seeds, collect per-seed metrics
        seed_psnrs, seed_ssims, seed_times = [], [], []

        for s in ga_seeds:
            m = _evaluate_ga_config(pop, gen, cx, mut, s, val_low, val_gt)
            seed_psnrs.append(float(m["psnr_mean"]))
            seed_ssims.append(float(m["ssim_mean"]))
            seed_times.append(float(m["time_ms_mean"]))

        row = {
            "experiment":       "exp4_ga_sweep",
            "method":           "ga_clahe",
            "config_id":        config_id,
            "clip_limit":       "",
            "grid_size":        "",
            "pop_size":         pop,
            "generations":      gen,
            "crossover_rate":   cx,
            "mutation_rate":    mut,
            "cnn_learning_rate": "",   # placeholder for Member 2
            "seed":             f"{ga_seeds}",
            "psnr_mean":        round(float(np.mean(seed_psnrs)), 4),
            "psnr_std":         round(float(np.std(seed_psnrs)), 4),
            "ssim_mean":        round(float(np.mean(seed_ssims)), 4),
            "ssim_std":         round(float(np.std(seed_ssims)), 4),
            "time_ms_mean":     round(float(np.mean(seed_times)), 2),
            "time_ms_std":      round(float(np.std(seed_times)), 2),
            "n_images":         len(val_low),
        }
        ga_rows.append(row)

    all_rows.extend(ga_rows)

    # Find best GA config
    if ga_rows:
        best_ga = max(ga_rows, key=lambda r: float(r.get("ssim_mean", 0)))
        print(f"\n  Best GA config: pop={best_ga['pop_size']}, gen={best_ga['generations']}, "
              f"cx={best_ga['crossover_rate']}, mut={best_ga['mutation_rate']}  →  "
              f"SSIM={best_ga['ssim_mean']}, PSNR={best_ga['psnr_mean']} dB, "
              f"time={best_ga['time_ms_mean']} ms")

    # ── Save to CSV ──────────────────────────────────────────────────────────
    _save_csv_generic(HPARAM_CSV, all_rows, HPARAM_COLUMNS)
    print(f"\n  → Saved {len(all_rows)} rows to {HPARAM_CSV}")

    print("\n  NOTE FOR MEMBER 2: The 'cnn_learning_rate' column is a placeholder.")
    print("  Please ask Member 2 to record their CNN training learning rate and")
    print("  append it to this CSV, or provide a separate CSV with columns:")
    print("    cnn_learning_rate, psnr_mean, ssim_mean, time_ms_mean")


# ===========================================================================
# EXPERIMENT 5 — Ablation Study (TEST split)
# ===========================================================================

def run_experiment_5(
    low_paths:  list[Path],
    gt_paths:   list[Path],
    model_path: Optional[Path] = None,
    ga_seed:    int = 42,
) -> None:
    """
    Experiment 5: Ablation study on the test split.

    Configurations:
        A — Default CLAHE only (clip=2.0, grid=8)
        B — CNN only (skipped if stub/no weights)
        C — GA-CLAHE only (no CNN refinement)
        D — Full hybrid (GA-CLAHE → CNN)

    All use the test split (eval15) and are saved to results/ablation.csv.
    This experiment isolates the contribution of each component.

    Parameters
    ----------
    low_paths  : list[Path]        — test-split low-light image paths
    gt_paths   : list[Path]        — test-split ground-truth image paths
    model_path : Path | None       — CNN weights for configs B and D
    ga_seed    : int               — random seed for the GA
    """
    print("\n" + "=" * 60)
    print("  EXPERIMENT 5 — Ablation Study  [test]")
    print("=" * 60)

    all_rows: list[dict] = []
    stubs = using_stubs()

    # ── Helper to evaluate one config ────────────────────────────────────────
    def _run_ablation_config(
        config: str,
        label: str,
        method_fn_paired: Callable,
    ) -> list[dict]:
        """Run one ablation config and return per-image rows."""
        rows = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for low_p, gt_p in tqdm(
                zip(low_paths, gt_paths),
                total=len(low_paths),
                desc=f"Abl-{config}",
            ):
                low_img = load_image(low_p)
                gt_img  = load_image(gt_p)

                t0  = time.perf_counter()
                out = method_fn_paired(low_img, gt_img)
                t1  = time.perf_counter()

                if out.dtype != np.uint8:
                    out = np.clip(out, 0, 255).astype(np.uint8)

                niqe_v = calculate_niqe(out)
                rows.append({
                    "config":       config,
                    "config_label": label,
                    "image":        low_p.name,
                    "psnr":         round(float(calculate_psnr(out, gt_img)), 4),
                    "ssim":         round(float(calculate_ssim(out, gt_img)), 4),
                    "entropy":      round(float(calculate_entropy(out)), 4),
                    "niqe":         round(float(niqe_v), 4)
                                    if not np.isnan(niqe_v) else float("nan"),
                    "contrast_improvement_index": round(
                        float(out.std()) / (float(low_img.std()) + 1e-8), 4
                    ),
                    "edge_preservation": round(
                        float(calculate_edge_preservation(out, gt_img)), 4
                    ),
                    "time_ms":      round((t1 - t0) * 1000.0, 2),
                })
        return rows

    # ── Config A: Default CLAHE ──────────────────────────────────────────────
    print("\n[A] Default CLAHE (clip=2.0, grid=8)")
    a_rows = _run_ablation_config(
        "A", "Default CLAHE",
        lambda low, gt: apply_clahe(low, clip_limit=2.0, grid_size=8),
    )
    all_rows.extend(a_rows)
    _print_summary("Config A — CLAHE", a_rows)

    # ── Config B: CNN only ───────────────────────────────────────────────────
    print("\n[B] CNN only")
    model = None
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
        except Exception as exc:
            print(f"  [WARN] Could not load CNN: {exc}")
            model = None

    if model is not None:
        b_rows = _run_ablation_config(
            "B", "CNN only",
            lambda low, gt: enhance_with_cnn(low, model),
        )
        all_rows.extend(b_rows)
        _print_summary("Config B — CNN", b_rows)
    else:
        print("  [SKIP] CNN not available (stub or no weights). Config B skipped.")

    # ── Config C: GA-CLAHE only (no CNN) ─────────────────────────────────────
    print("\n[C] GA-CLAHE only (no CNN)")
    c_rows = _run_ablation_config(
        "C", "GA-CLAHE (no CNN)",
        lambda low, gt: enhance_ga_clahe(low, ground_truth=None, seed=ga_seed),
    )
    all_rows.extend(c_rows)
    _print_summary("Config C — GA-CLAHE", c_rows)

    # ── Config D: Full hybrid (GA-CLAHE → CNN) ──────────────────────────────
    print("\n[D] Full hybrid (GA-CLAHE → CNN)")
    d_rows = _run_ablation_config(
        "D", "Full hybrid",
        lambda low, gt: enhance_hybrid(
            low, model, ground_truth=None, seed=ga_seed
        )["output"],
    )
    all_rows.extend(d_rows)
    _print_summary("Config D — Hybrid", d_rows)

    # ── Save ─────────────────────────────────────────────────────────────────
    _save_csv_generic(ABLATION_CSV, all_rows, ABLATION_COLUMNS)
    print(f"\n  → Saved {len(all_rows)} rows to {ABLATION_CSV}")


# ===========================================================================
# EXPERIMENT 6 — Performance Comparison Table
# ===========================================================================

def run_experiment_6(
    low_paths:   list[Path],
    gt_paths:    list[Path],
    model_path:  Optional[Path] = None,
    ga_seed:     int = 42,
) -> None:
    """
    Experiment 6: Aggregate performance comparison table.

    Approaches:
        Traditional — best of HE and default CLAHE (runs both, picks higher SSIM)
        ML/DL      — CNN only (skipped if stub)
        Proposed   — Full hybrid (GA-CLAHE → CNN)

    For each approach, reports: mean ± std for PSNR, SSIM, Entropy, NIQE,
    CII, EPI, time_ms, plus model size (MB) if available.

    Saved to results/comparison_table.csv.

    Parameters
    ----------
    low_paths   : list[Path]     — test-split low-light image paths
    gt_paths    : list[Path]     — test-split ground-truth image paths
    model_path  : Path | None    — CNN weights
    ga_seed     : int            — random seed
    """
    print("\n" + "=" * 60)
    print("  EXPERIMENT 6 — Performance Comparison Table  [test]")
    print("=" * 60)

    stubs = using_stubs()
    table_rows: list[dict] = []

    def _aggregate(label: str, approach: str, rows: list[dict],
                   model_size_mb: str = "") -> dict:
        """Aggregate per-image rows into a summary row."""
        def _m(key):
            vals = [float(r[key]) for r in rows
                    if isinstance(r.get(key), (int, float))
                    and not np.isnan(float(r.get(key, "nan")))]
            return (round(float(np.mean(vals)), 4) if vals else float("nan"),
                    round(float(np.std(vals)), 4)  if vals else float("nan"))

        pm, ps = _m("psnr")
        sm, ss = _m("ssim")
        em, es = _m("entropy")
        nm, ns = _m("niqe")
        cm, cs = _m("contrast_improvement_index")
        ep_m, ep_s = _m("edge_preservation")
        tm, ts = _m("time_ms")

        return {
            "approach":      approach,
            "method":        label,
            "psnr_mean":     pm, "psnr_std":     ps,
            "ssim_mean":     sm, "ssim_std":     ss,
            "entropy_mean":  em, "entropy_std":  es,
            "niqe_mean":     nm, "niqe_std":     ns,
            "cii_mean":      cm, "cii_std":      cs,
            "epi_mean":      ep_m, "epi_std":    ep_s,
            "time_ms_mean":  tm, "time_ms_std":  ts,
            "model_size_mb": model_size_mb,
            "n_images":      len(rows),
        }

    # ── Traditional: HE ──────────────────────────────────────────────────────
    print("\n[6] Running Histogram Equalisation …")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        he_rows = evaluate_method(
            method_fn=lambda img: apply_histogram_equalization(img),
            low_paths=low_paths, gt_paths=gt_paths, desc="HE-cmp",
        )
    table_rows.append(_aggregate("Histogram EQ", "Traditional", he_rows))

    # ── Traditional: CLAHE ────────────────────────────────────────────────────
    print("[6] Running default CLAHE …")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clahe_rows = evaluate_method(
            method_fn=lambda img: apply_clahe(img, clip_limit=2.0, grid_size=8),
            low_paths=low_paths, gt_paths=gt_paths, desc="CLAHE-cmp",
        )
    table_rows.append(_aggregate("Default CLAHE", "Traditional", clahe_rows))

    # ── CNN only ──────────────────────────────────────────────────────────────
    model = None
    model_size_mb = ""
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
            model_size_mb = round(model_path.stat().st_size / (1024 ** 2), 2)
        except Exception as exc:
            print(f"  [WARN] Could not load CNN: {exc}")
            model = None

    if model is not None:
        print("[6] Running CNN only …")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cnn_rows = evaluate_method(
                method_fn=lambda img: enhance_with_cnn(img, model),
                low_paths=low_paths, gt_paths=gt_paths, desc="CNN-cmp",
            )
        table_rows.append(_aggregate("CNN", "ML/DL", cnn_rows, str(model_size_mb)))
    else:
        print("  [SKIP] CNN not available. ML/DL row will be absent.")

    # ── Proposed: Hybrid ──────────────────────────────────────────────────────
    print("[6] Running Hybrid (GA-CLAHE → CNN) …")
    hybrid_rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for low_p, gt_p in tqdm(
            zip(low_paths, gt_paths), total=len(low_paths), desc="Hybrid-cmp"
        ):
            low_img = load_image(low_p)
            gt_img  = load_image(gt_p)

            t0 = time.perf_counter()
            result = enhance_hybrid(
                low_img, model, ground_truth=None, seed=ga_seed,
            )
            out = result["output"]
            t1 = time.perf_counter()

            if out.dtype != np.uint8:
                out = np.clip(out, 0, 255).astype(np.uint8)

            niqe_v = calculate_niqe(out)
            hybrid_rows.append({
                "image": low_p.name,
                "psnr":  round(float(calculate_psnr(out, gt_img)), 4),
                "ssim":  round(float(calculate_ssim(out, gt_img)), 4),
                "entropy": round(float(calculate_entropy(out)), 4),
                "niqe":  round(float(niqe_v), 4)
                         if not np.isnan(niqe_v) else float("nan"),
                "contrast_improvement_index": round(
                    float(out.std()) / (float(low_img.std()) + 1e-8), 4
                ),
                "edge_preservation": round(
                    float(calculate_edge_preservation(out, gt_img)), 4
                ),
                "time_ms": round((t1 - t0) * 1000.0, 2),
            })

    table_rows.append(_aggregate(
        "GA-CLAHE + CNN (Proposed)", "Proposed", hybrid_rows,
        str(model_size_mb) if model is not None else "",
    ))

    # ── Save & print ─────────────────────────────────────────────────────────
    _save_csv_generic(COMPARISON_CSV, table_rows, COMPARISON_COLUMNS)
    print(f"\n  → Saved {len(table_rows)} rows to {COMPARISON_CSV}")

    # Pretty-print the table
    print("\n  ┌──────────────────────────────────────────────────────────────┐")
    print("  │               PERFORMANCE COMPARISON TABLE                  │")
    print("  ├──────────────────────────────────────────────────────────────┤")
    for r in table_rows:
        psnr_s = (f"{r['psnr_mean']:.2f}±{r['psnr_std']:.2f}"
                  if not np.isnan(r['psnr_mean']) else "N/A")
        ssim_s = (f"{r['ssim_mean']:.4f}±{r['ssim_std']:.4f}"
                  if not np.isnan(r['ssim_mean']) else "N/A")
        t_s    = (f"{r['time_ms_mean']:.1f}±{r['time_ms_std']:.1f}"
                  if not np.isnan(r['time_ms_mean']) else "N/A")
        print(f"  │ {r['method']:25s}  PSNR={psnr_s:14s}  "
              f"SSIM={ssim_s:14s}  t={t_s:12s} │")
    print("  └──────────────────────────────────────────────────────────────┘")


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
        choices=["1", "2", "3", "4", "5", "6", "all"],
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
    parser.add_argument(
        "--n-configs",
        type=int,
        default=12,
        dest="n_configs",
        help="Number of random GA configurations to sample in Exp 4b.",
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
    if args.exp in ("4", "all"):
        print(f"  N configs   : {args.n_configs}")

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

    # ── Load images for the requested split (Exp 1-3) ────────────────────────
    # Exp 4 loads its own val split internally.
    # Exp 5 and 6 always use the test split.
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
    run_4 = args.exp in ("4", "all")
    run_5 = args.exp in ("5", "all")
    run_6 = args.exp in ("6", "all")

    if run_1:
        run_experiment_1(low_paths, gt_paths, split_label)
    if run_2:
        run_experiment_2(low_paths, gt_paths, split_label, args.model_path)
    if run_3:
        run_experiment_3(low_paths, gt_paths, split_label, args.model_path, args.seed)
    if run_4:
        # Exp 4 always uses VALIDATION split (never test) — loads its own data
        run_experiment_4(data_root, args.max_images, args.n_configs, args.seed)
    if run_5:
        # Exp 5 always uses TEST split
        test_low, test_gt = load_paired_split(data_root, "eval15")
        run_experiment_5(test_low, test_gt, args.model_path, args.seed)
    if run_6:
        # Exp 6 always uses TEST split
        test_low, test_gt = load_paired_split(data_root, "eval15")
        run_experiment_6(test_low, test_gt, args.model_path, args.seed)

    # ── Final CSV summary ────────────────────────────────────────────────────
    result_csvs = [METRICS_CSV, HPARAM_CSV, ABLATION_CSV, COMPARISON_CSV]
    for csv_file in result_csvs:
        if csv_file.exists():
            rows = _load_csv(csv_file)
            print(f"\n[DONE] {csv_file.name}: {len(rows)} total rows")

    print("\n[ALL EXPERIMENTS COMPLETE]")


if __name__ == "__main__":
    main()
