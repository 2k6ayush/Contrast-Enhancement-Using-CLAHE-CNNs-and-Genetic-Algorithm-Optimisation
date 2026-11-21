"""
optimization/hybrid.py — Hybrid GA-CLAHE → CNN Enhancement Pipeline
=====================================================================
Member 3 owns this file.

OVERVIEW
--------
This module chains two enhancement stages:

    Stage 1 — GA-optimised CLAHE
        The Genetic Algorithm (genetic_algorithm.py) searches for the best
        CLAHE parameters (clip_limit, tile_grid_size) for the specific input
        image, then applies CLAHE with those parameters.

    Stage 2 — CNN refinement
        The CLAHE output is passed through the teammate's CNN model
        (cnn/inference.py::enhance_with_cnn) for learned refinement.
        If model=None (CNN not yet available), Stage 2 is skipped with a
        warning so the pipeline stays functional at all times.

PUBLIC API
----------
    enhance_hybrid(image, model, ground_truth=None, **ga_kwargs) -> dict
        Full pipeline. Returns a rich dict with intermediate outputs and
        timing information.

    enhance_hybrid_image(image, model, **ga_kwargs) -> np.ndarray
        Thin wrapper — returns only the final uint8 RGB image.
        Satisfies the agreed interface: enhance_hybrid(image, model).

    enhance_ga_clahe(image, ground_truth=None, **ga_kwargs) -> np.ndarray
        Returns only the GA-CLAHE output (before CNN).
        Used for ablation study config C and the Streamlit app.

INTERFACES (agreed with team)
------------------------------
    Input  : np.ndarray, RGB, uint8, HxWx3
    Output : np.ndarray, RGB, uint8, HxWx3  (at all module boundaries)
    All BGR / LAB conversions happen *internally* inside compat helpers.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from optimization.genetic_algorithm import optimize_clahe
from optimization.compat import enhance_with_cnn


# ===========================================================================
# INPUT VALIDATION HELPER
# ===========================================================================

def _validate_image(image: Any, name: str = "image") -> None:
    """
    Validate that *image* is a proper RGB uint8 numpy array of shape HxWx3.

    Parameters
    ----------
    image : Any    — the value to validate
    name  : str    — variable name used in error messages

    Raises
    ------
    TypeError  — if image is not a numpy array
    ValueError — if shape, ndim, channels, or dtype are wrong
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"'{name}' must be a numpy.ndarray, got {type(image).__name__}."
        )
    if image.ndim != 3:
        raise ValueError(
            f"'{name}' must be 3-dimensional (H×W×C), got ndim={image.ndim}."
        )
    if image.shape[2] != 3:
        raise ValueError(
            f"'{name}' must have 3 channels (RGB), got {image.shape[2]} channels."
        )
    if image.dtype != np.uint8:
        raise ValueError(
            f"'{name}' must have dtype uint8, got {image.dtype}. "
            "Convert with: image.astype(np.uint8)"
        )


# ===========================================================================
# CORE PIPELINE
# ===========================================================================

def enhance_hybrid(
    image: np.ndarray,
    model: Any,
    ground_truth: Optional[np.ndarray] = None,
    **ga_kwargs,
) -> dict:
    """
    Full hybrid enhancement pipeline: GA-CLAHE → CNN refinement.

    Stage 1: The Genetic Algorithm finds optimal CLAHE parameters and
             applies CLAHE to produce an intermediate result.
    Stage 2: The CNN model refines the CLAHE output.
             If model is None, Stage 2 is skipped with a warning.

    Parameters
    ----------
    image        : np.ndarray        — RGB uint8 HxWx3, the low-light input
    model        : Any               — trained CNN model from cnn.model.build_model().
                                       Pass None to skip the CNN stage gracefully.
    ground_truth : np.ndarray | None — RGB uint8 HxWx3, normal-light reference.
                                       Forwarded to the GA for reference-based fitness.
    **ga_kwargs  : keyword arguments  — forwarded to optimize_clahe().
                                       e.g. pop_size=50, generations=50, seed=0

    Returns
    -------
    dict with keys:
        "output"         : np.ndarray — final RGB uint8 HxWx3 enhanced image
        "clahe_output"   : np.ndarray — GA-CLAHE result (before CNN)
        "clip_limit"     : float      — best clip limit chosen by GA
        "tile_grid_size" : int        — best tile grid size chosen by GA
        "ga_history"     : dict       — {"best": [...], "mean": [...]} per generation
        "time_ms"        : float      — total wall-clock time in milliseconds

    Raises
    ------
    TypeError  — if image is not a numpy ndarray
    ValueError — if image shape, channels, or dtype are invalid
    """
    # ── Validate inputs ──────────────────────────────────────────────────────
    _validate_image(image, "image")
    if ground_truth is not None:
        _validate_image(ground_truth, "ground_truth")

    t_start = time.perf_counter()

    # ── Stage 1: GA-optimised CLAHE ──────────────────────────────────────────
    # optimize_clahe() downscales internally for speed and returns the
    # full-resolution enhanced image in result["enhanced"].
    ga_result = optimize_clahe(image, ground_truth=ground_truth, **ga_kwargs)

    clahe_output: np.ndarray = ga_result["enhanced"]          # full-res, uint8, RGB
    clip_limit:   float      = ga_result["clip_limit"]
    tile_grid:    int        = ga_result["tile_grid_size"]
    ga_history:   dict       = ga_result["history"]

    # ── Stage 2: CNN refinement ──────────────────────────────────────────────
    if model is None:
        # CNN not yet available — skip gracefully
        warnings.warn(
            "\n[HYBRID] model=None detected. "
            "Skipping CNN refinement stage — returning GA-CLAHE output as final result.\n"
            "  Action: Pass a trained model from cnn.model.build_model() to enable CNN.",
            UserWarning,
            stacklevel=2,
        )
        final_output = clahe_output.copy()
    else:
        final_output = enhance_with_cnn(clahe_output, model)

    # ── Ensure output is uint8 RGB HxWx3 ────────────────────────────────────
    # Defensive: CNN might return float — clamp and cast.
    if final_output.dtype != np.uint8:
        final_output = np.clip(final_output, 0, 255).astype(np.uint8)

    t_end = time.perf_counter()

    return {
        "output":         final_output,
        "clahe_output":   clahe_output,
        "clip_limit":     clip_limit,
        "tile_grid_size": tile_grid,
        "ga_history":     ga_history,
        "time_ms":        (t_end - t_start) * 1000.0,
    }


# ===========================================================================
# CONVENIENCE WRAPPERS
# ===========================================================================

def enhance_hybrid_image(
    image: np.ndarray,
    model: Any,
    **ga_kwargs,
) -> np.ndarray:
    """
    Thin wrapper around enhance_hybrid() that returns only the final image.

    This satisfies the agreed team interface::

        enhance_hybrid(image, model) -> np.ndarray

    Teammates and the evaluation scripts can call this without unpacking a dict.

    Parameters
    ----------
    image     : np.ndarray — RGB uint8 HxWx3, the low-light input
    model     : Any        — trained CNN model (or None to skip CNN)
    **ga_kwargs            — forwarded to optimize_clahe()

    Returns
    -------
    np.ndarray — RGB uint8 HxWx3 enhanced image
    """
    result = enhance_hybrid(image, model, **ga_kwargs)
    return result["output"]


def enhance_ga_clahe(
    image: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    **ga_kwargs,
) -> np.ndarray:
    """
    Return only the GA-optimised CLAHE result (Stage 1 only, no CNN).

    Used for:
    - Ablation study config C (GA-CLAHE alone, without CNN)
    - Streamlit app (show GA-CLAHE as an intermediate result)

    Parameters
    ----------
    image        : np.ndarray        — RGB uint8 HxWx3, the low-light input
    ground_truth : np.ndarray | None — RGB uint8 HxWx3, reference (optional)
    **ga_kwargs                      — forwarded to optimize_clahe()

    Returns
    -------
    np.ndarray — RGB uint8 HxWx3 GA-CLAHE enhanced image
    """
    _validate_image(image, "image")
    if ground_truth is not None:
        _validate_image(ground_truth, "ground_truth")

    ga_result = optimize_clahe(image, ground_truth=ground_truth, **ga_kwargs)
    return ga_result["enhanced"]


# ===========================================================================
# DEMO / MAIN
# ===========================================================================

if __name__ == "__main__":
    """
    Quick demo that:
      1. Loads an image from argv[1] (or generates a synthetic dark image).
      2. Runs enhance_hybrid() with model=None (CNN skipped).
      3. Runs enhance_ga_clahe() standalone.
      4. Saves a 3-panel before/after image: Original | GA-CLAHE | Hybrid.
      5. Verifies output dtype, shape, and value range.

    Usage:
        python -m optimization.hybrid                      (synthetic image)
        python -m optimization.hybrid path/to/image.png   (real image)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 65)
    print("  optimization/hybrid.py  —  DEMO")
    print("=" * 65)

    # ── Load or generate image ───────────────────────────────────────────────
    if len(sys.argv) > 1:
        img_path = Path(sys.argv[1])
        raw = cv2.imread(str(img_path))
        if raw is None:
            print(f"[WARN] Could not read '{img_path}'. Falling back to synthetic.")
            img_path = None
        else:
            image = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            print(f"[IMAGE] Loaded '{img_path}'  shape={image.shape}")
    else:
        img_path = None

    if img_path is None or len(sys.argv) == 1:
        rng_img = np.random.default_rng(seed=7)
        base    = rng_img.integers(5, 50, size=(128, 128, 3), dtype=np.uint8)
        noise   = rng_img.integers(0, 10, size=(128, 128, 3), dtype=np.uint8)
        image   = np.clip(base.astype(np.int32) + noise, 0, 255).astype(np.uint8)
        print(f"[IMAGE] Synthetic dark image  shape={image.shape}  "
              f"mean={image.mean():.1f}")

    # ── Run enhance_ga_clahe (Stage 1 only) ──────────────────────────────────
    print("\n[STAGE 1] Running GA-CLAHE only (enhance_ga_clahe) …")
    ga_clahe_out = enhance_ga_clahe(image, seed=42, pop_size=30, generations=30)
    print(f"  Output  shape={ga_clahe_out.shape}  dtype={ga_clahe_out.dtype}  "
          f"mean={ga_clahe_out.mean():.1f}")

    # ── Run enhance_hybrid (Stage 1 + 2, model=None → CNN skipped) ──────────
    print("\n[STAGE 1+2] Running full hybrid pipeline (model=None) …")
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        result = enhance_hybrid(image, model=None, seed=42, pop_size=30, generations=30)

    # Print the CNN-skip warning so user sees it
    for w in caught_warnings:
        if "HYBRID" in str(w.message):
            print(f"  [WARNING] {w.message}")

    print(f"  clip_limit     : {result['clip_limit']:.4f}")
    print(f"  tile_grid_size : {result['tile_grid_size']}")
    print(f"  time_ms        : {result['time_ms']:.1f} ms")
    print(f"  clahe_output   : shape={result['clahe_output'].shape}  "
          f"dtype={result['clahe_output'].dtype}  "
          f"mean={result['clahe_output'].mean():.1f}")
    print(f"  output         : shape={result['output'].shape}  "
          f"dtype={result['output'].dtype}  "
          f"mean={result['output'].mean():.1f}")

    # ── Verification ─────────────────────────────────────────────────────────
    print("\n[CHECKS]")

    # (1) Output dtype and shape
    assert result["output"].dtype == np.uint8, "output dtype must be uint8!"
    assert result["output"].shape == image.shape, "output shape must match input!"
    print("  ✓ output dtype=uint8 and shape matches input")

    # (2) Value range
    assert result["output"].min() >= 0 and result["output"].max() <= 255, \
        "output values out of [0, 255] range!"
    print(f"  ✓ output value range [{result['output'].min()}, "
          f"{result['output'].max()}] ⊂ [0, 255]")

    # (3) enhance_hybrid_image wrapper returns a plain ndarray
    hybrid_arr = enhance_hybrid_image(image, model=None, seed=42)
    assert isinstance(hybrid_arr, np.ndarray), "enhance_hybrid_image must return ndarray!"
    assert hybrid_arr.dtype == np.uint8, "enhance_hybrid_image must return uint8!"
    print("  ✓ enhance_hybrid_image() returns uint8 ndarray (thin wrapper works)")

    # (4) GA-CLAHE output is consistent between standalone and hybrid call
    # (same seed → same GA → same clahe_output)
    assert np.array_equal(ga_clahe_out, result["clahe_output"]), \
        "GA-CLAHE output differs between standalone and hybrid calls!"
    print("  ✓ enhance_ga_clahe() output matches hybrid['clahe_output'] (same seed)")

    # (5) GA history list is non-empty
    assert len(result["ga_history"]["best"]) > 0, "ga_history must not be empty!"
    print(f"  ✓ ga_history has {len(result['ga_history']['best'])} generation entries")

    # ── Save 3-panel before/after image ─────────────────────────────────────
    out_dir = Path("results") / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].imshow(image)
    axes[0].set_title(f"Original\nmean={image.mean():.1f}", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(result["clahe_output"])
    axes[1].set_title(
        f"GA-CLAHE\nclip={result['clip_limit']:.2f}  "
        f"grid={result['tile_grid_size']}\n"
        f"mean={result['clahe_output'].mean():.1f}",
        fontsize=10,
    )
    axes[1].axis("off")

    axes[2].imshow(result["output"])
    axes[2].set_title(
        f"Hybrid (GA-CLAHE → CNN)\n"
        f"[CNN skipped: model=None]\n"
        f"mean={result['output'].mean():.1f}",
        fontsize=10,
    )
    axes[2].axis("off")

    fig.suptitle(
        f"Hybrid Pipeline Demo  |  time={result['time_ms']:.0f} ms",
        fontsize=12,
    )
    fig.tight_layout()

    out_path = out_dir / "hybrid_demo.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[SAVED]  3-panel demo → {out_path}")

    # ── Validation error demo ────────────────────────────────────────────────
    print("\n[VALIDATION] Testing bad-input error handling …")

    # Float image → should raise ValueError
    try:
        enhance_hybrid(image.astype(np.float32), model=None)
        print("  ✗ FAIL: should have raised ValueError for float input")
    except ValueError as exc:
        print(f"  ✓ float dtype caught: {exc}")

    # Wrong shape → should raise ValueError
    try:
        enhance_hybrid(image[:, :, 0], model=None)   # 2D grayscale
        print("  ✗ FAIL: should have raised ValueError for 2D input")
    except ValueError as exc:
        print(f"  ✓ 2D shape caught: {exc}")

    # Non-array → should raise TypeError
    try:
        enhance_hybrid([[1, 2, 3]], model=None)
        print("  ✗ FAIL: should have raised TypeError for list input")
    except TypeError as exc:
        print(f"  ✓ non-array type caught: {exc}")

    print("\n[ALL CHECKS PASSED]")
    print("=" * 65)
