"""
optimization/compat.py — Compatibility / Fallback Layer
========================================================
Member 3 owns this file.

PURPOSE
-------
Teammates' modules (traditional/, cnn/) may not exist yet during development.
This module tries to import every function we depend on from the real source.
If an import fails it installs a *fallback* stub that:
  1. Prints a loud WARNING so we always know a stub is active.
  2. Behaves sensibly enough for the rest of our code to run.

USAGE
-----
In your own modules, always import from here — never directly from
traditional/ or cnn/.  Once the real modules are merged, the imports below
will succeed automatically and all warnings will disappear.

    from optimization.compat import apply_clahe, enhance_with_cnn, ...

PUBLIC API (re-exported whether real or stub)
---------------------------------------------
  apply_histogram_equalization(image)
  apply_clahe(image, clip_limit, grid_size)
  calculate_psnr(enhanced, ground_truth)
  calculate_ssim(enhanced, ground_truth)
  calculate_entropy(image)
  calculate_niqe(image)
  build_model()
  enhance_with_cnn(image, model)
  calculate_edge_preservation(enhanced, reference)   ← always real (Member 3)
  estimate_noise(image)                              ← always real (Member 3)
  using_stubs()  → dict[str, bool]  — True means stub is active
"""

from __future__ import annotations

import warnings
import importlib
from typing import Any

import cv2
import numpy as np
from scipy import ndimage

# ---------------------------------------------------------------------------
# Internal registry: maps function name → whether a stub is in use
# ---------------------------------------------------------------------------
_STUB_STATUS: dict[str, bool] = {}


def _warn(name: str, reason: str) -> None:
    """Emit a loud runtime warning that a stub is active."""
    warnings.warn(
        f"\n[COMPAT STUB] '{name}' is using a FALLBACK implementation.\n"
        f"  Reason : {reason}\n"
        f"  Action : Once teammates' modules are merged this warning disappears.\n",
        stacklevel=3,
    )


def _try_import(module_path: str, func_name: str):
    """
    Try to import *func_name* from *module_path*.
    Returns the callable on success, or None on failure.
    """
    try:
        mod = importlib.import_module(module_path)
        return getattr(mod, func_name)
    except (ImportError, AttributeError):
        return None


# ===========================================================================
# Member 1 — traditional/traditional.py
# ===========================================================================

# ── apply_histogram_equalization ────────────────────────────────────────────
_real_he = _try_import("traditional.traditional", "apply_histogram_equalization")

if _real_he is not None:
    apply_histogram_equalization = _real_he
    _STUB_STATUS["apply_histogram_equalization"] = False
else:
    def apply_histogram_equalization(image: np.ndarray) -> np.ndarray:
        """
        STUB: Histogram equalisation via OpenCV on the Y channel of YCrCb.

        Parameters
        ----------
        image : np.ndarray  — RGB uint8 HxWx3

        Returns
        -------
        np.ndarray — RGB uint8 HxWx3
        """
        _warn(
            "apply_histogram_equalization",
            "traditional.traditional not found — using OpenCV YCrCb fallback.",
        )
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

    _STUB_STATUS["apply_histogram_equalization"] = True


# ── apply_clahe ─────────────────────────────────────────────────────────────
_real_clahe = _try_import("traditional.traditional", "apply_clahe")

if _real_clahe is not None:
    apply_clahe = _real_clahe
    _STUB_STATUS["apply_clahe"] = False
else:
    def apply_clahe(
        image: np.ndarray,
        clip_limit: float = 2.0,
        grid_size: int = 8,
    ) -> np.ndarray:
        """
        STUB: CLAHE on the L channel of LAB colour space via OpenCV.

        Parameters
        ----------
        image      : np.ndarray — RGB uint8 HxWx3
        clip_limit : float      — CLAHE clip limit (default 2.0)
        grid_size  : int        — tile grid size in pixels (default 8)

        Returns
        -------
        np.ndarray — RGB uint8 HxWx3
        """
        _warn(
            "apply_clahe",
            "traditional.traditional not found — using OpenCV LAB-channel fallback.",
        )
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        clahe_obj = cv2.createCLAHE(
            clipLimit=float(clip_limit),
            tileGridSize=(int(grid_size), int(grid_size)),
        )
        lab[:, :, 0] = clahe_obj.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    _STUB_STATUS["apply_clahe"] = True


# ===========================================================================
# Member 1 — traditional/metrics.py
# ===========================================================================

# ── calculate_psnr ──────────────────────────────────────────────────────────
_real_psnr = _try_import("traditional.metrics", "calculate_psnr")

if _real_psnr is not None:
    calculate_psnr = _real_psnr
    _STUB_STATUS["calculate_psnr"] = False
else:
    def calculate_psnr(
        enhanced: np.ndarray,
        ground_truth: np.ndarray,
    ) -> float:
        """
        STUB: Peak Signal-to-Noise Ratio via scikit-image.

        Parameters
        ----------
        enhanced     : np.ndarray — RGB uint8 HxWx3
        ground_truth : np.ndarray — RGB uint8 HxWx3

        Returns
        -------
        float — PSNR in dB (higher is better)
        """
        _warn(
            "calculate_psnr",
            "traditional.metrics not found — using skimage.metrics fallback.",
        )
        from skimage.metrics import peak_signal_noise_ratio
        return float(
            peak_signal_noise_ratio(ground_truth, enhanced, data_range=255)
        )

    _STUB_STATUS["calculate_psnr"] = True


# ── calculate_ssim ──────────────────────────────────────────────────────────
_real_ssim = _try_import("traditional.metrics", "calculate_ssim")

if _real_ssim is not None:
    calculate_ssim = _real_ssim
    _STUB_STATUS["calculate_ssim"] = False
else:
    def calculate_ssim(
        enhanced: np.ndarray,
        ground_truth: np.ndarray,
    ) -> float:
        """
        STUB: Structural Similarity Index via scikit-image.

        Parameters
        ----------
        enhanced     : np.ndarray — RGB uint8 HxWx3
        ground_truth : np.ndarray — RGB uint8 HxWx3

        Returns
        -------
        float — SSIM in [0, 1] (higher is better)
        """
        _warn(
            "calculate_ssim",
            "traditional.metrics not found — using skimage.metrics fallback.",
        )
        from skimage.metrics import structural_similarity
        return float(
            structural_similarity(ground_truth, enhanced, channel_axis=2, data_range=255)
        )

    _STUB_STATUS["calculate_ssim"] = True


# ── calculate_entropy ───────────────────────────────────────────────────────
_real_entropy = _try_import("traditional.metrics", "calculate_entropy")

if _real_entropy is not None:
    calculate_entropy = _real_entropy
    _STUB_STATUS["calculate_entropy"] = False
else:
    def calculate_entropy(image: np.ndarray) -> float:
        """
        STUB: Shannon entropy of the grayscale histogram.

        Parameters
        ----------
        image : np.ndarray — RGB uint8 HxWx3

        Returns
        -------
        float — entropy in bits (higher = more information)
        """
        _warn(
            "calculate_entropy",
            "traditional.metrics not found — using NumPy histogram fallback.",
        )
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
        # Remove zero bins to avoid log(0)
        hist = hist[hist > 0].astype(np.float64)
        prob = hist / hist.sum()
        return float(-np.sum(prob * np.log2(prob)))

    _STUB_STATUS["calculate_entropy"] = True


# ── calculate_niqe ──────────────────────────────────────────────────────────
_real_niqe = _try_import("traditional.metrics", "calculate_niqe")

if _real_niqe is not None:
    calculate_niqe = _real_niqe
    _STUB_STATUS["calculate_niqe"] = False
else:
    def calculate_niqe(image: np.ndarray) -> float:
        """
        STUB: Returns NaN — NIQE requires the full traditional.metrics module.

        Parameters
        ----------
        image : np.ndarray — RGB uint8 HxWx3

        Returns
        -------
        float — NaN (placeholder)
        """
        _warn(
            "calculate_niqe",
            "traditional.metrics not found — NIQE returns NaN until real module is merged.",
        )
        return float("nan")

    _STUB_STATUS["calculate_niqe"] = True


# ===========================================================================
# Member 2 — cnn/model.py and cnn/inference.py
# ===========================================================================

# ── build_model ─────────────────────────────────────────────────────────────
_real_build_model = _try_import("cnn.model", "build_model")

if _real_build_model is not None:
    build_model = _real_build_model
    _STUB_STATUS["build_model"] = False
else:
    def build_model() -> Any:
        """
        STUB: Returns a sentinel object representing 'no CNN'.

        Returns
        -------
        None  — the stub enhance_with_cnn will recognise this and pass through.
        """
        _warn(
            "build_model",
            "cnn.model not found — returning None as a no-op model placeholder.",
        )
        return None

    _STUB_STATUS["build_model"] = True


# ── enhance_with_cnn ────────────────────────────────────────────────────────
_real_enhance_cnn = _try_import("cnn.inference", "enhance_with_cnn")

if _real_enhance_cnn is not None:
    enhance_with_cnn = _real_enhance_cnn
    _STUB_STATUS["enhance_with_cnn"] = False
else:
    def enhance_with_cnn(image: np.ndarray, model: Any) -> np.ndarray:
        """
        STUB: Identity pass-through — returns the input image unchanged.

        Parameters
        ----------
        image : np.ndarray — RGB uint8 HxWx3
        model : Any        — ignored (None from stub build_model)

        Returns
        -------
        np.ndarray — RGB uint8 HxWx3 (unchanged)
        """
        _warn(
            "enhance_with_cnn",
            "cnn.inference not found — returning input image unchanged (identity CNN).",
        )
        return image.copy()

    _STUB_STATUS["enhance_with_cnn"] = True


# ===========================================================================
# Member 3 — always-real helpers (we own these)
# ===========================================================================

def calculate_edge_preservation(
    enhanced: np.ndarray,
    reference: np.ndarray,
) -> float:
    """
    Edge Preservation Index (EPI) — Pearson correlation between the Sobel
    gradient magnitudes of *enhanced* and *reference*.

    A score of 1.0 means perfect edge preservation; 0.0 means no correlation.

    Parameters
    ----------
    enhanced  : np.ndarray — RGB uint8 HxWx3 (the processed image)
    reference : np.ndarray — RGB uint8 HxWx3 (the ground-truth or input)

    Returns
    -------
    float — Pearson correlation coefficient in [-1, 1]
    """
    # Convert to float32 grayscale for gradient computation
    enh_gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY).astype(np.float32)
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Sobel gradient magnitude
    def _sobel_magnitude(img: np.ndarray) -> np.ndarray:
        gx = ndimage.sobel(img, axis=1)
        gy = ndimage.sobel(img, axis=0)
        return np.hypot(gx, gy).ravel()

    enh_edges = _sobel_magnitude(enh_gray)
    ref_edges = _sobel_magnitude(ref_gray)

    # Pearson correlation
    if enh_edges.std() < 1e-8 or ref_edges.std() < 1e-8:
        return 0.0  # degenerate (flat) image — no correlation possible

    corr_matrix = np.corrcoef(enh_edges, ref_edges)
    return float(corr_matrix[0, 1])


def estimate_noise(image: np.ndarray) -> float:
    """
    Fast noise sigma estimation using Immerkaer's method.

    Reference: Immerkaer, J. (1996). "Fast Noise Variance Estimation."
    Computer Vision and Image Understanding, 64(2), 300-302.

    The estimator convolves the grayscale image with a Laplacian kernel
    designed to suppress low-frequency content and isolate noise.

    Parameters
    ----------
    image : np.ndarray — RGB uint8 HxWx3

    Returns
    -------
    float — estimated noise standard deviation (lower = cleaner image)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float64)
    h, w = gray.shape

    # Immerkaer's 3×3 Laplacian-based noise kernel
    kernel = np.array(
        [[ 1, -2,  1],
         [-2,  4, -2],
         [ 1, -2,  1]],
        dtype=np.float64,
    )

    response = ndimage.convolve(gray, kernel)
    # Normalisation constant from the paper
    sigma = np.sqrt(np.pi / 2.0) * np.mean(np.abs(response)) / 6.0
    return float(sigma)


# ===========================================================================
# Reporting helper
# ===========================================================================

def using_stubs() -> dict[str, bool]:
    """
    Return a dict mapping each managed function name to True if a stub
    (fallback) is currently active, False if the real implementation loaded.

    Example
    -------
    >>> from optimization.compat import using_stubs
    >>> for fn, is_stub in using_stubs().items():
    ...     print(f"{'STUB' if is_stub else 'REAL':4s}  {fn}")
    """
    return dict(_STUB_STATUS)


# ===========================================================================
# Smoke test — run with:  python -m optimization.compat
# ===========================================================================

if __name__ == "__main__":
    import warnings as _warnings

    # Show all compat warnings during the smoke test
    _warnings.simplefilter("always")

    print("=" * 65)
    print("  optimization/compat.py  —  SMOKE TEST")
    print("=" * 65)

    # ── Build a synthetic dark image (50×50 RGB, dim and slightly noisy) ──
    rng = np.random.default_rng(seed=42)
    dark_base = rng.integers(10, 60, size=(50, 50, 3), dtype=np.uint8)
    noise = rng.integers(0, 10, size=(50, 50, 3), dtype=np.uint8)
    dark_img: np.ndarray = np.clip(dark_base.astype(np.int32) + noise, 0, 255).astype(np.uint8)

    # Build a "ground-truth" that is brighter (simulate normal-light image)
    bright_img: np.ndarray = np.clip(dark_img.astype(np.int32) + 100, 0, 255).astype(np.uint8)

    print(f"\n[INPUT]  dark_img  shape={dark_img.shape}  dtype={dark_img.dtype}  "
          f"mean={dark_img.mean():.1f}")
    print(f"[INPUT]  bright_img shape={bright_img.shape}  dtype={bright_img.dtype}  "
          f"mean={bright_img.mean():.1f}\n")

    # ── 1. apply_histogram_equalization ────────────────────────────────────
    he_out = apply_histogram_equalization(dark_img)
    print(f"[HE]      output shape={he_out.shape}  mean={he_out.mean():.1f}")

    # ── 2. apply_clahe ─────────────────────────────────────────────────────
    clahe_out = apply_clahe(dark_img, clip_limit=3.0, grid_size=8)
    print(f"[CLAHE]   output shape={clahe_out.shape}  mean={clahe_out.mean():.1f}")

    # ── 3. calculate_psnr ──────────────────────────────────────────────────
    psnr_val = calculate_psnr(clahe_out, bright_img)
    print(f"[PSNR]    CLAHE vs bright = {psnr_val:.2f} dB")

    # ── 4. calculate_ssim ──────────────────────────────────────────────────
    ssim_val = calculate_ssim(clahe_out, bright_img)
    print(f"[SSIM]    CLAHE vs bright = {ssim_val:.4f}")

    # ── 5. calculate_entropy ───────────────────────────────────────────────
    ent_dark  = calculate_entropy(dark_img)
    ent_clahe = calculate_entropy(clahe_out)
    print(f"[ENTROPY] dark={ent_dark:.3f} bits  clahe={ent_clahe:.3f} bits")

    # ── 6. calculate_niqe ──────────────────────────────────────────────────
    niqe_val = calculate_niqe(clahe_out)
    print(f"[NIQE]    value = {niqe_val}  (NaN expected until real module merged)")

    # ── 7. build_model + enhance_with_cnn ──────────────────────────────────
    model = build_model()
    cnn_out = enhance_with_cnn(clahe_out, model)
    print(f"[CNN]     model={model}  output shape={cnn_out.shape}  "
          f"mean={cnn_out.mean():.1f}")

    # ── 8. calculate_edge_preservation ─────────────────────────────────────
    epi = calculate_edge_preservation(clahe_out, bright_img)
    print(f"[EDGE PRES] EPI = {epi:.4f}  (Pearson corr of Sobel gradients)")

    # ── 9. estimate_noise ──────────────────────────────────────────────────
    noise_dark  = estimate_noise(dark_img)
    noise_clahe = estimate_noise(clahe_out)
    print(f"[NOISE EST] dark sigma={noise_dark:.3f}  clahe sigma={noise_clahe:.3f}")

    # ── 10. using_stubs report ─────────────────────────────────────────────
    print("\n--- Stub Status Report ---")
    for fn_name, is_stub in using_stubs().items():
        tag = "STUB ⚠" if is_stub else "REAL  ✓"
        print(f"  {tag}  {fn_name}")

    print("\n[SMOKE TEST PASSED] All functions ran without exceptions.")
    print("=" * 65)
