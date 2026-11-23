"""Image quality metrics.

Reference-based (need ground truth): PSNR, SSIM
No-reference:                        Entropy, NIQE
Comparison with the input image:     Contrast Improvement Index, Edge Preservation

All functions take RGB uint8 arrays (H x W x 3, range 0-255).
"""
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _prep(a):
    return np.clip(np.asarray(a), 0, 255).astype(np.uint8)


def _gray(image):
    return cv2.cvtColor(_prep(image), cv2.COLOR_RGB2GRAY)


def _match_shape(a, b):
    a, b = _prep(a), _prep(b)
    if a.shape != b.shape:
        raise ValueError(f"image shapes differ: {a.shape} vs {b.shape}")
    return a, b


# ---------------------------------------------------------------- reference-based
def calculate_psnr(enhanced, ground_truth):
    """PSNR in dB (higher = closer to ground truth)."""
    enhanced, ground_truth = _match_shape(enhanced, ground_truth)
    return float(peak_signal_noise_ratio(ground_truth, enhanced, data_range=255))


def calculate_ssim(enhanced, ground_truth):
    """SSIM in [-1, 1] (higher = more structurally similar)."""
    enhanced, ground_truth = _match_shape(enhanced, ground_truth)
    return float(structural_similarity(ground_truth, enhanced, channel_axis=2, data_range=255))


# ---------------------------------------------------------------- no-reference
def calculate_entropy(image):
    """Shannon entropy of the grayscale histogram, in bits (max 8).

    Do not maximise blindly: noise also raises entropy.
    """
    hist = np.bincount(_gray(image).ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


_niqe_metric = None


def calculate_niqe(image):
    """NIQE (lower = more natural-looking). Needs the `pyiqa` package.

    pip install pyiqa
    (pyiqa downloads the pretrained NIQE parameters on first use.)
    """
    global _niqe_metric
    try:
        import torch
        import pyiqa
    except ImportError as exc:
        raise ImportError("NIQE needs pyiqa. Install it with: pip install pyiqa") from exc

    if _niqe_metric is None:
        _niqe_metric = pyiqa.create_metric("niqe", device=torch.device("cpu"))
    tensor = torch.from_numpy(_prep(image)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return float(_niqe_metric(tensor).item())


# ---------------------------------------------------------------- vs. input image
def calculate_contrast_improvement(original, enhanced):
    """Contrast Improvement Index = RMS contrast(enhanced) / RMS contrast(original).

    RMS contrast = standard deviation of grayscale intensities.
    > 1 means contrast increased. Too high usually means over-enhancement.
    """
    c_orig = float(_gray(original).std())
    c_enh = float(_gray(enhanced).std())
    return c_enh / c_orig if c_orig > 1e-8 else float("inf")


def _gradient_magnitude(gray):
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return np.hypot(gx, gy)


def calculate_edge_preservation(original, enhanced):
    """Edge preservation = correlation of Sobel gradient maps (higher = better).

    Returns a value in [-1, 1]; 1 means the edge structure is fully kept.
    """
    g1 = _gradient_magnitude(_gray(original)).ravel()
    g2 = _gradient_magnitude(_gray(enhanced)).ravel()
    if g1.std() < 1e-8 or g2.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(g1, g2)[0, 1])


# ---------------------------------------------------------------- convenience
def evaluate_all(original, enhanced, ground_truth=None, include_niqe=False):
    """Return every metric for one image as a dict (used by results tables).

    PSNR / SSIM are only included when `ground_truth` is given.
    NIQE is optional because it needs pyiqa and is slow.
    """
    result = {}
    if ground_truth is not None:
        result["PSNR"] = calculate_psnr(enhanced, ground_truth)
        result["SSIM"] = calculate_ssim(enhanced, ground_truth)
    result["Entropy"] = calculate_entropy(enhanced)
    if include_niqe:
        result["NIQE"] = calculate_niqe(enhanced)
    result["Contrast Improvement"] = calculate_contrast_improvement(original, enhanced)
    result["Edge Preservation"] = calculate_edge_preservation(original, enhanced)
    return result
