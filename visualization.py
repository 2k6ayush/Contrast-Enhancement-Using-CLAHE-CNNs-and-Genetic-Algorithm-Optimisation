"""Visual comparison helpers (Matplotlib)."""
import os

import cv2
import matplotlib

matplotlib.use("Agg")  # save figures without opening a window
import matplotlib.pyplot as plt
import numpy as np


def show_comparison(images, titles=None, save_path=None, figsize_per_image=3.5):
    """Plot images side by side.

    Args:
        images: list of RGB uint8 arrays, e.g. [low, HE, CLAHE, ground_truth].
        titles: list of titles (same length).
        save_path: PNG path; folders are created automatically.
    Returns:
        The Matplotlib figure.
    """
    n = len(images)
    titles = titles or [f"Image {i + 1}" for i in range(n)]
    fig, axes = plt.subplots(1, n, figsize=(figsize_per_image * n, figsize_per_image))
    axes = np.atleast_1d(axes)
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def error_map(enhanced, ground_truth):
    """Absolute difference between two images as a grayscale array (0-255)."""
    diff = np.abs(enhanced.astype(np.int16) - ground_truth.astype(np.int16))
    return diff.mean(axis=2).astype(np.uint8)


def show_error_maps(enhanced_dict, ground_truth, save_path=None):
    """Show |enhanced - ground truth| for each method (README asks for error maps).

    Args:
        enhanced_dict: {"HE": img, "CLAHE": img, ...}
    """
    n = len(enhanced_dict)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.5))
    axes = np.atleast_1d(axes)
    im = None
    for ax, (name, img) in zip(axes, enhanced_dict.items()):
        im = ax.imshow(error_map(img, ground_truth), cmap="inferno", vmin=0, vmax=255)
        ax.set_title(f"Error: {name}", fontsize=10)
        ax.axis("off")
    fig.colorbar(im, ax=list(axes), fraction=0.02, pad=0.02)
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def show_histograms(images, titles, save_path=None):
    """Plot grayscale histograms to show how each method changes intensity spread."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for img, title in zip(images, titles):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ax.hist(gray.ravel(), bins=64, range=(0, 255), histtype="step", label=title)
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Pixels")
    ax.legend()
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
