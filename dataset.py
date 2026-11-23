"""Dataset helpers for LOL (primary) and BSD500 (extra testing).

Expected LOL layout after download and unzip:

    dataset/LOL/
        our485/low/*.png    our485/high/*.png     (485 training pairs)
        eval15/low/*.png    eval15/high/*.png     (15 test pairs)

All images are returned as RGB uint8 arrays (H x W x 3), the agreed format.
"""
import os

import cv2
import numpy as np

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def load_image(path, size=None):
    """Read an image as RGB uint8. `size` is an optional (width, height)."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if size is not None:
        rgb = cv2.resize(rgb, tuple(size), interpolation=cv2.INTER_AREA)
    return rgb


def save_image(path, image):
    """Save an RGB uint8 array to disk (creates folders as needed)."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(np.asarray(image, dtype=np.uint8), cv2.COLOR_RGB2BGR))


def _list_images(folder):
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTENSIONS))


def load_lol_pairs(root="dataset/LOL", split="eval15", size=None, limit=None):
    """Load paired (low-light, normal-light) images from LOL.

    Args:
        root: LOL folder.
        split: "our485" (train) or "eval15" (test).
        size: optional (width, height) to resize both images.
        limit: load only the first N pairs (handy for quick tests).
    Returns:
        list of (name, low_rgb, high_rgb).
    """
    low_dir = os.path.join(root, split, "low")
    high_dir = os.path.join(root, split, "high")
    for d in (low_dir, high_dir):
        if not os.path.isdir(d):
            raise FileNotFoundError(
                f"{d} not found. Download LOL and unzip it into '{root}' "
                "(see dataset/README.md)."
            )
    names = [n for n in _list_images(low_dir) if os.path.exists(os.path.join(high_dir, n))]
    if limit is not None:
        names = names[:limit]
    return [
        (n, load_image(os.path.join(low_dir, n), size), load_image(os.path.join(high_dir, n), size))
        for n in names
    ]


def load_folder_images(folder, size=None, limit=None):
    """Load every image in a folder (e.g. BSD500). Returns list of (name, rgb)."""
    names = _list_images(folder)
    if limit is not None:
        names = names[:limit]
    return [(n, load_image(os.path.join(folder, n), size)) for n in names]


def darken_image(image, gamma=2.5, brightness=0.6, noise_sigma=0.0, seed=0):
    """Create an artificial low-light image (for BSD500 experiments).

    out = 255 * (image/255)^gamma * brightness (+ Gaussian noise)
    The README requires that the darkening procedure is documented,
    so quote these parameters in the report.
    """
    x = np.asarray(image, dtype=np.float64) / 255.0
    x = (x ** gamma) * brightness
    if noise_sigma > 0:
        rng = np.random.default_rng(seed)
        x = x + rng.normal(0.0, noise_sigma / 255.0, x.shape)
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)
