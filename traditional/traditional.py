"""Traditional image enhancement: Histogram Equalization and CLAHE.

Interface contract (README section 9):
    * input / output: NumPy arrays, RGB, shape H x W x 3, dtype uint8, range 0-255
    * OpenCV BGR/LAB conversions are handled inside this module only.

Both methods work on the lightness channel (L of CIELAB) so that colours are
not distorted. Equalising R, G and B separately would shift hues.
"""
import cv2
import numpy as np


def _check_image(image):
    """Validate that `image` follows the agreed H x W x 3 uint8 RGB format."""
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must have shape H x W x 3, got {image.shape}")
    if image.dtype != np.uint8:
        # Be forgiving: accept float/int arrays in the 0-255 range.
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def apply_histogram_equalization(image):
    """Global Histogram Equalization on the lightness channel.

    Args:
        image: RGB uint8 array (H x W x 3).
    Returns:
        Enhanced RGB uint8 array (H x W x 3).
    """
    image = _check_image(image)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def apply_clahe(image, clip_limit=2.0, grid_size=8):
    """CLAHE on the lightness channel.

    Args:
        image: RGB uint8 array (H x W x 3).
        clip_limit: contrast limit per tile (higher = stronger, more noise).
        grid_size: number of tiles per side (int -> grid_size x grid_size),
                   or a (cols, rows) tuple.
    Returns:
        Enhanced RGB uint8 array (H x W x 3).
    """
    image = _check_image(image)
    if clip_limit <= 0:
        raise ValueError("clip_limit must be > 0")
    if isinstance(grid_size, (tuple, list)):
        tiles = (int(grid_size[0]), int(grid_size[1]))
    else:
        tiles = (int(grid_size), int(grid_size))
    if tiles[0] < 1 or tiles[1] < 1:
        raise ValueError("grid_size must be >= 1")

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tiles)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
