"""
optimization/hybrid.py — Hybrid GA-CLAHE → CNN Pipeline
=========================================================
Member 3 owns this file.

This is a placeholder skeleton that will be fully implemented in Step 3.
It exposes the agreed interface so other modules can import it safely now.

Public interface
----------------
    enhance_hybrid(image, model) -> np.ndarray
        Apply GA-optimised CLAHE, then refine with the CNN.
        Returns an RGB uint8 HxWx3 enhanced image.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from optimization.genetic_algorithm import optimize_clahe
from optimization.compat import apply_clahe, enhance_with_cnn


def enhance_hybrid(image: np.ndarray, model: Any) -> np.ndarray:
    """
    Full hybrid enhancement pipeline.

    Step 1 — GA finds optimal CLAHE parameters for the input image.
    Step 2 — CLAHE is applied with those parameters.
    Step 3 — CNN refines the CLAHE output.

    Parameters
    ----------
    image : np.ndarray — RGB uint8 HxWx3 low-light input
    model : Any        — trained CNN model (from cnn.model.build_model)

    Returns
    -------
    np.ndarray — RGB uint8 HxWx3 enhanced image
    """
    # Step 1: optimise CLAHE parameters
    best_params = optimize_clahe(image)
    clip_limit: float = best_params["clip_limit"]
    grid_size: int = best_params["grid_size"]

    # Step 2: apply CLAHE with optimal params
    clahe_out = apply_clahe(image, clip_limit=clip_limit, grid_size=grid_size)

    # Step 3: CNN refinement
    enhanced = enhance_with_cnn(clahe_out, model)

    return enhanced
