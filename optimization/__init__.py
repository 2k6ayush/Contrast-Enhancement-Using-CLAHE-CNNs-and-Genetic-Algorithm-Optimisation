"""
optimization package — Member 3 (Genetic Algorithm + Integration)
==================================================================
Exposes the public interfaces agreed upon with the team:

    optimize_clahe(image, ground_truth=None, **kwargs) -> dict
        Find optimal CLAHE parameters using a Genetic Algorithm.

    enhance_hybrid(image, model, ground_truth=None, **kwargs) -> dict
        Full pipeline: GA-CLAHE → CNN refinement (rich dict return).

    enhance_hybrid_image(image, model, **kwargs) -> np.ndarray
        Thin wrapper — returns only the final uint8 image (agreed interface).

    enhance_ga_clahe(image, ground_truth=None, **kwargs) -> np.ndarray
        GA-CLAHE only (no CNN), for ablation and app use.

Sub-modules
-----------
genetic_algorithm : GA core (fitness, selection, crossover, mutation)
hybrid            : Chains GA-optimised CLAHE → CNN refinement
compat            : Fallback shims when teammates' code is absent
"""

from optimization.genetic_algorithm import optimize_clahe           # noqa: F401
from optimization.hybrid import (                                    # noqa: F401
    enhance_hybrid,
    enhance_hybrid_image,
    enhance_ga_clahe,
)

__all__ = [
    "optimize_clahe",
    "enhance_hybrid",
    "enhance_hybrid_image",
    "enhance_ga_clahe",
]
