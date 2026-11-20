"""
optimization package — Member 3 (Genetic Algorithm + Integration)
==================================================================
Exposes the two public interfaces agreed upon with the team:

    optimize_clahe(image, ground_truth=None) -> dict
    enhance_hybrid(image, model)             -> np.ndarray

Sub-modules
-----------
genetic_algorithm : GA core (fitness, selection, crossover, mutation)
hybrid            : Chains GA-optimised CLAHE → CNN refinement
compat            : Fallback shims when teammates' code is absent
"""

from optimization.genetic_algorithm import optimize_clahe  # noqa: F401
from optimization.hybrid import enhance_hybrid             # noqa: F401

__all__ = ["optimize_clahe", "enhance_hybrid"]
