"""
optimization/genetic_algorithm.py — Genetic Algorithm for CLAHE Optimisation
=============================================================================
Member 3 owns this file.

OVERVIEW
--------
This module implements a Genetic Algorithm (GA) that searches for the best
CLAHE (Contrast-Limited Adaptive Histogram Equalization) parameters for a
given input image.

CHROMOSOME ENCODING
-------------------
Each individual (chromosome) is a pair:
    [clip_limit (float), tile_grid_size (int)]

Where:
    clip_limit     ∈ [CLIP_MIN, CLIP_MAX]       = [1.0, 8.0]
    tile_grid_size ∈ [GRID_MIN, GRID_MAX]       = [4,   16]

GA OPERATORS
------------
- Selection   : Tournament selection (selects the best of k random candidates)
- Crossover   : Blend crossover (BLX-α) for clip_limit (continuous)
                Uniform crossover for tile_grid_size (discrete)
- Mutation    : Gaussian perturbation for clip_limit (clipped to range)
                ±1 step mutation for tile_grid_size (clipped to range)
- Elitism     : Top-k individuals always survive to the next generation

FITNESS FUNCTION
----------------
Two modes, controlled by whether ground_truth is provided:

  Reference-based (GT available):
      fitness = W_SSIM * SSIM
              + W_PSNR * clip(PSNR / PSNR_NORM, 0, 1)
              + W_EDGE_REF * edge_preservation(enhanced, GT)
              - W_NOISE * noise_penalty

  No-reference (GT not available):
      fitness = W_ENTROPY * (entropy / 8)
              + W_EDGE_NR  * edge_preservation(enhanced, original)
              + W_BRIGHT   * brightness_score
              - W_NOISE    * noise_penalty

  brightness_score = clip(1 - |mean(enhanced)/255 - 0.5| * 2, 0, 1)
  noise_penalty    = clip(max(0, σ(enhanced) - σ(original)) / NOISE_DIV, 0, 1)

PUBLIC API
----------
    optimize_clahe(image, ground_truth=None, **kwargs) -> dict
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Import metric helpers from our compatibility layer.
# This guarantees we always have working implementations even if teammates'
# code is not yet merged.
# ---------------------------------------------------------------------------
from optimization.compat import (
    apply_clahe,
    calculate_psnr,
    calculate_ssim,
    calculate_entropy,
    calculate_edge_preservation,
    estimate_noise,
)

# ===========================================================================
# MODULE-LEVEL CONSTANTS  (tune these to adjust GA behaviour)
# ===========================================================================

# ── Chromosome ranges ───────────────────────────────────────────────────────
CLIP_MIN: float = 1.0    # minimum clip limit
CLIP_MAX: float = 8.0    # maximum clip limit
GRID_MIN: int   = 4      # minimum tile grid size (pixels)
GRID_MAX: int   = 16     # maximum tile grid size (pixels)

# ── Reference-based fitness weights (must sum to 1 for the positive terms) ─
W_SSIM:     float = 0.50   # weight for Structural Similarity Index
W_PSNR:     float = 0.40   # weight for PSNR (normalised)
W_EDGE_REF: float = 0.10   # weight for edge preservation vs ground truth
W_NOISE:    float = 0.20   # penalty weight for noise amplification
PSNR_NORM:  float = 40.0   # PSNR is divided by this before clipping to [0,1]

# ── No-reference fitness weights ────────────────────────────────────────────
W_ENTROPY:  float = 0.35   # weight for normalised entropy
W_EDGE_NR:  float = 0.35   # weight for edge preservation vs input
W_BRIGHT:   float = 0.30   # weight for brightness score
# W_NOISE is shared with the reference-based formula (0.20)
NOISE_DIV:  float = 10.0   # noise sigma difference is divided by this

# ── Blend crossover alpha (BLX-α) ───────────────────────────────────────────
BLX_ALPHA:  float = 0.5    # exploration width around parent values


# ===========================================================================
# FITNESS FUNCTION
# ===========================================================================

def _compute_fitness(
    enhanced: np.ndarray,
    original: np.ndarray,
    ground_truth: Optional[np.ndarray],
) -> float:
    """
    Compute the fitness of a CLAHE-enhanced image.

    Uses reference-based metrics when ground_truth is available, otherwise
    falls back to no-reference metrics.

    Parameters
    ----------
    enhanced     : np.ndarray        — RGB uint8 HxWx3, the CLAHE output
    original     : np.ndarray        — RGB uint8 HxWx3, the low-light input
    ground_truth : np.ndarray | None — RGB uint8 HxWx3, normal-light reference

    Returns
    -------
    float — fitness value (higher is better; maximum ≈ 1.0 without noise penalty)
    """
    # ── Shared: noise penalty ────────────────────────────────────────────────
    # We penalise noise AMPLIFICATION, not the noise that was already present.
    # If CLAHE makes the image noisier, fitness is reduced.
    sigma_enh  = estimate_noise(enhanced)
    sigma_orig = estimate_noise(original)
    noise_amplification = max(0.0, sigma_enh - sigma_orig)
    noise_penalty = float(np.clip(noise_amplification / NOISE_DIV, 0.0, 1.0))

    if ground_truth is not None:
        # ── Reference-based fitness ──────────────────────────────────────────
        # Resize ground truth to match enhanced if shapes differ (happens on
        # the downscaled working image).
        gt = ground_truth
        if gt.shape != enhanced.shape:
            h, w = enhanced.shape[:2]
            gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_AREA)

        ssim_score = calculate_ssim(enhanced, gt)                        # [0, 1]
        psnr_score = float(np.clip(calculate_psnr(enhanced, gt) / PSNR_NORM, 0.0, 1.0))
        edge_score = calculate_edge_preservation(enhanced, gt)           # [-1, 1]
        edge_score = float(np.clip(edge_score, 0.0, 1.0))               # [0, 1]

        fitness = (
              W_SSIM     * ssim_score
            + W_PSNR     * psnr_score
            + W_EDGE_REF * edge_score
            - W_NOISE    * noise_penalty
        )
    else:
        # ── No-reference fitness ─────────────────────────────────────────────
        # Entropy: how much information is in the image? Normalised to 8 bits.
        entropy_score = float(np.clip(calculate_entropy(enhanced) / 8.0, 0.0, 1.0))

        # Edge preservation: how well does enhanced keep the edges from input?
        edge_score = calculate_edge_preservation(enhanced, original)
        edge_score = float(np.clip(edge_score, 0.0, 1.0))

        # Brightness score: penalise images that are too dark OR too bright.
        # Peaks at 1.0 when mean intensity = 127.5 (mid-grey), falls to 0 at
        # the extremes (0 or 255).
        mean_intensity = float(enhanced.mean()) / 255.0
        brightness_score = float(np.clip(1.0 - abs(mean_intensity - 0.5) * 2.0, 0.0, 1.0))

        fitness = (
              W_ENTROPY * entropy_score
            + W_EDGE_NR * edge_score
            + W_BRIGHT  * brightness_score
            - W_NOISE   * noise_penalty
        )

    return float(fitness)


# ===========================================================================
# GA INTERNAL HELPERS
# ===========================================================================

def _init_population(
    pop_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Randomly initialise a population of chromosomes.

    Each row is one individual: [clip_limit (float), tile_grid_size (float)].
    tile_grid_size is stored as float for uniform array dtype, but is always
    rounded to int when evaluating.

    Parameters
    ----------
    pop_size : int              — number of individuals
    rng      : np.random.Generator

    Returns
    -------
    np.ndarray of shape (pop_size, 2), dtype float64
    """
    clips = rng.uniform(CLIP_MIN, CLIP_MAX, size=pop_size)
    grids = rng.integers(GRID_MIN, GRID_MAX + 1, size=pop_size).astype(np.float64)
    return np.column_stack([clips, grids])


def _tournament_select(
    population: np.ndarray,
    fitnesses: np.ndarray,
    tournament_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Select one individual via tournament selection.

    Randomly picks `tournament_size` candidates and returns the one with
    the highest fitness. This applies selection pressure without needing
    to rank the whole population.

    Parameters
    ----------
    population     : np.ndarray  — shape (pop_size, 2)
    fitnesses      : np.ndarray  — shape (pop_size,)
    tournament_size: int         — number of competitors per tournament
    rng            : np.random.Generator

    Returns
    -------
    np.ndarray of shape (2,) — the winning chromosome
    """
    pop_size = len(population)
    # Draw `tournament_size` unique indices
    contestants = rng.choice(pop_size, size=tournament_size, replace=False)
    winner_idx  = contestants[np.argmax(fitnesses[contestants])]
    return population[winner_idx].copy()


def _crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    crossover_rate: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Produce two children from two parents.

    clip_limit    — Blend crossover (BLX-α):
        Children are sampled uniformly from the interval
        [min(p1,p2) - α·d,  max(p1,p2) + α·d]  where d = |p1 - p2|.
        This allows slight exploration beyond the parent values.

    tile_grid_size — Uniform crossover:
        Each child independently picks either parent's value with 50/50 probability.

    If a uniform random draw exceeds crossover_rate, the parents are returned
    unchanged (no crossover occurs).

    Parameters
    ----------
    parent1, parent2 : np.ndarray shape (2,)
    crossover_rate   : float
    rng              : np.random.Generator

    Returns
    -------
    child1, child2 : np.ndarray shape (2,) each
    """
    child1, child2 = parent1.copy(), parent2.copy()

    if rng.random() < crossover_rate:
        # ── BLX-α for clip_limit (index 0) ──────────────────────────────────
        c1, c2 = parent1[0], parent2[0]
        d      = abs(c1 - c2)
        lo     = min(c1, c2) - BLX_ALPHA * d
        hi     = max(c1, c2) + BLX_ALPHA * d
        child1[0] = float(np.clip(rng.uniform(lo, hi), CLIP_MIN, CLIP_MAX))
        child2[0] = float(np.clip(rng.uniform(lo, hi), CLIP_MIN, CLIP_MAX))

        # ── Uniform crossover for tile_grid_size (index 1) ──────────────────
        if rng.random() < 0.5:
            child1[1], child2[1] = parent2[1], parent1[1]

    return child1, child2


def _mutate(
    individual: np.ndarray,
    mutation_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Apply mutation to one individual (in-place, returns mutated copy).

    clip_limit    — Gaussian perturbation:
        Add N(0, σ) where σ = (CLIP_MAX - CLIP_MIN) / 6, then clip to range.
        Using σ = range/6 means ≈99.7% of perturbations stay in range before
        clipping.

    tile_grid_size — ±1 step mutation:
        Randomly add or subtract 1, then clip to [GRID_MIN, GRID_MAX].

    Parameters
    ----------
    individual     : np.ndarray shape (2,)
    mutation_rate  : float
    rng            : np.random.Generator

    Returns
    -------
    np.ndarray shape (2,) — mutated individual
    """
    ind = individual.copy()

    # Mutate clip_limit
    if rng.random() < mutation_rate:
        sigma = (CLIP_MAX - CLIP_MIN) / 6.0
        ind[0] = float(np.clip(ind[0] + rng.normal(0.0, sigma), CLIP_MIN, CLIP_MAX))

    # Mutate tile_grid_size
    if rng.random() < mutation_rate:
        step   = rng.choice([-1, 1])
        ind[1] = float(np.clip(ind[1] + step, GRID_MIN, GRID_MAX))

    return ind


def _decode(individual: np.ndarray) -> tuple[float, int]:
    """
    Convert a raw chromosome array to typed parameter values.

    Parameters
    ----------
    individual : np.ndarray shape (2,)

    Returns
    -------
    clip_limit     : float
    tile_grid_size : int
    """
    clip_limit     = float(np.clip(individual[0], CLIP_MIN, CLIP_MAX))
    tile_grid_size = int(np.clip(round(individual[1]), GRID_MIN, GRID_MAX))
    return clip_limit, tile_grid_size


def _cache_key(clip_limit: float, tile_grid_size: int) -> tuple:
    """
    Build a hashable cache key from chromosome parameters.

    clip_limit is rounded to 2 decimal places to avoid floating-point
    duplicates that differ only by tiny rounding errors.
    """
    return (round(clip_limit, 2), tile_grid_size)


def _downscale(image: np.ndarray, work_size: int) -> np.ndarray:
    """
    Downscale *image* so its longest side equals *work_size* (or less).

    If the image is already smaller, it is returned as-is to avoid upscaling.

    Parameters
    ----------
    image     : np.ndarray — RGB uint8 HxWx3
    work_size : int        — target longest side in pixels

    Returns
    -------
    np.ndarray — RGB uint8, possibly smaller than input
    """
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= work_size:
        return image  # already small enough — no scaling needed
    scale  = work_size / longest
    new_w  = max(1, int(w * scale))
    new_h  = max(1, int(h * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ===========================================================================
# PUBLIC API
# ===========================================================================

def optimize_clahe(
    image: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    pop_size: int      = 30,
    generations: int   = 30,
    crossover_rate: float = 0.80,
    mutation_rate: float  = 0.05,
    tournament_size: int  = 3,
    elitism: int       = 2,
    seed: int          = 42,
    patience: int      = 10,
    work_size: int     = 256,
) -> dict:
    """
    Find optimal CLAHE parameters using a Genetic Algorithm.

    SPEED STRATEGY
    --------------
    The GA fitness evaluations are run on a downscaled copy of the image
    (longest side = `work_size` pixels) to keep each generation fast.
    The best chromosome found is then applied to the FULL-RESOLUTION image
    at the very end, so the returned enhanced image is full-size.

    TERMINATION
    -----------
    Stops when `generations` is reached OR if the best fitness has not
    improved by more than 1e-6 for `patience` consecutive generations.

    REPRODUCIBILITY
    ---------------
    Pass the same `seed` to get identical results on the same image.
    All randomness goes through a single np.random.Generator instance.

    Parameters
    ----------
    image          : np.ndarray        — RGB uint8 HxWx3, the low-light input
    ground_truth   : np.ndarray | None — RGB uint8 HxWx3, normal-light reference.
                                         When None, a no-reference fitness is used.
    pop_size       : int   — population size (default 30)
    generations    : int   — maximum number of generations (default 30)
    crossover_rate : float — probability that crossover occurs (default 0.80)
    mutation_rate  : float — per-gene mutation probability (default 0.05)
    tournament_size: int   — competitors per tournament selection (default 3)
    elitism        : int   — top-k individuals copied unchanged (default 2)
    seed           : int   — random seed for reproducibility (default 42)
    patience       : int   — stop early if no improvement for this many
                             consecutive generations (default 10)
    work_size      : int   — longest side of the downscaled working image (default 256)

    Returns
    -------
    dict with keys:
        clip_limit     : float  — best clip limit found
        tile_grid_size : int    — best tile grid size found (pixels)
        best_fitness   : float  — fitness of the best individual
        history        : dict   — {"best": [float, ...], "mean": [float, ...]}
                                   one entry per generation run
        enhanced       : np.ndarray — full-resolution RGB uint8 enhanced image
        generations_run: int    — actual number of generations executed
        time_sec       : float  — wall-clock seconds taken
    """
    start_time = time.perf_counter()

    # ── Validate inputs ──────────────────────────────────────────────────────
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be an RGB uint8 numpy array of shape HxWx3")
    if image.dtype != np.uint8:
        raise ValueError(f"image dtype must be uint8, got {image.dtype}")

    # ── Downscale to work_size for fast fitness evaluation ───────────────────
    work_img = _downscale(image, work_size)
    work_gt  = _downscale(ground_truth, work_size) if ground_truth is not None else None

    # ── Initialise RNG and population ────────────────────────────────────────
    rng        = np.random.default_rng(seed=seed)
    population = _init_population(pop_size, rng)

    # ── Fitness evaluation with caching ─────────────────────────────────────
    # Cache avoids re-evaluating the same (clip, grid) pair in later generations.
    fitness_cache: dict[tuple, float] = {}

    def evaluate(individual: np.ndarray) -> float:
        """Evaluate fitness of one chromosome, using cache if available."""
        clip, grid = _decode(individual)
        key = _cache_key(clip, grid)
        if key not in fitness_cache:
            enhanced_work = apply_clahe(work_img, clip_limit=clip, grid_size=grid)
            fitness_cache[key] = _compute_fitness(enhanced_work, work_img, work_gt)
        return fitness_cache[key]

    # ── Evaluate initial population ──────────────────────────────────────────
    fitnesses = np.array([evaluate(ind) for ind in population])

    # ── History tracking ─────────────────────────────────────────────────────
    history_best: list[float] = []
    history_mean: list[float] = []

    best_fitness_ever   = np.max(fitnesses)
    no_improve_count    = 0
    generations_run     = 0

    # ── Main GA loop ─────────────────────────────────────────────────────────
    for gen in range(generations):
        generations_run = gen + 1

        # Record this generation's stats BEFORE creating the next generation
        history_best.append(float(np.max(fitnesses)))
        history_mean.append(float(np.mean(fitnesses)))

        # ── Elitism: preserve top-k individuals ─────────────────────────────
        elite_indices = np.argsort(fitnesses)[::-1][:elitism]
        elites        = population[elite_indices].copy()

        # ── Build next generation through selection, crossover, mutation ─────
        new_population: list[np.ndarray] = list(elites)

        while len(new_population) < pop_size:
            # Tournament selection for two parents
            p1 = _tournament_select(population, fitnesses, tournament_size, rng)
            p2 = _tournament_select(population, fitnesses, tournament_size, rng)

            # Crossover
            c1, c2 = _crossover(p1, p2, crossover_rate, rng)

            # Mutation
            c1 = _mutate(c1, mutation_rate, rng)
            c2 = _mutate(c2, mutation_rate, rng)

            new_population.append(c1)
            if len(new_population) < pop_size:
                new_population.append(c2)

        population = np.array(new_population)
        fitnesses  = np.array([evaluate(ind) for ind in population])

        # ── Early stopping (patience) ────────────────────────────────────────
        current_best = float(np.max(fitnesses))
        if current_best > best_fitness_ever + 1e-6:
            best_fitness_ever = current_best
            no_improve_count  = 0
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            # Stopped early — record last generation and break
            history_best.append(float(np.max(fitnesses)))
            history_mean.append(float(np.mean(fitnesses)))
            break

    # ── Extract best chromosome ──────────────────────────────────────────────
    best_idx          = int(np.argmax(fitnesses))
    best_clip, best_grid = _decode(population[best_idx])
    final_fitness     = float(fitnesses[best_idx])

    # ── Apply best params to FULL-RESOLUTION image ───────────────────────────
    enhanced_full = apply_clahe(image, clip_limit=best_clip, grid_size=best_grid)

    elapsed = time.perf_counter() - start_time

    return {
        "clip_limit":      best_clip,
        "tile_grid_size":  best_grid,
        "best_fitness":    final_fitness,
        "history": {
            "best": history_best,
            "mean": history_mean,
        },
        "enhanced":        enhanced_full,
        "generations_run": generations_run,
        "time_sec":        elapsed,
    }


# ===========================================================================
# DEMO / MAIN
# ===========================================================================

if __name__ == "__main__":
    """
    Quick demo that:
      1. Loads an image from argv[1] (or generates a synthetic dark image).
      2. Runs the GA twice with the same seed — confirms identical results.
      3. Saves a before/after PNG and a fitness-history plot to results/figures/.
      4. Verifies:
           (a) Same seed → identical results.
           (b) Best fitness history is non-decreasing (elitism works).
           (c) Best parameters stay within defined ranges.

    Usage:
        python -m optimization.genetic_algorithm              (synthetic image)
        python -m optimization.genetic_algorithm path/to/img  (real image)
    """
    import matplotlib
    matplotlib.use("Agg")  # headless backend — no display needed
    import matplotlib.pyplot as plt

    print("=" * 65)
    print("  optimization/genetic_algorithm.py  —  DEMO")
    print("=" * 65)

    # ── Load or generate image ───────────────────────────────────────────────
    if len(sys.argv) > 1:
        img_path = Path(sys.argv[1])
        raw = cv2.imread(str(img_path))
        if raw is None:
            print(f"[ERROR] Could not read '{img_path}'. Using synthetic image.")
            img_path = None
        else:
            image = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            print(f"[IMAGE] Loaded '{img_path}'  shape={image.shape}")
    else:
        img_path = None

    if img_path is None or len(sys.argv) == 1:
        # Synthetic dark image: 128×128, low intensity, slight spatial texture
        rng_img = np.random.default_rng(seed=0)
        base    = rng_img.integers(10, 55, size=(128, 128, 3), dtype=np.uint8)
        noise   = rng_img.integers(0, 12, size=(128, 128, 3), dtype=np.uint8)
        image   = np.clip(base.astype(np.int32) + noise, 0, 255).astype(np.uint8)
        print(f"[IMAGE] Using synthetic dark image  shape={image.shape}  "
              f"mean={image.mean():.1f}")

    # ── Run GA — first pass ──────────────────────────────────────────────────
    print("\n[GA RUN 1] Running with seed=42 …")
    result1 = optimize_clahe(image, ground_truth=None, seed=42,
                             pop_size=30, generations=30)

    print(f"  Best clip_limit    : {result1['clip_limit']:.4f}")
    print(f"  Best tile_grid_size: {result1['tile_grid_size']}")
    print(f"  Best fitness       : {result1['best_fitness']:.6f}")
    print(f"  Generations run    : {result1['generations_run']}")
    print(f"  Time               : {result1['time_sec']:.2f}s")

    # ── Run GA — second pass (same seed → must be identical) ─────────────────
    print("\n[GA RUN 2] Same seed=42, verifying reproducibility …")
    result2 = optimize_clahe(image, ground_truth=None, seed=42,
                             pop_size=30, generations=30)

    # ── Verification (a): reproducibility ────────────────────────────────────
    assert result1["clip_limit"]     == result2["clip_limit"],     "clip_limit mismatch!"
    assert result1["tile_grid_size"] == result2["tile_grid_size"], "tile_grid_size mismatch!"
    assert result1["best_fitness"]   == result2["best_fitness"],   "fitness mismatch!"
    assert result1["history"]["best"] == result2["history"]["best"], "history mismatch!"
    print("  ✓ PASS  Results are IDENTICAL for the same seed.")

    # ── Verification (b): non-decreasing best history (elitism) ─────────────
    best_hist = result1["history"]["best"]
    for i in range(1, len(best_hist)):
        assert best_hist[i] >= best_hist[i - 1] - 1e-9, (
            f"Best fitness decreased at generation {i}: "
            f"{best_hist[i-1]:.6f} → {best_hist[i]:.6f}"
        )
    print(f"  ✓ PASS  Best-fitness history is non-decreasing across "
          f"{len(best_hist)} generations.")

    # ── Verification (c): parameters in range ────────────────────────────────
    assert CLIP_MIN <= result1["clip_limit"]     <= CLIP_MAX, "clip_limit out of range!"
    assert GRID_MIN <= result1["tile_grid_size"] <= GRID_MAX, "tile_grid_size out of range!"
    print(f"  ✓ PASS  clip_limit={result1['clip_limit']:.4f} ∈ "
          f"[{CLIP_MIN}, {CLIP_MAX}]  |  "
          f"tile_grid_size={result1['tile_grid_size']} ∈ [{GRID_MIN}, {GRID_MAX}]")

    # ── Save before/after image ──────────────────────────────────────────────
    out_dir = Path("results") / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(image)
    axes[0].set_title(f"Original (mean={image.mean():.1f})", fontsize=11)
    axes[0].axis("off")

    enhanced_full = result1["enhanced"]
    axes[1].imshow(enhanced_full)
    axes[1].set_title(
        f"GA-CLAHE  clip={result1['clip_limit']:.2f}  "
        f"grid={result1['tile_grid_size']}\n"
        f"fitness={result1['best_fitness']:.4f}  "
        f"mean={enhanced_full.mean():.1f}",
        fontsize=10,
    )
    axes[1].axis("off")

    fig.suptitle("Genetic Algorithm CLAHE Optimisation — Before / After", fontsize=12)
    fig.tight_layout()
    before_after_path = out_dir / "ga_demo_before_after.png"
    fig.savefig(before_after_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[SAVED]  Before/after image → {before_after_path}")

    # ── Save fitness-history plot ────────────────────────────────────────────
    gens    = list(range(1, len(best_hist) + 1))
    mean_h  = result1["history"]["mean"]

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(gens, best_hist, "b-o", markersize=4, label="Best fitness")
    ax2.plot(gens, mean_h,   "r--s", markersize=3, label="Mean fitness")
    ax2.set_xlabel("Generation", fontsize=11)
    ax2.set_ylabel("Fitness", fontsize=11)
    ax2.set_title(
        f"GA Convergence  |  pop={30}  gens_run={result1['generations_run']}  "
        f"seed=42",
        fontsize=11,
    )
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    history_path = out_dir / "ga_demo_fitness_history.png"
    fig2.savefig(history_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"[SAVED]  Fitness history plot → {history_path}")

    print("\n[ALL CHECKS PASSED]")
    print("=" * 65)
