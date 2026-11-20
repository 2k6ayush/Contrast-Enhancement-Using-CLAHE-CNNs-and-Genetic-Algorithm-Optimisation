# Ayush (Member 3) — Work Log
> Project: Contrast Enhancement Using CLAHE, CNNs and Genetic Algorithm Optimisation  
> Role: Genetic Algorithm + Integration  
> Branch: `member-3-ga-app`  
> Evaluation: 24 Nov 2026

---

## Rules Summary (always follow)

| Rule | Detail |
|------|--------|
| Branch | Work only on `member-3-ga-app`. Never commit to `main`. No force push. |
| My files | `optimization/` (all), `app.py`, `evaluation/experiments.py`, `evaluation/comparison.py` |
| Off-limits | `traditional/`, `cnn/`, `requirements.txt`, `README.md`, `.gitignore` |
| No commit | datasets, `*.pth`/`*.pt`, venvs, `__pycache__` |
| Interfaces | NumPy RGB uint8 HxWx3 at all boundaries. BGR/LAB internal only. |
| My exports | `optimize_clahe(image, ground_truth=None)`, `enhance_hybrid(image, model)` |

---

## Step 1 — Setup & Compat Layer
**Date:** 2026-11-20  
**Commit:** `64bf0f3` — *"Member 3: project skeleton and compat layer"*

### What was done
1. **Repo state check** — repo was on `main` with only `README.md` and `.gitattributes`. Clean, nothing to pull.
2. **Branch created** — `git switch -c member-3-ga-app`, pushed with `-u origin member-3-ga-app`.
3. **Directory structure created:**
   ```
   optimization/
     __init__.py
     genetic_algorithm.py   (skeleton — TODO Step 2)
     hybrid.py              (skeleton — TODO Step 3)
     compat.py              ← FULLY IMPLEMENTED
   evaluation/
     experiments.py         (skeleton — TODO Step 5)
     comparison.py          (skeleton — TODO Step 5)
   app.py                   (skeleton — TODO Step 4)
   results/figures/.gitkeep
   ```

### optimization/compat.py — Design Decisions

**Purpose:** Acts as an insulation layer between Member 3's code and teammates' code. If a teammate's module isn't merged yet, the stub silently takes over and prints a loud `[COMPAT STUB]` warning so we always know.

**How it works:**
- For each function, `importlib.import_module()` tries to load the real module.
- If it succeeds → real function is used (`_STUB_STATUS[name] = False`).
- If it fails → a fallback stub is defined (`_STUB_STATUS[name] = True`).
- `using_stubs()` returns the full status dict at any time.

**Fallback implementations:**

| Function | Real source | Fallback strategy |
|----------|-------------|-------------------|
| `apply_histogram_equalization` | `traditional.traditional` | OpenCV `equalizeHist` on Y channel (YCrCb) |
| `apply_clahe` | `traditional.traditional` | OpenCV CLAHE on L channel (LAB) |
| `calculate_psnr` | `traditional.metrics` | `skimage.metrics.peak_signal_noise_ratio` |
| `calculate_ssim` | `traditional.metrics` | `skimage.metrics.structural_similarity` |
| `calculate_entropy` | `traditional.metrics` | NumPy histogram + Shannon formula |
| `calculate_niqe` | `traditional.metrics` | Returns `NaN` (needs real model) |
| `build_model` | `cnn.model` | Returns `None` placeholder |
| `enhance_with_cnn` | `cnn.inference` | Identity pass-through (returns input unchanged) |

**Always-real helpers (Member 3 owns these):**

| Function | Algorithm |
|----------|-----------|
| `calculate_edge_preservation(enhanced, reference)` | Pearson correlation of Sobel gradient magnitudes |
| `estimate_noise(image)` | Immerkaer's fast noise sigma estimation (Laplacian convolution) |

### Smoke Test Output
```
=================================================================
  optimization/compat.py  —  SMOKE TEST
=================================================================

[INPUT]  dark_img  shape=(50, 50, 3)  dtype=uint8  mean=38.8
[INPUT]  bright_img shape=(50, 50, 3)  dtype=uint8  mean=138.8

[HE]      output shape=(50, 50, 3)  mean=131.1
[CLAHE]   output shape=(50, 50, 3)  mean=93.1
[PSNR]    CLAHE vs bright = 13.03 dB
[SSIM]    CLAHE vs bright = 0.4028
[ENTROPY] dark=5.280 bits  clahe=7.315 bits
[NIQE]    value = nan  (NaN expected until real module merged)
[CNN]     model=None  output shape=(50, 50, 3)  mean=93.1
[EDGE PRES] EPI = 0.9371  (Pearson corr of Sobel gradients)
[NOISE EST] dark sigma=9.520  clahe sigma=43.066

--- Stub Status Report ---
  STUB ⚠  apply_histogram_equalization
  STUB ⚠  apply_clahe
  STUB ⚠  calculate_psnr
  STUB ⚠  calculate_ssim
  STUB ⚠  calculate_entropy
  STUB ⚠  calculate_niqe
  STUB ⚠  build_model
  STUB ⚠  enhance_with_cnn

[SMOKE TEST PASSED] All functions ran without exceptions.
=================================================================
```
> All stubs active (expected — teammates' code not merged yet).  
> CLAHE increases entropy from 5.28 → 7.31 bits ✓  
> Edge preservation EPI = 0.94 (high, expected for similar synthetic images) ✓  
> NIQE = NaN (expected stub behaviour) ✓

### Git Log
```
64bf0f3 Member 3: project skeleton and compat layer
219af2b updated Readme file
7becfdb Initial commit
```

### Notes for teammates
- **requirements.txt** — please add these packages (coordinate with team):
  ```
  opencv-python>=4.8.0
  numpy>=1.24.0
  scikit-image>=0.21.0
  scipy>=1.11.0
  streamlit>=1.28.0
  ```
- `__pycache__/` is untracked — ensure `.gitignore` has `**/__pycache__/` and `*.pyc`.

---

## Step 2 — Genetic Algorithm Core
**Date:** 2026-11-20  
**Commit:** `6f05f9c` — *"Member 3: genetic algorithm for CLAHE parameter optimisation"*

### What was done
Fully implemented `optimization/genetic_algorithm.py` replacing the Step 1 skeleton.

### Design Decisions

**Chromosome encoding:**
- Index 0 → `clip_limit` (float ∈ [1.0, 8.0])
- Index 1 → `tile_grid_size` (stored as float, cast to int on decode, ∈ [4, 16])
- Using float64 array for uniform dtype — grid size is rounded on `_decode()`

**Speed strategy (downscaling):**
- GA evaluations run on a downscaled copy (`work_size=256` longest side) — fast!
- Best parameters are applied to the **full-resolution** image at the end only once.
- `_downscale()` skips scaling if the image is already small enough.

**Fitness cache:**
- `dict[tuple, float]` keyed on `(round(clip, 2), int(grid))`.
- Avoids re-evaluating chromosomes that round to the same params across generations.

**GA operators:**
| Operator | Clip limit | Tile grid size |
|----------|-----------|----------------|
| Crossover | BLX-α (blend, α=0.5) — allows slight exploration beyond parent range | Uniform crossover — randomly picks either parent's value |
| Mutation | Gaussian N(0, range/6) + clip | ±1 step + clip |
| Selection | Tournament (k=3 by default) — simple, controllable selection pressure |
| Elitism | Top-k copied unchanged — guarantees non-decreasing best fitness |

**Fitness function (module-level weight constants for easy tuning):**
```python
# Reference-based (ground truth available):
fitness = 0.50*SSIM + 0.40*clip(PSNR/40, 0,1) + 0.10*edge_pres - 0.20*noise_pen

# No-reference (no ground truth):
fitness = 0.35*(entropy/8) + 0.35*edge_pres + 0.30*brightness_score - 0.20*noise_pen

# noise_penalty = clip(max(0, σ(enhanced) - σ(input)) / 10, 0, 1)
# brightness_score = clip(1 - |mean/255 - 0.5|*2, 0, 1)
```

**Early stopping:** `patience=10` — stops if no improvement > 1e-6 for 10 consecutive generations.

### Demo Output
```
=================================================================
  optimization/genetic_algorithm.py  —  DEMO
=================================================================
[IMAGE] Using synthetic dark image  shape=(128, 128, 3)  mean=37.5

[GA RUN 1] Running with seed=42 …
  Best clip_limit    : 7.9987
  Best tile_grid_size: 4
  Best fitness       : 0.752260
  Generations run    : 15        ← stopped early (patience=10)
  Time               : 0.41s

[GA RUN 2] Same seed=42, verifying reproducibility …
  ✓ PASS  Results are IDENTICAL for the same seed.
  ✓ PASS  Best-fitness history is non-decreasing across 16 generations.
  ✓ PASS  clip_limit=7.9987 ∈ [1.0, 8.0]  |  tile_grid_size=4 ∈ [4, 16]

[ALL CHECKS PASSED]
```

### Output files (not committed — generated artifacts)
- `results/figures/ga_demo_before_after.png`
- `results/figures/ga_demo_fitness_history.png`

### Git Log
```
6f05f9c  Member 3: genetic algorithm for CLAHE parameter optimisation
93393fd  Add paired low-light image datasets   ← teammate's commit on main
64bf0f3  Member 3: project skeleton and compat layer
```

---

## Upcoming Steps

| Step | Topic | Status |
|------|-------|--------|
| Step 1 | Setup + compat layer | ✅ Done |
| Step 2 | GA core (`genetic_algorithm.py`) | ✅ Done |
| Step 3 | Hybrid pipeline (`hybrid.py`) | ✅ Done |
| Step 4 | Streamlit app (`app.py`) | ⏳ Pending |
| Step 5 | Experiments + comparison | ⏳ Pending |
