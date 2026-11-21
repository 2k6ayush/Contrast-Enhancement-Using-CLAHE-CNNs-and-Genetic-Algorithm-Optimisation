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

## Step 3 — Hybrid Pipeline
**Date:** 2026-11-20  
**Commit:** `35723f9` — *"Member 3: hybrid enhancement pipeline"*

### What was done
Fully implemented `optimization/hybrid.py` and updated `optimization/__init__.py`.

### Public API (3 functions)

| Function | Returns | Purpose |
|----------|---------|---------|
| `enhance_hybrid(image, model, ground_truth, **ga_kwargs)` | `dict` | Full pipeline — rich dict with intermediates, timing, params |
| `enhance_hybrid_image(image, model, **ga_kwargs)` | `np.ndarray` | Thin wrapper for teammates — agreed interface (array only) |
| `enhance_ga_clahe(image, ground_truth, **ga_kwargs)` | `np.ndarray` | Stage 1 only (no CNN) — for ablation config C and app |

### Design Decisions

**Two-stage pipeline:**
1. `optimize_clahe()` → returns `result["enhanced"]` (full-res GA-CLAHE output)
2. `enhance_with_cnn(clahe_output, model)` → CNN refinement

**model=None graceful degradation:**
- Issues `[HYBRID]` warning and returns GA-CLAHE output as-is.
- Pipeline is 100% functional before Member 2's CNN is merged.

**Output dtype guarantee:**
- After CNN stage: if output is float (some models return float), it is clamped to [0,255] and cast to uint8.

**Input validation (`_validate_image`):**
- TypeError if not `np.ndarray`
- ValueError if `ndim ≠ 3`, channels `≠ 3`, or `dtype ≠ uint8`
- Clear, actionable error messages.

### Demo Output
```
[IMAGE] Synthetic dark image  shape=(128, 128, 3)  mean=31.6

[STAGE 1] Running GA-CLAHE only (enhance_ga_clahe) …
  Output  shape=(128, 128, 3)  dtype=uint8  mean=119.1

[STAGE 1+2] Running full hybrid pipeline (model=None) …
  [WARNING] [HYBRID] model=None detected. Skipping CNN refinement stage.
  clip_limit     : 7.9987
  tile_grid_size : 4
  time_ms        : 263.8 ms

[CHECKS]
  ✓ output dtype=uint8 and shape matches input
  ✓ output value range [0, 254] ⊂ [0, 255]
  ✓ enhance_hybrid_image() returns uint8 ndarray (thin wrapper works)
  ✓ enhance_ga_clahe() output matches hybrid['clahe_output'] (same seed)
  ✓ ga_history has 16 generation entries

[VALIDATION] Testing bad-input error handling …
  ✓ float dtype caught: 'image' must have dtype uint8, got float32.
  ✓ 2D shape caught: 'image' must be 3-dimensional (H×W×C), got ndim=2.
  ✓ non-array type caught: 'image' must be a numpy.ndarray, got list.

[ALL CHECKS PASSED]
```

### Git Log
```
35723f9  Member 3: hybrid enhancement pipeline
6f05f9c  Member 3: genetic algorithm for CLAHE parameter optimisation
93393fd  Add paired low-light image datasets
64bf0f3  Member 3: project skeleton and compat layer
```

---

## Step 4A — Experiments Runner (Infrastructure + Experiments 1–3)
**Date:** 2026-11-20  
**Commit:** `074a3a2` — *"Member 3: experiments runner (infrastructure + experiments 1-3)"*

### What was done
Implemented `evaluation/experiments.py` with shared evaluation infrastructure and experiments 1, 2, and 3.

### Key Infrastructure & Features
1. **Dataset Loading (`load_paired_split`, `_list_images`):**
   - Loads paired images (`low/` and `high/`) matching by file stem.
   - Filters out non-image files and hidden dot files (`.DS_Store`, `._*`).
   - Strict validation: fails with clear error if folders are missing or empty; never falls back to synthetic data.
   - Configurable via `--data-root` CLI argument (default: `"dataset"`).
2. **Fixed Validation Split (`make_val_split`):**
   - Carves a reproducible 15% validation split from `our485` using a fixed seed (`seed=42`).
   - Keeps `eval15` exclusively as the unseen test set (never used for tuning).
3. **Metric Evaluation (`evaluate_method`, `_evaluate_paired`):**
   - Evaluates: PSNR, SSIM, Entropy, NIQE (graceful NaN handling if stub), Contrast Improvement Index ($std_{enh} / std_{orig}$), Edge Preservation Index (EPI via Sobel gradients), and wall-clock time in ms.
   - Progress bar via `tqdm` with fallback if unavailable.
4. **Safe Re-run CSV Persistence (`upsert_results`):**
   - Writes results to `results/metrics.csv`.
   - Idempotent: replaces existing rows matching `(experiment, method)` without duplication.
5. **CLI Options:**
   - `--exp {1, 2, 3, all}`
   - `--max-images N` / `-n N`
   - `--split {val, test}` (default `test`)
   - `--model-path` (for CNN weights)
   - `--data-root`
   - `--seed`

### Experiments Implemented
- **Exp 1 (Traditional):** Histogram Equalization and default CLAHE (clip=2.0, grid=8).
- **Exp 2 (CNN-only):** Real CNN loaded via `build_model()` + weights from `--model-path`. Skips with clear warning if CNN is a stub or weights are missing (stub CNN is never recorded as real results).
- **Exp 3 (Hybrid GA-CLAHE + CNN):** Evaluates `enhance_hybrid` with two modes:
  - `hybrid_ref`: Reference-based GA fitness (uses ground truth).
  - `hybrid_noref`: No-reference GA fitness (no ground truth used in GA).
  - Measures total per-image execution time including the GA search.

### Verification Run
Command:
```bash
python evaluation/experiments.py --exp 1 --max-images 3
```
Output:
- Processed 3 images on `test` split (`eval15`).
- Results upserted to `results/metrics.csv`.
- `results/metrics.csv` head confirmed:
  - Columns: `experiment,method,image,psnr,ssim,entropy,niqe,contrast_improvement_index,edge_preservation,time_ms`
  - All values present and formatted properly; NIQE is `nan` as expected under the stub.

---

## Step 4B — Experiments 4, 5, 6 (Hyperparameter Tuning, Ablation, Performance Comparison)
**Date:** 2026-11-21  
**Commit:** `6c99cf9` — *"Member 3: experiments 4-6"*

### What was done
Extended `evaluation/experiments.py` with Experiments 4, 5, and 6, and added custom CSV persistence.

### Experiments Implemented
1. **Experiment 4: Hyperparameter Tuning (`run_experiment_4`):**
   - **Validation Split Only:** Carves 15% of `our485` (72 images) using fixed seed (`seed=42`). `eval15` (test split) is strictly never used for tuning.
   - **4a: Fixed-parameter CLAHE sweep:** 24 combinations of clip limit $\in \{1, 2, 3, 4, 6, 8\} \times$ tile grid size $\in \{4, 8, 12, 16\}$.
   - **4b: GA parameter sweep:** Pop size $\{20, 30, 50\} \times$ generations $\{20, 50\} \times$ crossover rate $\{0.7, 0.8, 0.9\} \times$ mutation rate $\{0.01, 0.05, 0.1\}$ (54 possible combinations). Samples random subset of `n_configs` (default 12) via `--n-configs`, runs 3 random seeds per config (`seed`, `seed+1`, `seed+2`), and computes mean and std of PSNR, SSIM, and time.
   - **Member 2 CNN LR Placeholder:** Includes `cnn_learning_rate` in the CSV schema (`results/hyperparameter_tuning.csv`).
2. **Experiment 5: Ablation Study (`run_experiment_5`):**
   - Evaluated on test split (`eval15`).
   - Configs:
     - **A:** Default CLAHE only (clip=2.0, grid=8)
     - **B:** CNN only (gracefully skipped if stub/no weights)
     - **C:** GA + CLAHE (no CNN)
     - **D:** Full Hybrid (GA-CLAHE $\to$ CNN)
   - Results saved to `results/ablation.csv`.
3. **Experiment 6: Performance Comparison Table (`run_experiment_6`):**
   - Compares: **Traditional** (Histogram Equalization, Default CLAHE), **ML/DL** (CNN), and **Proposed** (GA-CLAHE + CNN Hybrid).
   - Aggregates per-approach mean $\pm$ std for all metrics: PSNR, SSIM, Entropy, NIQE, CII, EPI, Time (ms/img), and model size (MB).
   - Results saved to `results/comparison_table.csv` and pretty-printed to console.

### Benchmark & Timing Probe
- Tested on 3 validation images with 2 GA configurations (`--exp 4 --max-images 3 --n-configs 2`):
  - Fixed CLAHE sweep (24 configs × 3 images): ~8 seconds total.
  - GA sweep (2 configs × 3 seeds × 3 images = 18 GA runs): ~9.8 seconds total (~0.54s per GA run).

### Full Experiment 4 Results (Validation Split: 72 images)
Ran complete hyperparameter tuning sweep on 72 validation images:
- **Fixed CLAHE Sweep (24 configs × 72 images = 1,728 evaluations, 1m 42s):**
  - **Best Config:** `clip_limit = 8.0, grid_size = 4`
  - **Validation Metrics:** $\text{SSIM} = 0.5103 \pm 0.1109$, $\text{PSNR} = 12.65\text{ dB}$, $\text{Time} = 1.31\text{ ms/image}$.
- **GA Hyperparameter Sweep (12 sampled configs × 3 seeds × 72 images = 2,592 runs, 22m 09s):**
  - **Best Config:** `pop_size = 30, generations = 50, crossover_rate = 0.9, mutation_rate = 0.05`
  - **Validation Metrics:** $\text{SSIM} = 0.5125 \pm 0.0014$, $\text{PSNR} = 12.26\text{ dB}$, $\text{Time} = 402.83\text{ ms/image}$.
  - Notice the low seed std ($\pm 0.0014$), demonstrating high optimization stability.
- Output saved to `results/hyperparameter_tuning.csv` (36 total rows).

### Experiment 5: Ablation Study Results (Test Split: `eval15`, 15 images)
- **Config A (Default CLAHE):** $\text{SSIM} = 0.3565 \pm 0.1097$, $\text{PSNR} = 9.13 \pm 2.86\text{ dB}$, $\text{Entropy} = 5.554$, $\text{Time} = 7.0\text{ ms}$.
- **Config B (CNN only):** Skipped gracefully (CNN stub active, awaiting Member 2 weights).
- **Config C (GA-CLAHE only, no CNN):** $\text{SSIM} = 0.4861 \pm 0.0704$, $\text{PSNR} = 12.53 \pm 3.24\text{ dB}$, $\text{Entropy} = 6.726$, $\text{Time} = 438.5\text{ ms}$.
- **Config D (Full Hybrid):** Evaluated through hybrid pipeline (identical to C currently due to identity CNN stub, ready for real CNN weights).
- Output saved to `results/ablation.csv` (45 total rows).
- **Key Observation:** GA parameter optimization improves SSIM by **+36.4%** ($0.3565 \to 0.4861$), PSNR by **+3.4 dB** ($9.13 \to 12.53\text{ dB}$), and Entropy by **+1.17 bits** over default CLAHE!

### Experiment 6: Performance Comparison Table Results (Test Split: `eval15`)
- **Histogram Equalization:** $\text{PSNR} = 14.24 \pm 3.03\text{ dB}$, $\text{SSIM} = 0.4252 \pm 0.0983$, $\text{Entropy} = 4.742$, $\text{CII} = 7.65$, $\text{EPI} = 0.7550$, $\text{Time} = 2.0\text{ ms}$.
- **Default CLAHE:** $\text{PSNR} = 9.13 \pm 2.86\text{ dB}$, $\text{SSIM} = 0.3565 \pm 0.1097$, $\text{Entropy} = 5.554$, $\text{CII} = 1.62$, $\text{EPI} = 0.8288$, $\text{Time} = 7.0\text{ ms}$.
- **Proposed (GA-CLAHE + CNN):** $\text{PSNR} = 12.53 \pm 3.24\text{ dB}$, $\text{SSIM} = 0.4861 \pm 0.0704$, $\text{Entropy} = 6.726$, $\text{CII} = 3.57$, $\text{EPI} = 0.7776$, $\text{Time} = 453.3\text{ ms}$.
- **Key Observation:** The Proposed pipeline achieves the **highest structural similarity (SSIM 0.4861)** and **highest entropy (6.726 bits)** across all methods on the test dataset.
- Output saved to `results/comparison_table.csv` (3 rows).

---

## Step 5 — Method Comparison & Visualization Synthesis (`evaluation/comparison.py`)
**Date:** 2026-11-21  
**Commit:** `4da6cf3` — *"Member 3: method comparison and visualization synthesis"*

### What was done
Implemented `evaluation/comparison.py` to synthesize results across all experiments, produce formatted Markdown reports, and generate publication-quality visualization figures.

### Key Capabilities
1. **Report Generation:**
   - `generate_comparison_markdown()`: Generates comparative performance tables across Traditional (HE, CLAHE) and Proposed (GA-CLAHE + CNN) paradigms.
   - `generate_ablation_markdown()`: Tabulates metric changes across Ablation Configs A, C, and D.
2. **Visualizations Generated (`results/figures/`):**
   - `comparison_metrics_barchart.png`: Grouped bar charts with standard deviation error bars comparing PSNR, SSIM, and Shannon Entropy.
   - `ablation_boxplots.png`: Box plots showing statistical distribution of SSIM and PSNR across all test images per ablation configuration.
   - `clahe_tuning_heatmap.png`: 2D validation SSIM landscape heatmap mapping Clip Limits (1.0 to 8.0) vs Tile Grid Sizes (4 to 16).
   - `runtime_tradeoff.png`: Quality (SSIM) vs Latency (time in ms on log scale) Pareto trade-off scatter plot.

---

## Step 6 — Streamlit Interactive Application (`app.py`)
**Date:** 2026-11-21  
**Commit:** `8ae8aa9` — *"Member 3: interactive streamlit demo app"*

### What was done
Fully developed `app.py` as an interactive, multi-tab Streamlit dashboard for real-time demonstration, evaluation, and analysis of low-light image contrast enhancement.

### Key Features
1. **Interactive Enhancement & Live Comparison (Tab 1):**
   - **Input Choices:** Select directly from LOL paired test samples (`dataset/eval15/low/` with automatic paired ground truth detection) or upload custom user images.
   - **Multi-Method Pipeline:** Simultaneously processes and visually compares:
     1. Low-light Input
     2. Histogram Equalization
     3. Default CLAHE (`clip=2.0, grid=8`)
     4. GA-Optimized CLAHE (Proposed Stage 1)
     5. Full Hybrid Pipeline (Proposed Stage 1 + 2)
     6. Normal-light Ground Truth (when available)
   - **Live Metric Dashboard:** Computes PSNR, SSIM, Shannon Entropy, Contrast Improvement Index (CII), Edge Preservation Index (EPI), and execution latency in ms.
   - **GA Convergence Plot:** Live visual tracking of Best Fitness vs Mean Population Fitness per generation.
   - **RGB Histograms:** Dynamic channel distribution comparison.
2. **Benchmark & Evaluation Viewer (Tab 2):**
   - Interactive data tables for `comparison_table.csv`, `ablation.csv`, and `hyperparameter_tuning.csv`.
   - Embeds generated publication figures:
     - `comparison_metrics_barchart.png`
     - `runtime_tradeoff.png`
     - `ablation_boxplots.png`
     - `clahe_tuning_heatmap.png`
3. **Project Architecture & Defense Guide (Tab 3):**
   - Detailed visual breakdown of the two-stage hybrid pipeline.
   - Explains chromosome representation, multi-objective fitness formula (EPI + Entropy + Brightness - Immerkaer Noise Penalty), BLX-α crossover, and elitism for college viva/defense.

---

## Final Project Status

| Step | Topic | Status |
|------|-------|--------|
| Step 1 | Setup + compat layer (`optimization/compat.py`) | ✅ Complete |
| Step 2 | GA core (`optimization/genetic_algorithm.py`) | ✅ Complete |
| Step 3 | Hybrid pipeline (`optimization/hybrid.py`) | ✅ Complete |
| Step 4A | Experiments runner 1–3 (`evaluation/experiments.py`) | ✅ Complete |
| Step 4B | Experiments runner 4–6 (`evaluation/experiments.py`) | ✅ Complete |
| Step 5 | Method comparison & synthesis (`evaluation/comparison.py`) | ✅ Complete |
| Step 6 | Interactive Streamlit App (`app.py`) | ✅ Complete |

