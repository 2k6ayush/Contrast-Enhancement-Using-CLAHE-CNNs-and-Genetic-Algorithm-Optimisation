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

## Upcoming Steps

| Step | Topic | Status |
|------|-------|--------|
| Step 1 | Setup + compat layer | ✅ Done |
| Step 2 | GA core (`genetic_algorithm.py`) | ⏳ Pending |
| Step 3 | Hybrid pipeline (`hybrid.py`) | ⏳ Pending |
| Step 4 | Streamlit app (`app.py`) | ⏳ Pending |
| Step 5 | Experiments + comparison | ⏳ Pending |
