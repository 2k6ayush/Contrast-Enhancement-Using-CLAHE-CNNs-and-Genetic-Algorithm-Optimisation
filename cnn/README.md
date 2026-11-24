# CNN training and inference

This package implements one compact U-Net architecture for two deliberately separate experiments:

- `baseline`: raw low-light RGB to normal-light RGB.
- `ga_clahe_refinement`: reference-free GA-CLAHE RGB to normal-light RGB.

The default model has two encoder levels, a bottleneck, bilinear decoder upsampling, and skip connections. Group normalization makes small batches practical. Inputs are internally padded to a multiple of four and cropped back, so odd image sizes are preserved. The final sigmoid produces values in `[0, 1]`. With `base_channels=32`, the model has **436,515 trainable parameters**.

## Data isolation

`Dataset/our485` is the only training/validation source. The first training run creates `splits/our485_split.json`; all later baseline and refinement runs reuse it. Files are paired by filename, decoded and dimension-checked. Exact duplicate low or high files are detected by SHA-256 and their complete duplicate group stays in one split. Paired crops and flips share the same random choices. Validation and `Dataset/eval15` are evaluated at original resolution. `eval15` is final-test-only.

## Local commands

From the repository root:

```powershell
python -m cnn.train --input-mode baseline --dataset-root Dataset --output-dir models/baseline
python -m cnn.train --input-mode ga_clahe_refinement --dataset-root Dataset --output-dir models/refinement
```

The hybrid command runs the existing genetic algorithm with `ground_truth=None` and caches its outputs plus exact settings below `cache/ga_clahe_our485`. This preprocessing is intentionally expensive. It is never silently replaced by fixed CLAHE. Use `--rebuild-hybrid-cache` only when intentionally replacing a cache after settings change.

Defaults are batch size 8, Adam at `1e-3`, 50 epochs, 256-pixel crops, seed 42, ReduceLROnPlateau scheduling, and CUDA AMP when available. These are starting settings, not claimed optima. `best.pt`, `latest.pt`, `history.csv`, `training_history.png`, and `run_config.json` are written to the output directory. Resume with `--resume models/baseline/latest.pt` and keep the same `--input-mode`.

Every accepted checkpoint uses `clahe-cnn-v1` and contains model weights, architecture, input-mode and preprocessing metadata. Resume checkpoints additionally contain optimizer, scheduler, epoch, best validation SSIM, and random states. Raw state dictionaries are rejected so an untrained or mismatched model cannot silently appear as a trained enhancement model.

Final testing (not tuning):

```powershell
python -m cnn.evaluate --dataset-root Dataset `
  --baseline-checkpoint models/baseline/best.pt `
  --refinement-checkpoint models/refinement/best.pt
```

This writes new, provenance-rich results below `results/cnn_evaluation`; it does not overwrite teammates' legacy CSVs. The fixed-CLAHE+CNN row is explicitly a fixed-weight input-preprocessing ablation. Retraining for fixed CLAHE would be a different experiment.

Inference accepts only RGB `uint8` NumPy arrays shaped `H x W x 3`. Load checkpoints with `cnn.inference.load_checkpoint`, then call `enhance_with_cnn`. Missing or invalid trained weights raise clear errors; there is no identity or random-model fallback in this package.

## Integrated preprocessing compatibility

The refinement checkpoint was trained with the compatibility path that converts RGB to OpenCV LAB, applies CLAHE only to the L channel, and converts LAB back to RGB. Member 1's integrated `traditional.apply_clahe` uses the same operations and parameter interpretation, so the live GA-CLAHE input remains compatible. Its histogram-equalisation implementation uses LAB rather than the older compatibility fallback's YCrCb path; histogram equalisation is a separate comparison method and is not refinement-model preprocessing.

GA fitness intentionally continues to use `optimization.compat.calculate_edge_preservation` and `optimization.compat.estimate_noise`. These are the SciPy Sobel-correlation and Immerkaer estimators used to generate the refinement training cache. Member 1's separately exported edge metric uses OpenCV Sobel and a different argument convention, so it is not silently substituted into GA fitness. The deployment and cache paths both call `optimize_clahe(..., ground_truth=None)` with population 30, generations 30, crossover 0.80, mutation 0.05, seed 42, and work size 256 unless the user explicitly changes advanced settings. The UI warns when those settings no longer match checkpoint metadata.
