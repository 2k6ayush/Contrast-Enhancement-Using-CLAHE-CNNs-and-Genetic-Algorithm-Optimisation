# Traditional Image Processing and Evaluation (Member 1)

## Enhancement methods (`traditional.py`)

```python
from traditional import apply_histogram_equalization, apply_clahe

out = apply_histogram_equalization(image)
out = apply_clahe(image, clip_limit=2.0, grid_size=8)
```

Both work on the lightness channel (LAB) so colours are not distorted.

## Metrics (`metrics.py`)

| Function | Needs ground truth | Better |
|---|---|---|
| `calculate_psnr(enhanced, ground_truth)` | yes | higher |
| `calculate_ssim(enhanced, ground_truth)` | yes | higher |
| `calculate_entropy(image)` | no | context-dependent |
| `calculate_niqe(image)` (needs `pip install pyiqa`) | no | lower |
| `calculate_contrast_improvement(original, enhanced)` | no | context-dependent |
| `calculate_edge_preservation(original, enhanced)` | no | higher |
| `evaluate_all(original, enhanced, ground_truth=None, include_niqe=False)` | optional | - |

Entropy and contrast improvement should not be maximised blindly, because noise
also increases them.

## Run Experiment 1

```powershell
python experiment1_traditional.py --niqe                 # test set (15 images)
python experiment1_traditional.py --split our485         # 485 images, no NIQE
```

Results are saved in `results/` (CSV tables, figures, enhanced images).

## Tests

```powershell
python -m pytest
```