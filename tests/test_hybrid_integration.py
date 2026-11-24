from __future__ import annotations

import numpy as np

from cnn.model import build_model
from optimization.hybrid import enhance_hybrid


def test_hybrid_uses_cnn_model_and_preserves_shape():
    # Random weights are intentionally test-only; this checks wiring, not enhancement quality.
    model = build_model(base_channels=4)
    image = np.random.default_rng(3).integers(10, 80, (23, 25, 3), dtype=np.uint8)
    result = enhance_hybrid(
        image,
        model,
        ground_truth=None,
        pop_size=4,
        generations=1,
        tournament_size=2,
        elitism=1,
        work_size=32,
        seed=4,
    )
    assert result["output"].shape == image.shape
    assert result["output"].dtype == np.uint8
    assert result["clahe_output"].shape == image.shape
