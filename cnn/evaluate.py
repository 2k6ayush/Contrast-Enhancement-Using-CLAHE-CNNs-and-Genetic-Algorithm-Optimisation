"""Evaluate traditional, CNN, and hybrid methods without touching legacy results."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cnn.dataset import discover_pairs
from cnn.inference import enhance_with_cnn, load_checkpoint, resolve_device
from optimization.compat import apply_clahe, apply_histogram_equalization, calculate_edge_preservation
from optimization.genetic_algorithm import optimize_clahe


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _metrics(output: np.ndarray, target: np.ndarray, source: np.ndarray) -> dict[str, float | None]:
    return {
        "psnr": float(peak_signal_noise_ratio(target, output, data_range=255)),
        "ssim": float(structural_similarity(target, output, channel_axis=2, data_range=255)),
        "entropy": float(_entropy(output)),
        "contrast_improvement_index": float(output.std() / (source.std() + 1e-8)),
        "edge_preservation": float(calculate_edge_preservation(output, target)),
        "niqe": None,  # No validated NIQE implementation is bundled.
    }


def _entropy(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    histogram = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    probabilities = histogram[histogram > 0] / histogram.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    device = resolve_device(args.device)
    pairs = discover_pairs(args.dataset_root, "eval15", verify=True)
    models, load_times = {}, {}
    for name, path, mode in (
        ("baseline", args.baseline_checkpoint, "baseline"),
        ("refinement", args.refinement_checkpoint, "ga_clahe_refinement"),
    ):
        if path is None:
            continue
        started = time.perf_counter()
        model, metadata, _ = load_checkpoint(path, device, mode)
        _synchronize(device)
        load_times[name] = (time.perf_counter() - started) * 1000
        models[name] = (model, metadata, path.expanduser().resolve())

    ga_settings = {
        "pop_size": args.ga_pop_size,
        "generations": args.ga_generations,
        "crossover_rate": args.ga_crossover,
        "mutation_rate": args.ga_mutation,
        "seed": args.ga_seed,
        "work_size": args.ga_work_size,
    }
    rows: list[dict] = []
    for pair in pairs:
        source, target = _load_rgb(pair.low), _load_rgb(pair.high)
        methods: list[tuple[str, Callable[[], np.ndarray], str, str | None, float | None]] = [
            ("Histogram Equalization", lambda: apply_histogram_equalization(source), "HE", None, None),
            ("Standard CLAHE", lambda: apply_clahe(source, 2.0, 8), "clip=2.0,grid=8", None, None),
        ]
        if "baseline" in models:
            model, _, path = models["baseline"]
            methods.append(("CNN alone", lambda model=model: enhance_with_cnn(source, model), "raw RGB", str(path), load_times["baseline"]))

        # Compute GA once per image and reuse it for both GA methods.
        ga_started = time.perf_counter()
        ga_result = optimize_clahe(source, ground_truth=None, **ga_settings)
        ga_output = ga_result["enhanced"]
        ga_ms = (time.perf_counter() - ga_started) * 1000
        methods.append(("GA-CLAHE alone", lambda: ga_output, json.dumps(ga_settings, sort_keys=True), None, None))
        if "refinement" in models:
            model, _, path = models["refinement"]
            methods.extend(
                [
                    (
                        "Fixed CLAHE + CNN",
                        lambda model=model: enhance_with_cnn(apply_clahe(source, 2.0, 8), model),
                        "clip=2.0,grid=8; refinement weights held fixed (input mismatch ablation)",
                        str(path),
                        load_times["refinement"],
                    ),
                    (
                        "GA-CLAHE + CNN",
                        lambda model=model: enhance_with_cnn(ga_output, model),
                        json.dumps(ga_settings, sort_keys=True) + "; matched refinement checkpoint",
                        str(path),
                        load_times["refinement"],
                    ),
                ]
            )
        for method, operation, preprocessing, checkpoint_path, loading_ms in methods:
            _synchronize(device)
            started = time.perf_counter()
            output = operation()
            _synchronize(device)
            inference_ms = (time.perf_counter() - started) * 1000
            if method.startswith("GA-CLAHE"):
                inference_ms += ga_ms
            metric_values = _metrics(output, target, source)
            rows.append(
                {
                    "image": pair.name,
                    "method": method,
                    **metric_values,
                    "inference_ms": inference_ms,
                    "model_loading_ms": loading_ms,
                    "checkpoint": checkpoint_path,
                    "checkpoint_sha256": _sha256(Path(checkpoint_path)) if checkpoint_path else None,
                    "preprocessing": preprocessing,
                    "result_status": "measured",
                }
            )
    per_image = pd.DataFrame(rows)
    numeric = ["psnr", "ssim", "entropy", "contrast_improvement_index", "edge_preservation", "inference_ms"]
    aggregate = per_image.groupby("method", sort=False)[numeric].agg(["mean", "std"]).reset_index()
    aggregate.columns = ["_".join(filter(None, column)).rstrip("_") for column in aggregate.columns.to_flat_index()]
    destination = args.output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    per_image.to_csv(destination / "per_image_metrics.csv", index=False)
    aggregate.to_csv(destination / "aggregate_metrics.csv", index=False)
    run_metadata = {
        "split": "eval15 final test",
        "model_selection_or_tuning_performed": False,
        "device": str(device),
        "ga_ground_truth_used": False,
        "ga_settings": ga_settings,
        "baseline_checkpoint": str(args.baseline_checkpoint) if args.baseline_checkpoint else None,
        "refinement_checkpoint": str(args.refinement_checkpoint) if args.refinement_checkpoint else None,
        "niqe": "unavailable: no validated implementation bundled",
        "legacy_results_modified": False,
    }
    (destination / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8")
    return per_image, aggregate


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final eval15 comparison for trained CNN checkpoints")
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "Dataset")
    parser.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--refinement-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "cnn_evaluation")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ga-pop-size", type=int, default=30)
    parser.add_argument("--ga-generations", type=int, default=30)
    parser.add_argument("--ga-crossover", type=float, default=0.8)
    parser.add_argument("--ga-mutation", type=float, default=0.05)
    parser.add_argument("--ga-seed", type=int, default=42)
    parser.add_argument("--ga-work-size", type=int, default=256)
    return parser.parse_args(argv)


if __name__ == "__main__":
    _, summary = evaluate(parse_args())
    print(summary.to_string(index=False))
