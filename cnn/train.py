"""Training entry point for baseline and GA-CLAHE refinement CNNs."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cnn.dataset import (
    PairedImageDataset,
    cache_ga_clahe_inputs,
    create_or_load_split_manifest,
    records_from_manifest,
)
from cnn.inference import CHECKPOINT_FORMAT, resolve_device, validate_checkpoint
from cnn.losses import L1SSIMLoss, ssim
from cnn.model import build_model, count_parameters


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_random_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_random_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def checkpoint_payload(
    model: torch.nn.Module,
    metadata: dict[str, Any],
    epoch: int,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    best_validation_ssim: float | None = None,
) -> dict[str, Any]:
    architecture = getattr(model, "architecture_config", None)
    if not isinstance(architecture, dict):
        raise ValueError("Model must expose architecture_config for reproducible checkpoints")
    payload: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "model_state": model.state_dict(),
        "architecture": architecture,
        "metadata": metadata,
        "epoch": int(epoch),
        "best_validation_ssim": best_validation_ssim,
        "random_state": capture_random_state(),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    return payload


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    metadata: dict[str, Any],
    epoch: int,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    best_validation_ssim: float | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        checkpoint_payload(model, metadata, epoch, optimizer, scheduler, best_validation_ssim),
        destination,
    )


def load_training_checkpoint(
    path: str | Path,
    device: torch.device,
    expected_input_mode: str,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = validate_checkpoint(
        torch.load(Path(path), map_location=device, weights_only=False), expected_input_mode
    )
    model = build_model(**checkpoint["architecture"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint


def _tensor_metrics(prediction: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    mse = torch.mean((prediction.float() - target.float()).square()).clamp_min(1e-12)
    psnr = float((10 * torch.log10(1.0 / mse)).item())
    return psnr, float(ssim(prediction, target).item())


def validate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    psnr_values, ssim_values = [], []
    with torch.inference_mode():
        for batch in loader:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            predictions = model(inputs).clamp(0, 1)
            psnr_value, ssim_value = _tensor_metrics(predictions, targets)
            psnr_values.append(psnr_value)
            ssim_values.append(ssim_value)
    return {"psnr": float(np.mean(psnr_values)), "ssim": float(np.mean(ssim_values))}


def _write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_history(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=130)
    axes[0].plot(epochs, [row["train_loss"] for row in rows], label="train L1+SSIM")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Training loss")
    axes[1].plot(epochs, [row["validation_psnr"] for row in rows], label="PSNR")
    second = axes[1].twinx()
    second.plot(epochs, [row["validation_ssim"] for row in rows], color="tab:orange", label="SSIM")
    axes[1].set(xlabel="Epoch", ylabel="PSNR (dB)", title="Original-resolution validation")
    second.set_ylabel("SSIM")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def train(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    device = resolve_device(args.device)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = create_or_load_split_manifest(
        args.dataset_root, args.manifest, args.validation_fraction, args.seed
    )
    training_records = records_from_manifest(args.dataset_root, manifest, "train")
    validation_records = records_from_manifest(args.dataset_root, manifest, "validation")

    ga_cache_metadata = None
    if args.input_mode == "ga_clahe_refinement":
        ga_settings = {
            "pop_size": args.ga_pop_size,
            "generations": args.ga_generations,
            "crossover_rate": args.ga_crossover,
            "mutation_rate": args.ga_mutation,
            "seed": args.ga_seed,
            "work_size": args.ga_work_size,
        }
        combined, ga_cache_metadata = cache_ga_clahe_inputs(
            training_records + validation_records,
            args.hybrid_cache,
            ga_settings,
            force=args.rebuild_hybrid_cache,
        )
        by_name = {record.name: record for record in combined}
        training_records = [by_name[record.name] for record in training_records]
        validation_records = [by_name[record.name] for record in validation_records]

    training_dataset = PairedImageDataset(training_records, crop_size=args.crop_size, augment=True)
    validation_dataset = PairedImageDataset(validation_records, crop_size=None, augment=False)
    generator = torch.Generator().manual_seed(args.seed)
    training_loader = DataLoader(
        training_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=args.workers)

    start_epoch, best_ssim, history = 1, float("-inf"), []
    if args.resume:
        model, resume = load_training_checkpoint(args.resume, device, args.input_mode)
        start_epoch = int(resume.get("epoch", 0)) + 1
        best_ssim = float(resume.get("best_validation_ssim") or float("-inf"))
    else:
        model = build_model(base_channels=args.base_channels).to(device)
        resume = None

    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=args.lr_patience)
    if resume:
        if "optimizer_state" not in resume or "scheduler_state" not in resume:
            raise ValueError("Resume checkpoint lacks optimizer or scheduler state")
        optimizer.load_state_dict(resume["optimizer_state"])
        scheduler.load_state_dict(resume["scheduler_state"])
        if "random_state" in resume:
            restore_random_state(resume["random_state"])
        history_path = output_dir / "history.csv"
        if history_path.exists():
            with history_path.open(newline="", encoding="utf-8") as handle:
                history = [
                    {
                        "epoch": int(row["epoch"]),
                        "train_loss": float(row["train_loss"]),
                        "validation_psnr": float(row["validation_psnr"]),
                        "validation_ssim": float(row["validation_ssim"]),
                        "learning_rate": float(row["learning_rate"]),
                    }
                    for row in csv.DictReader(handle)
                ]

    criterion = L1SSIMLoss(args.l1_weight, args.ssim_weight)
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    metadata = {
        "input_mode": args.input_mode,
        "dataset_subset": "our485",
        "split_manifest": str(args.manifest.expanduser().resolve()),
        "seed": args.seed,
        "crop_size": args.crop_size,
        "training_config": {
            "optimizer": "Adam",
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "requested_epochs": args.epochs,
            "loss": "L1SSIMLoss",
            "l1_weight": args.l1_weight,
            "ssim_weight": args.ssim_weight,
            "scheduler": "ReduceLROnPlateau",
            "mixed_precision": amp_enabled,
        },
        "ga_preprocessing": ga_cache_metadata,
        "test_only": False,
    }
    print(f"Device: {device}; train={len(training_dataset)}, validation={len(validation_dataset)}")
    print(f"Trainable parameters: {count_parameters(model):,}")
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        losses = []
        for batch in training_loader:
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                predictions = model(inputs)
                # The loss promotes itself to float32 for stable SSIM statistics.
                loss = criterion(predictions, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().item()))
        metrics = validate(model, validation_loader, device)
        scheduler.step(metrics["ssim"])
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_psnr": metrics["psnr"],
            "validation_ssim": metrics["ssim"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        _write_history(output_dir / "history.csv", history)
        _plot_history(output_dir / "training_history.png", history)
        improved = metrics["ssim"] > best_ssim
        best_ssim = max(best_ssim, metrics["ssim"])
        save_checkpoint(output_dir / "latest.pt", model, metadata, epoch, optimizer, scheduler, best_ssim)
        if improved:
            save_checkpoint(output_dir / "best.pt", model, metadata, epoch, optimizer, scheduler, best_ssim)
        print(
            f"epoch={epoch}/{args.epochs} loss={row['train_loss']:.5f} "
            f"val_psnr={metrics['psnr']:.3f} val_ssim={metrics['ssim']:.5f}"
        )
    (output_dir / "run_config.json").write_text(
        json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return {"best_validation_ssim": best_ssim, "output_dir": str(output_dir)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train compact U-Net image enhancement models")
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "Dataset")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "splits" / "our485_split.json")
    parser.add_argument("--input-mode", choices=("baseline", "ga_clahe_refinement"), default="baseline")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "models" / "baseline")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--l1-weight", type=float, default=0.8)
    parser.add_argument("--ssim-weight", type=float, default=0.2)
    parser.add_argument("--lr-patience", type=int, default=3)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hybrid-cache", type=Path, default=PROJECT_ROOT / "cache" / "ga_clahe_our485")
    parser.add_argument("--rebuild-hybrid-cache", action="store_true")
    parser.add_argument("--ga-pop-size", type=int, default=30)
    parser.add_argument("--ga-generations", type=int, default=30)
    parser.add_argument("--ga-crossover", type=float, default=0.8)
    parser.add_argument("--ga-mutation", type=float, default=0.05)
    parser.add_argument("--ga-seed", type=int, default=42)
    parser.add_argument("--ga-work-size", type=int, default=256)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("epochs, batch-size, and learning-rate must be positive")
    if args.input_mode == "ga_clahe_refinement" and args.output_dir == PROJECT_ROOT / "models" / "baseline":
        args.output_dir = PROJECT_ROOT / "models" / "refinement"
    return args


if __name__ == "__main__":
    train(parse_args())
