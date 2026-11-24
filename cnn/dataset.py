"""Paired LOL dataset validation, leakage-safe splits, and paired transforms."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MANIFEST_VERSION = 1


@dataclass(frozen=True)
class PairRecord:
    name: str
    low: Path
    high: Path
    low_sha256: str = ""
    high_sha256: str = ""


def _image_names(folder: Path) -> set[str]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {folder}")
    return {path.name for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_pairs(dataset_root: str | Path, subset: str = "our485", verify: bool = True) -> list[PairRecord]:
    """Match low/high images by filename and optionally decode/validate every pair."""
    subset_root = Path(dataset_root).expanduser().resolve() / subset
    low_root, high_root = subset_root / "low", subset_root / "high"
    low_names, high_names = _image_names(low_root), _image_names(high_root)
    missing_high = sorted(low_names - high_names)
    missing_low = sorted(high_names - low_names)
    if missing_high or missing_low:
        details = []
        if missing_high:
            details.append(f"missing high partners: {missing_high[:10]}")
        if missing_low:
            details.append(f"missing low partners: {missing_low[:10]}")
        raise ValueError("Pairing failed; " + "; ".join(details))

    records: list[PairRecord] = []
    errors: list[str] = []
    for name in sorted(low_names):
        low_path, high_path = low_root / name, high_root / name
        if verify:
            try:
                with Image.open(low_path) as low_image, Image.open(high_path) as high_image:
                    low_image.verify()
                    high_image.verify()
                with Image.open(low_path) as low_image, Image.open(high_path) as high_image:
                    if low_image.size != high_image.size:
                        errors.append(f"{name}: low {low_image.size} != high {high_image.size}")
                        continue
                    if low_image.width < 1 or low_image.height < 1:
                        errors.append(f"{name}: empty dimensions")
                        continue
            except (OSError, UnidentifiedImageError) as exc:
                errors.append(f"{name}: unreadable ({exc})")
                continue
        records.append(PairRecord(name, low_path, high_path))
    if errors:
        raise ValueError("Invalid image pairs:\n" + "\n".join(errors[:20]))
    if not records:
        raise ValueError(f"No paired images found below {subset_root}")
    return records


def _duplicate_groups(records: list[PairRecord]) -> tuple[list[list[int]], list[PairRecord]]:
    """Union pairs sharing an exact low or high file hash."""
    hashed = [PairRecord(r.name, r.low, r.high, _sha256(r.low), _sha256(r.high)) for r in records]
    parent = list(range(len(hashed)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        root_a, root_b = find(first), find(second)
        if root_a != root_b:
            parent[root_b] = root_a

    seen_low: dict[str, int] = {}
    seen_high: dict[str, int] = {}
    for index, record in enumerate(hashed):
        for digest, seen in ((record.low_sha256, seen_low), (record.high_sha256, seen_high)):
            if digest in seen:
                union(index, seen[digest])
            else:
                seen[digest] = index
    groups: dict[int, list[int]] = {}
    for index in range(len(hashed)):
        groups.setdefault(find(index), []).append(index)
    return list(groups.values()), hashed


def create_or_load_split_manifest(
    dataset_root: str | Path,
    manifest_path: str | Path,
    validation_fraction: float = 0.15,
    seed: int = 42,
) -> dict:
    """Create one reproducible our485 split while keeping exact duplicates together."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    root = Path(dataset_root).expanduser().resolve()
    destination = Path(manifest_path).expanduser().resolve()
    records = discover_pairs(root, "our485", verify=True)
    names = {record.name for record in records}
    groups, hashed = _duplicate_groups(records)
    current_by_name = {record.name: record for record in hashed}
    test_overlap_check = {"subset": "eval15", "exact_low_overlaps": [], "exact_high_overlaps": []}
    if (root / "eval15" / "low").is_dir() and (root / "eval15" / "high").is_dir():
        _, hashed_test = _duplicate_groups(discover_pairs(root, "eval15", verify=True))
        training_low = {record.low_sha256: record.name for record in hashed}
        training_high = {record.high_sha256: record.name for record in hashed}
        test_overlap_check["exact_low_overlaps"] = [
            {"test": record.name, "our485": training_low[record.low_sha256]}
            for record in hashed_test
            if record.low_sha256 in training_low
        ]
        test_overlap_check["exact_high_overlaps"] = [
            {"test": record.name, "our485": training_high[record.high_sha256]}
            for record in hashed_test
            if record.high_sha256 in training_high
        ]
        if test_overlap_check["exact_low_overlaps"] or test_overlap_check["exact_high_overlaps"]:
            raise ValueError(f"Exact duplicate leakage exists between our485 and eval15: {test_overlap_check}")
    if destination.exists():
        manifest = json.loads(destination.read_text(encoding="utf-8"))
        if manifest.get("version") != MANIFEST_VERSION:
            raise ValueError(f"Unsupported split manifest version in {destination}")
        manifest_names = {item["name"] for item in manifest.get("pairs", [])}
        if manifest_names != names:
            raise ValueError("Existing split manifest does not match the current our485 filenames")
        for item in manifest["pairs"]:
            current = current_by_name[item["name"]]
            if item.get("low_sha256") != current.low_sha256 or item.get("high_sha256") != current.high_sha256:
                raise ValueError(f"Dataset content changed after the manifest was created: {item['name']}")
        if manifest.get("test_overlap_check") != test_overlap_check:
            manifest["test_overlap_check"] = test_overlap_check
            destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return manifest

    random.Random(seed).shuffle(groups)
    target_validation = round(len(records) * validation_fraction)
    validation_indices: set[int] = set()
    for group in groups:
        if len(validation_indices) >= target_validation:
            break
        validation_indices.update(group)
    pairs = []
    for index, record in enumerate(hashed):
        pairs.append(
            {
                "name": record.name,
                "split": "validation" if index in validation_indices else "train",
                "low_sha256": record.low_sha256,
                "high_sha256": record.high_sha256,
            }
        )
    manifest = {
        "version": MANIFEST_VERSION,
        "dataset_subset": "our485",
        "seed": seed,
        "validation_fraction": validation_fraction,
        "duplicate_policy": "pairs sharing an exact low or high SHA-256 stay together",
        "test_overlap_check": test_overlap_check,
        "pairs": pairs,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def records_from_manifest(dataset_root: str | Path, manifest: dict, split: str) -> list[PairRecord]:
    if split not in {"train", "validation"}:
        raise ValueError("split must be train or validation")
    root = Path(dataset_root).expanduser().resolve() / "our485"
    chosen = [item for item in manifest["pairs"] if item["split"] == split]
    return [PairRecord(item["name"], root / "low" / item["name"], root / "high" / item["name"]) for item in chosen]


def _load_rgb(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as image:
            return np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"Could not decode RGB image {path}: {exc}") from exc


class PairedImageDataset(Dataset):
    """Loads aligned pairs with shared crop/flip randomness."""

    def __init__(
        self,
        records: Iterable[PairRecord],
        crop_size: int | None = 256,
        augment: bool = False,
    ) -> None:
        self.records = list(records)
        self.crop_size = crop_size
        self.augment = augment
        if not self.records:
            raise ValueError("PairedImageDataset requires at least one pair")
        if crop_size is not None and crop_size < 1:
            raise ValueError("crop_size must be positive or None")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        low, high = _load_rgb(record.low), _load_rgb(record.high)
        if low.shape != high.shape:
            raise ValueError(f"Pair {record.name} has incompatible decoded shapes {low.shape} and {high.shape}")
        height, width = low.shape[:2]
        if self.crop_size is not None:
            crop = min(self.crop_size, height, width)
            if self.augment:
                top = random.randint(0, height - crop)
                left = random.randint(0, width - crop)
            else:
                top = (height - crop) // 2
                left = (width - crop) // 2
            low = low[top : top + crop, left : left + crop]
            high = high[top : top + crop, left : left + crop]
        if self.augment and random.random() < 0.5:
            low, high = np.flip(low, axis=1), np.flip(high, axis=1)
        if self.augment and random.random() < 0.5:
            low, high = np.flip(low, axis=0), np.flip(high, axis=0)
        low_tensor = torch.from_numpy(np.ascontiguousarray(low)).permute(2, 0, 1).float().div(255)
        high_tensor = torch.from_numpy(np.ascontiguousarray(high)).permute(2, 0, 1).float().div(255)
        return {"input": low_tensor, "target": high_tensor, "name": record.name}


def cache_ga_clahe_inputs(
    records: Iterable[PairRecord],
    cache_root: str | Path,
    ga_settings: dict,
    force: bool = False,
) -> tuple[list[PairRecord], dict]:
    """Generate reproducible reference-free GA-CLAHE inputs for refinement training."""
    import cv2
    from optimization.genetic_algorithm import optimize_clahe

    source_records = list(records)
    destination = Path(cache_root).expanduser().resolve()
    low_destination = destination / "low"
    low_destination.mkdir(parents=True, exist_ok=True)
    metadata_path = destination / "metadata.json"
    expected_metadata = {
        "format": "ga-clahe-cache-v1",
        "ground_truth_used": False,
        "ga_settings": ga_settings,
        "source": [{"name": r.name, "low_sha256": _sha256(r.low)} for r in source_records],
    }
    if metadata_path.exists() and not force:
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing != expected_metadata:
            raise ValueError(
                f"GA cache metadata mismatch at {metadata_path}; use --rebuild-hybrid-cache explicitly"
            )
    else:
        for record in source_records:
            rgb = _load_rgb(record.low)
            result = optimize_clahe(rgb, ground_truth=None, **ga_settings)
            output = result["enhanced"]
            if not cv2.imwrite(str(low_destination / record.name), cv2.cvtColor(output, cv2.COLOR_RGB2BGR)):
                raise OSError(f"Could not write cached GA image {low_destination / record.name}")
        metadata_path.write_text(json.dumps(expected_metadata, indent=2) + "\n", encoding="utf-8")
    missing = [record.name for record in source_records if not (low_destination / record.name).is_file()]
    if missing:
        raise FileNotFoundError(f"GA cache is incomplete; missing {missing[:10]}")
    cached = [PairRecord(r.name, low_destination / r.name, r.high) for r in source_records]
    return cached, expected_metadata
