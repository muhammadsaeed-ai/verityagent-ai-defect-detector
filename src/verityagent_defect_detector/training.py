"""Dataset loading and reproducible normal-only autoencoder training."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .anomaly_model import ConvAutoencoder, preprocess_for_model, save_checkpoint
from .config import ModelConfig
from .image_io import SUPPORTED_SUFFIXES


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    image_size: int = 256
    latent_channels: int = 16
    epochs: int = 20
    batch_size: int = 8
    learning_rate: float = 1e-3
    threshold_percentile: float = 99.5
    seed: int = 42
    device: str = "cpu"
    num_workers: int = 0

    def __post_init__(self) -> None:
        ModelConfig(self.image_size, self.latent_channels)
        if self.epochs < 1: raise ValueError("epochs must be at least 1")
        if self.batch_size < 1: raise ValueError("batch_size must be at least 1")
        if not 0.0 < self.learning_rate < 1.0: raise ValueError("learning_rate must be between 0 and 1")
        if not 90.0 <= self.threshold_percentile < 100.0: raise ValueError("threshold_percentile must be in [90, 100)")
        if self.device not in {"cpu", "cuda"}: raise ValueError("device must be 'cpu' or 'cuda'")
        if self.num_workers < 0: raise ValueError("num_workers cannot be negative")


class NormalImageDataset(Dataset[torch.Tensor]):
    """Load only supported image files from one defect-free directory."""

    def __init__(self, directory: str | Path, image_size: int) -> None:
        self.directory = Path(directory).expanduser().resolve()
        if not self.directory.exists() or not self.directory.is_dir():
            raise FileNotFoundError(f"Image dataset directory not found: {self.directory}")
        self.paths = sorted(path for path in self.directory.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES)
        if not self.paths: raise ValueError(f"No supported images found in {self.directory}")
        forbidden_labels = {"defective", "defectives", "bad", "anomalous", "anomaly"}
        for path in self.paths:
            if any(part.lower() in forbidden_labels for part in path.relative_to(self.directory).parts[:-1]):
                raise ValueError("Normal-only training/validation directories must not contain defective/anomalous subfolders")
        if self.directory.name.lower() in forbidden_labels:
            raise ValueError("Training and validation directories must contain defect-free ('good') images only")
        self.image_size = image_size

    def __len__(self) -> int: return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.paths[index]
        try:
            with Image.open(path) as image:
                image_rgb = np.asarray(image.convert("RGB"))
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(f"Could not read dataset image: {path.name}") from exc
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        tensor, _ = preprocess_for_model(image_bgr, self.image_size)
        return tensor[0]


def validate_dataset_separation(train_dir: Path, validation_dir: Path) -> None:
    train, validation = train_dir.expanduser().resolve(), validation_dir.expanduser().resolve()
    if train == validation or train in validation.parents or validation in train.parents:
        raise ValueError("Training and validation directories must be separate and must not contain each other")


def _set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def estimate_threshold(model: ConvAutoencoder, validation_loader: DataLoader[torch.Tensor], *, device: torch.device, percentile: float = 99.5, histogram_bins: int = 65536) -> float:
    """Estimate an empirical normal-pixel MSE percentile without a huge RAM buffer."""
    if not 0.0 < percentile < 100.0: raise ValueError("percentile must be strictly between 0 and 100")
    histogram = np.zeros(histogram_bins, dtype=np.int64); total_values = 0; model.eval()
    with torch.inference_mode():
        for inputs in validation_loader:
            inputs = inputs.to(device); reconstruction = model(inputs)
            errors = torch.mean(torch.square(inputs - reconstruction), dim=1).detach().cpu().numpy()
            values = np.clip(errors, 0.0, 1.0).reshape(-1)
            current, _ = np.histogram(values, bins=histogram_bins, range=(0.0, 1.0)); histogram += current; total_values += int(values.size)
    if total_values == 0: raise ValueError("Cannot estimate threshold because validation data is empty")
    target_rank = max(1, math.ceil(percentile / 100.0 * total_values)); cumulative = np.cumsum(histogram)
    bin_index = int(np.searchsorted(cumulative, target_rank, side="left"))
    return float(min(1.0, (bin_index + 1) / (histogram_bins - 1)))


def _epoch_loss(model: ConvAutoencoder, loader: DataLoader[torch.Tensor], criterion: nn.Module, device: torch.device, optimizer: torch.optim.Optimizer | None = None) -> float:
    training = optimizer is not None; model.train(training); total_loss = 0.0; sample_count = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for inputs in loader:
            inputs = inputs.to(device)
            if optimizer is not None: optimizer.zero_grad(set_to_none=True)
            reconstruction = model(inputs); loss = criterion(reconstruction, inputs)
            if optimizer is not None: loss.backward(); optimizer.step()
            batch = int(inputs.shape[0]); total_loss += float(loss.detach().cpu()) * batch; sample_count += batch
    if sample_count == 0: raise ValueError("A training or validation dataloader is empty")
    return total_loss / sample_count


def train_autoencoder(train_dir: str | Path, validation_dir: str | Path, output_path: str | Path, *, config: TrainingConfig | None = None, overwrite: bool = False) -> dict[str, Any]:
    """Train on good-only data and save checkpoint plus history."""
    settings = config or TrainingConfig(); validate_dataset_separation(Path(train_dir), Path(validation_dir))
    if settings.device == "cuda" and not torch.cuda.is_available(): raise ValueError("CUDA was selected but is unavailable; use --device cpu")
    destination = Path(output_path).expanduser(); history_path = destination.with_name(f"{destination.stem}_training_history.json")
    if not overwrite and (destination.exists() or history_path.exists()): raise FileExistsError("Checkpoint or training history already exists; pass overwrite=True to replace it")
    _set_seed(settings.seed)
    train_dataset = NormalImageDataset(train_dir, settings.image_size); validation_dataset = NormalImageDataset(validation_dir, settings.image_size)
    train_loader: DataLoader[torch.Tensor] = DataLoader(train_dataset, batch_size=settings.batch_size, shuffle=True, num_workers=settings.num_workers, drop_last=False)
    validation_loader: DataLoader[torch.Tensor] = DataLoader(validation_dataset, batch_size=settings.batch_size, shuffle=False, num_workers=settings.num_workers, drop_last=False)
    device = torch.device(settings.device); model = ConvAutoencoder(settings.image_size, settings.latent_channels).to(device)
    criterion = nn.MSELoss(); optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate); history: list[dict[str, float | int]] = []
    for epoch in range(1, settings.epochs + 1):
        history.append({"epoch": epoch, "train_mse": _epoch_loss(model, train_loader, criterion, device, optimizer), "validation_mse": _epoch_loss(model, validation_loader, criterion, device)})
    threshold = estimate_threshold(model, validation_loader, device=device, percentile=settings.threshold_percentile)
    checkpoint = save_checkpoint(destination, model, threshold=threshold, config=ModelConfig(settings.image_size, settings.latent_channels), metadata={"training_config": asdict(settings), "train_image_count": len(train_dataset), "validation_image_count": len(validation_dataset), "final_train_mse": history[-1]["train_mse"], "final_validation_mse": history[-1]["validation_mse"]}, overwrite=overwrite)
    payload = {"project": "VerityAgent AI Defect Detector", "training_config": asdict(settings), "train_image_count": len(train_dataset), "validation_image_count": len(validation_dataset), "threshold": threshold, "threshold_percentile": settings.threshold_percentile, "history": history, "note": "Threshold is calibrated on defect-free validation pixels only; use labeled defect tests for accuracy evaluation."}
    history_path.parent.mkdir(parents=True, exist_ok=True); history_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"checkpoint": str(checkpoint), "history_path": str(history_path), **payload}
