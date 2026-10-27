"""PyTorch convolutional autoencoder for normal-only anomaly modelling."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn

from .config import ModelConfig
from .image_io import to_bgr


class ConvAutoencoder(nn.Module):
    """Compact four-stage convolutional autoencoder."""

    def __init__(self, image_size: int = 256, latent_channels: int = 16) -> None:
        super().__init__()
        ModelConfig(image_size=image_size, latent_channels=latent_channels)
        self.image_size = image_size
        self.latent_channels = latent_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, latent_channels, 4, 2, 1), nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(inputs))


def preprocess_for_model(image_bgr: np.ndarray, image_size: int) -> tuple[torch.Tensor, dict[str, int]]:
    """Convert an image to a normalized, aspect-preserving letterboxed tensor."""
    image_bgr = to_bgr(image_bgr)
    source_h, source_w = image_bgr.shape[:2]
    scale = min(image_size / source_w, image_size / source_h)
    resized_w = max(1, int(round(source_w * scale)))
    resized_h = max(1, int(round(source_h * scale)))
    resized = cv2.resize(image_bgr, (resized_w, resized_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    canvas = np.full((image_size, image_size, 3), 127, dtype=np.uint8)
    left = (image_size - resized_w) // 2
    top = (image_size - resized_h) // 2
    canvas[top : top + resized_h, left : left + resized_w] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float().div_(255.0)
    return tensor.unsqueeze(0), {"source_width": source_w, "source_height": source_h, "resized_width": resized_w, "resized_height": resized_h, "pad_left": left, "pad_top": top, "image_size": image_size}


def map_error_to_source(error_map: np.ndarray, transform: dict[str, int]) -> np.ndarray:
    """Remove letterbox padding and resize an error map to source dimensions."""
    top, left = transform["pad_top"], transform["pad_left"]
    cropped = error_map[top : top + transform["resized_height"], left : left + transform["resized_width"]]
    return cv2.resize(cropped, (transform["source_width"], transform["source_height"]), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def reconstruction_error(inputs: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
    """Return single-channel per-pixel mean squared reconstruction error."""
    if inputs.shape != reconstruction.shape or inputs.ndim != 4 or inputs.shape[1] != 3:
        raise ValueError("inputs and reconstruction must share a [batch, 3, height, width] shape")
    return torch.mean(torch.square(inputs - reconstruction), dim=1)


def save_checkpoint(path: str | Path, model: ConvAutoencoder, *, threshold: float, config: ModelConfig, metadata: dict[str, Any] | None = None, overwrite: bool = False) -> Path:
    """Save a state-dict checkpoint plus primitive preprocessing metadata."""
    destination = Path(path).expanduser()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Checkpoint already exists: {destination}; pass overwrite=True to replace it")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    destination.parent.mkdir(parents=True, exist_ok=True)
    checkpoint: dict[str, Any] = {"format_version": 1, "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "model_config": {"image_size": config.image_size, "latent_channels": config.latent_channels}, "preprocessing": {"color_space": "RGB", "range": "0..1", "resize": "aspect-preserving letterbox", "pad_value": 127}, "threshold": float(threshold), "threshold_method": "validation-pixel-error percentile", "metadata": metadata or {}}
    torch.save(checkpoint, destination)
    return destination


def load_model(path: str | Path, *, device: str | torch.device = "cpu") -> tuple[ConvAutoencoder, dict[str, Any]]:
    """Safely load and strictly validate a local checkpoint."""
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.exists() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as exc:
        raise ValueError("Model checkpoint could not be safely decoded") from exc
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    model_config, state_dict = checkpoint.get("model_config"), checkpoint.get("model_state_dict")
    if not isinstance(model_config, dict) or not isinstance(state_dict, dict):
        raise ValueError("Checkpoint is missing its model configuration or weights")
    if not isinstance(checkpoint.get("threshold"), (int, float)):
        raise ValueError("Checkpoint is missing a numeric anomaly threshold")
    threshold = float(checkpoint["threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Checkpoint threshold is outside the valid 0..1 range")
    try:
        config = ModelConfig(image_size=int(model_config["image_size"]), latent_channels=int(model_config["latent_channels"]))
        model = ConvAutoencoder(config.image_size, config.latent_channels)
        model.load_state_dict(state_dict, strict=True)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("Checkpoint model configuration or weight tensors are invalid") from exc
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    model.to(target_device); model.eval()
    return model, {"checkpoint_path": str(checkpoint_path), "threshold": threshold, "threshold_method": checkpoint.get("threshold_method", "unspecified"), "model_config": {"image_size": config.image_size, "latent_channels": config.latent_channels}, "preprocessing": checkpoint.get("preprocessing", {}), "metadata": checkpoint.get("metadata", {})}
