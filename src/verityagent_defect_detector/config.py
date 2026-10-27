"""Validated configuration objects shared across inspection pipelines."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ComparisonConfig:
    """Settings for reference-image comparison."""

    threshold: int = 25
    blur_kernel: int = 3
    min_area: int = 100
    align: bool = False
    mode: str = "grayscale"
    normalize_illumination: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.threshold <= 255:
            raise ValueError("threshold must be between 0 and 255")
        if self.blur_kernel < 1 or self.blur_kernel % 2 == 0:
            raise ValueError("blur_kernel must be a positive odd integer")
        if self.min_area < 1:
            raise ValueError("min_area must be at least 1 pixel")
        if self.mode not in {"grayscale", "color"}:
            raise ValueError("mode must be 'grayscale' or 'color'")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Autoencoder and preprocessing settings."""

    image_size: int = 256
    latent_channels: int = 16

    def __post_init__(self) -> None:
        if self.image_size < 32 or self.image_size % 16 != 0:
            raise ValueError("image_size must be >= 32 and divisible by 16")
        if not 4 <= self.latent_channels <= 256:
            raise ValueError("latent_channels must be between 4 and 256")
