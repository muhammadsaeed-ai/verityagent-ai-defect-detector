"""Safe image loading, validation, resizing, and tensor preprocessing helpers."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_MAX_IMAGE_BYTES = 25 * 1024 * 1024
DEFAULT_MAX_PIXELS = 40_000_000


def validate_image_array(image: np.ndarray, *, name: str = "image") -> np.ndarray:
    """Validate a decoded image and return it unchanged."""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if image.size == 0 or image.ndim not in (2, 3):
        raise ValueError(f"{name} is empty or has an unsupported shape")
    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise ValueError(f"{name} must have 1, 3, or 4 channels")
    if image.shape[0] < 1 or image.shape[1] < 1:
        raise ValueError(f"{name} has invalid dimensions")
    if image.shape[0] * image.shape[1] > DEFAULT_MAX_PIXELS:
        raise ValueError(f"{name} exceeds the {DEFAULT_MAX_PIXELS:,}-pixel safety limit")
    if image.dtype != np.uint8:
        raise ValueError(f"{name} must use uint8 pixel values")
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    return image


def load_image(source: str | Path | np.ndarray, *, name: str = "image") -> np.ndarray:
    """Load an image from a path or validate an existing array."""
    if isinstance(source, np.ndarray):
        return validate_image_array(source, name=name)
    path = Path(source).expanduser()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{name} file does not exist: {path}")
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported image format: {path.suffix or '(no extension)'}")
    if path.stat().st_size > DEFAULT_MAX_IMAGE_BYTES:
        raise ValueError(f"{name} exceeds the {DEFAULT_MAX_IMAGE_BYTES // (1024 * 1024)} MB file limit")
    try:
        with Image.open(path) as header_image:
            if header_image.width * header_image.height > DEFAULT_MAX_PIXELS:
                raise ValueError(f"{name} exceeds the {DEFAULT_MAX_PIXELS:,}-pixel safety limit")
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
        raise ValueError(f"Could not decode or inspect {name}; the file may be corrupt or unsupported") from exc
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not decode {name}; the file may be corrupt or unsupported")
    return validate_image_array(image, name=name)


def decode_image_bytes(data: bytes | bytearray | BinaryIO, *, name: str = "uploaded image") -> np.ndarray:
    """Safely decode image bytes, including Streamlit upload objects."""
    if hasattr(data, "getvalue"):
        raw = data.getvalue()  # type: ignore[union-attr]
    elif hasattr(data, "read"):
        raw = data.read()  # type: ignore[union-attr]
    else:
        raw = bytes(data)
    if not raw:
        raise ValueError(f"{name} is empty")
    if len(raw) > DEFAULT_MAX_IMAGE_BYTES:
        raise ValueError(f"{name} exceeds the {DEFAULT_MAX_IMAGE_BYTES // (1024 * 1024)} MB file limit")
    try:
        with Image.open(BytesIO(raw)) as pil_image:
            if pil_image.width * pil_image.height > DEFAULT_MAX_PIXELS:
                raise ValueError(f"{name} exceeds the {DEFAULT_MAX_PIXELS:,}-pixel safety limit")
            pil_image.load()
            rgb = np.asarray(pil_image.convert("RGB"))
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
        raise ValueError(f"Could not decode {name}; upload a valid supported image") from exc
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert grayscale or BGR image to a three-channel BGR image."""
    image = validate_image_array(image)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 3:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)


def to_gray(image: np.ndarray) -> np.ndarray:
    """Convert grayscale or color input to an 8-bit grayscale image."""
    image = validate_image_array(image)
    if image.ndim == 2:
        return image.copy()
    if image.shape[2] == 1:
        return image[:, :, 0].copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize_to_reference(reference: np.ndarray, inspection: np.ndarray, *, aspect_tolerance: float = 0.015) -> tuple[np.ndarray, dict[str, object]]:
    """Resize inspection to reference dimensions without changing aspect ratio."""
    reference = validate_image_array(reference, name="reference image")
    inspection = validate_image_array(inspection, name="inspection image")
    ref_h, ref_w = reference.shape[:2]
    ins_h, ins_w = inspection.shape[:2]
    ref_ratio = ref_w / ref_h
    ins_ratio = ins_w / ins_h
    relative_ratio_error = abs(ref_ratio - ins_ratio) / max(ref_ratio, ins_ratio)
    if relative_ratio_error > aspect_tolerance:
        raise ValueError("Reference and inspection aspect ratios differ materially. Crop or align the images to the same field of view; automatic resizing would distort the comparison.")
    metadata: dict[str, object] = {"reference_dimensions": {"width": ref_w, "height": ref_h}, "inspection_dimensions_original": {"width": ins_w, "height": ins_h}, "inspection_resized": (ref_w, ref_h) != (ins_w, ins_h), "aspect_ratio_relative_error": float(relative_ratio_error)}
    if (ref_w, ref_h) == (ins_w, ins_h):
        return inspection.copy(), metadata
    resized = cv2.resize(inspection, (ref_w, ref_h), interpolation=cv2.INTER_AREA if ins_w > ref_w or ins_h > ref_h else cv2.INTER_CUBIC)
    return resized, metadata
