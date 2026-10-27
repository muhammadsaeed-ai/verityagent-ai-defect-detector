"""Common visualization helpers for inspection results."""

from __future__ import annotations

import cv2
import numpy as np

from .image_io import to_bgr


def colorize_difference(difference_map: np.ndarray) -> np.ndarray:
    """Create a JET heatmap from an 8-bit grayscale difference map."""
    if difference_map.ndim != 2 or difference_map.dtype != np.uint8:
        raise ValueError("difference_map must be an 8-bit grayscale array")
    return cv2.applyColorMap(difference_map, cv2.COLORMAP_JET)


def make_overlay(image: np.ndarray, heatmap: np.ndarray, *, alpha: float = 0.4) -> np.ndarray:
    """Blend a color heatmap over an image."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    image_bgr = to_bgr(image)
    if heatmap.shape != image_bgr.shape:
        heatmap = cv2.resize(heatmap, (image_bgr.shape[1], image_bgr.shape[0]))
    return cv2.addWeighted(image_bgr, 1.0 - alpha, heatmap, alpha, 0)


def annotate_regions(image: np.ndarray, regions: list[object], *, color: tuple[int, int, int] = (40, 40, 245)) -> np.ndarray:
    """Draw bounding boxes for region objects exposing x/y/width/height."""
    annotated = to_bgr(image)
    for index, region in enumerate(regions, start=1):
        x = int(getattr(region, "x")); y = int(getattr(region, "y"))
        width = int(getattr(region, "width")); height = int(getattr(region, "height"))
        cv2.rectangle(annotated, (x, y), (x + width, y + height), color, 2)
        cv2.putText(annotated, f"Region {index}", (x, max(18, y - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return annotated
