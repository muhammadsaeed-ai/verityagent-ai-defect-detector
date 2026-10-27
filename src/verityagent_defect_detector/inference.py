"""Inference utilities for a trained reconstruction-based anomaly model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
import torch

from .anomaly_model import ConvAutoencoder, load_model, map_error_to_source, preprocess_for_model
from .comparison import DefectRegion
from .image_io import load_image, to_bgr
from .visualization import annotate_regions, colorize_difference, make_overlay


@dataclass(slots=True)
class AnomalyResult:
    """Outputs from neural reconstruction-error inference."""

    inspection_image: np.ndarray
    anomaly_map: np.ndarray
    binary_mask: np.ndarray
    heatmap: np.ndarray
    overlay: np.ndarray
    annotated_image: np.ndarray
    regions: list[DefectRegion]
    anomaly_score: float
    affected_pixel_ratio: float
    threshold: float
    processing_time_ms: float
    metadata: dict[str, Any]

    def to_report(self) -> dict[str, Any]:
        return {"project": "VerityAgent AI Defect Detector", "method": "pytorch_reconstruction_anomaly_detection", "status": "candidate_regions_detected" if self.regions else "no_candidate_regions_detected", "anomaly_score": round(self.anomaly_score, 8), "anomaly_score_definition": "mean per-pixel RGB reconstruction MSE; not a probability", "pixel_threshold": round(self.threshold, 8), "affected_pixel_ratio": round(self.affected_pixel_ratio, 8), "affected_pixel_percentage": round(self.affected_pixel_ratio * 100.0, 4), "detected_region_count": len(self.regions), "regions": [region.to_dict() for region in self.regions], "processing_time_ms": round(self.processing_time_ms, 3), "image_dimensions": {"width": int(self.inspection_image.shape[1]), "height": int(self.inspection_image.shape[0])}, "metadata": self.metadata, "limitations": ["Reconstruction error is an anomaly signal, not a calibrated probability.", "The threshold is estimated from defect-free validation images only.", "Evaluate with representative labeled defect images before operational use."]}


def predict_anomalies(model: ConvAutoencoder, inspection: str | Path | np.ndarray, *, threshold: float, min_area: int = 100, device: str | torch.device = "cpu", metadata: dict[str, Any] | None = None) -> AnomalyResult:
    """Run the model and map reconstruction anomalies back to source pixels."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if min_area < 1:
        raise ValueError("min_area must be at least 1 pixel")
    started = perf_counter()
    image = load_image(inspection, name="inspection image")
    image_bgr = to_bgr(image)
    tensor, transform = preprocess_for_model(image_bgr, model.image_size)
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    model = model.to(target_device)
    model.eval()
    with torch.inference_mode():
        reconstruction = model(tensor.to(target_device))
        error_tensor = torch.mean(torch.square(tensor.to(target_device) - reconstruction), dim=1)[0]
        error_square = error_tensor.detach().cpu().numpy().astype(np.float32)
    error_map = map_error_to_source(error_square, transform)
    normalized = np.clip(error_map, 0.0, 1.0)
    mask = (normalized >= threshold).astype(np.uint8) * 255
    if min(mask.shape) >= 3:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = np.zeros_like(mask)
    regions: list[DefectRegion] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        cv2.drawContours(filtered, [contour], -1, 255, thickness=cv2.FILLED)
        roi = normalized[y : y + height, x : x + width]
        regions.append(DefectRegion(x, y, width, height, int(round(area)), float(roi.mean()) if roi.size else 0.0))
    regions.sort(key=lambda region: region.area_pixels, reverse=True)
    heatmap = colorize_difference(np.clip(normalized * 255.0, 0, 255).astype(np.uint8))
    overlay = make_overlay(image_bgr, heatmap, alpha=0.38)
    annotated = annotate_regions(image_bgr, regions)
    count = filtered.size
    return AnomalyResult(inspection_image=image_bgr, anomaly_map=normalized, binary_mask=filtered, heatmap=heatmap, overlay=overlay, annotated_image=annotated, regions=regions, anomaly_score=float(normalized.mean()), affected_pixel_ratio=float(np.count_nonzero(filtered) / count) if count else 0.0, threshold=float(threshold), processing_time_ms=(perf_counter() - started) * 1000.0, metadata={"threshold_method": "caller supplied; calibrated on held-out normal images during training", "min_area_pixels": min_area, "source_dimensions": {"width": int(image_bgr.shape[1]), "height": int(image_bgr.shape[0])}, "preprocessing_transform": transform, **(metadata or {})})


def predict_with_checkpoint(checkpoint_path: str | Path, inspection: str | Path | np.ndarray, *, min_area: int = 100, device: str = "cpu") -> AnomalyResult:
    """Load a validated checkpoint and run inference at its stored threshold."""
    model, checkpoint_metadata = load_model(checkpoint_path, device=device)
    return predict_anomalies(model, inspection, threshold=float(checkpoint_metadata["threshold"]), min_area=min_area, device=device, metadata=checkpoint_metadata)
