"""Optional ECC-based alignment for small affine camera shifts."""

from __future__ import annotations

import cv2
import numpy as np

from .image_io import to_gray


def align_to_reference(reference: np.ndarray, inspection: np.ndarray, *, iterations: int = 100, epsilon: float = 1e-6) -> tuple[np.ndarray, dict[str, object]]:
    """Align an inspection image to a reference using ECC affine registration."""
    if reference.shape[:2] != inspection.shape[:2]:
        raise ValueError("Images must have matching dimensions before alignment")
    ref_gray = to_gray(reference)
    ins_gray = to_gray(inspection)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, epsilon)
    try:
        correlation, warp = cv2.findTransformECC(ref_gray, ins_gray, warp, cv2.MOTION_AFFINE, criteria, None, 5)
        aligned = cv2.warpAffine(inspection, warp, (reference.shape[1], reference.shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
        return aligned, {"alignment_requested": True, "alignment_success": True, "alignment_method": "ECC affine", "alignment_correlation": float(correlation), "alignment_warp_matrix": warp.astype(float).tolist()}
    except (cv2.error, ValueError) as exc:
        return inspection.copy(), {"alignment_requested": True, "alignment_success": False, "alignment_method": "ECC affine", "alignment_error": str(exc).splitlines()[0][:180]}
