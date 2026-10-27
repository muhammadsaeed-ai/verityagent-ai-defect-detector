from __future__ import annotations

import cv2
import numpy as np
import pytest


@pytest.fixture
def simple_reference() -> np.ndarray:
    image = np.full((180, 240, 3), 75, dtype=np.uint8)
    cv2.rectangle(image, (35, 25), (205, 155), (130, 140, 150), -1)
    for center in ((78, 66), (162, 66), (78, 119), (162, 119)):
        cv2.circle(image, center, 13, (20, 25, 30), -1)
    cv2.line(image, (40, 90), (200, 90), (138, 145, 151), 1)
    return image


@pytest.fixture
def scratch_pair(simple_reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = simple_reference.copy(); inspection = reference.copy()
    cv2.line(inspection, (116, 48), (125, 117), (245, 248, 250), 4, cv2.LINE_AA)
    cv2.line(inspection, (120, 50), (129, 116), (18, 20, 24), 2, cv2.LINE_AA)
    return reference, inspection
