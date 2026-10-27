"""Generate deterministic, clearly synthetic product images for the demo."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def create_synthetic_product(width: int = 640, height: int = 480, seed: int = 17) -> np.ndarray:
    """Render a stylized machined metal component with deterministic texture."""
    rng = np.random.default_rng(seed); y, x = np.mgrid[0:height, 0:width]
    background = np.zeros((height, width, 3), dtype=np.float32)
    radial = np.sqrt(((x - width * 0.5) / width) ** 2 + ((y - height * 0.5) / height) ** 2)
    lighting = 27 + 16 * (1 - np.clip(radial * 1.7, 0, 1)) + 8 * np.sin(y / 29); noise = rng.normal(0, 2.0, (height, width))
    for channel, offset in enumerate((0, 2, 5)): background[:, :, channel] = lighting + noise + offset
    image = np.clip(background, 0, 255).astype(np.uint8); center = (width // 2, height // 2)
    cv2.circle(image, center, 178, (62, 69, 76), -1, cv2.LINE_AA)
    for radius in range(174, 120, -3):
        brightness = int(90 + (174 - radius) * 1.45 + 12 * np.sin(radius / 7)); color = (brightness - 8, brightness, min(255, brightness + 10)); cv2.circle(image, center, radius, color, 3, cv2.LINE_AA)
    cv2.circle(image, center, 118, (52, 59, 64), -1, cv2.LINE_AA); cv2.circle(image, center, 110, (146, 155, 163), 5, cv2.LINE_AA); cv2.circle(image, center, 94, (42, 49, 56), -1, cv2.LINE_AA); cv2.circle(image, center, 77, (16, 23, 29), 4, cv2.LINE_AA); cv2.circle(image, center, 65, (25, 33, 40), -1, cv2.LINE_AA)
    cv2.ellipse(image, center, (154, 36), -16, 190, 350, (210, 220, 227), 3, cv2.LINE_AA); cv2.ellipse(image, center, (151, 34), -16, 10, 170, (42, 49, 54), 3, cv2.LINE_AA)
    for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        bx = int(center[0] + 148 * np.cos(angle)); by = int(center[1] + 148 * np.sin(angle)); cv2.circle(image, (bx, by), 13, (21, 28, 34), -1, cv2.LINE_AA); cv2.circle(image, (bx - 2, by - 2), 9, (117, 127, 135), 2, cv2.LINE_AA); cv2.circle(image, (bx, by), 4, (8, 13, 18), -1, cv2.LINE_AA)
    for offset in range(-70, 71, 7): cv2.line(image, (center[0] + offset, center[1] - 88), (center[0] + offset + 7, center[1] + 88), (61, 70, 77), 1, cv2.LINE_AA)
    cv2.ellipse(image, (center[0] - 4, center[1] - 3), (172, 169), 0, 198, 320, (207, 215, 221), 2, cv2.LINE_AA)
    return image


def add_synthetic_scratch(image: np.ndarray) -> np.ndarray:
    """Add a visible, deterministic two-edge scratch to a copy of the image."""
    output = image.copy(); cv2.line(output, (359, 166), (405, 286), (238, 244, 248), 4, cv2.LINE_AA); cv2.line(output, (363, 166), (409, 286), (20, 25, 30), 2, cv2.LINE_AA); cv2.line(output, (361, 170), (405, 278), (174, 187, 198), 1, cv2.LINE_AA); return output


def generate_demo_data(output_dir: str | Path) -> dict[str, str]:
    """Create reference, clean, and defective images in a new/empty directory."""
    destination = Path(output_dir).expanduser()
    if destination.exists() and any(destination.iterdir()): raise FileExistsError(f"Demo output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True); reference = create_synthetic_product(); clean = reference.copy(); defective = add_synthetic_scratch(reference)
    paths = {"reference": destination / "reference_synthetic.png", "clean": destination / "inspection_clean_synthetic.png", "defective": destination / "inspection_scratch_synthetic.png"}
    for key, image in (("reference", reference), ("clean", clean), ("defective", defective)):
        if not cv2.imwrite(str(paths[key]), image): raise OSError(f"Could not write synthetic demo image: {paths[key]}")
    return {key: str(path) for key, path in paths.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output-dir", type=Path, default=Path("outputs/synthetic-demo-data")); args = parser.parse_args(); paths = generate_demo_data(args.output_dir); print("Synthetic demo images created (not real industrial photographs):"); [print(f"{key}: {path}") for key, path in paths.items()]; return 0


if __name__ == "__main__": raise SystemExit(main())
