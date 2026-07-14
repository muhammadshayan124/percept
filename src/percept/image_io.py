"""Image loading helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_image(path: str | Path) -> np.ndarray:
    """Load an image as a BGR array (OpenCV convention).

    Raises:
        FileNotFoundError: if the path doesn't exist.
        ValueError: if the file exists but isn't a decodable image.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"image not found: {path}")

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"could not decode image: {path}")
    return image
