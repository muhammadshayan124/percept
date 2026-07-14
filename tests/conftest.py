"""Shared synthetic-image fixtures. Every test image is generated
deterministically at test time, so ground truth (shape counts, blur levels,
colors) is always exactly known.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest


def make_blank(
    width: int = 200, height: int = 200, color: tuple[int, int, int] = (255, 255, 255)
) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = color
    return image


def draw_circles(
    image: np.ndarray, count: int, radius: int = 15, color: tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """Draws `count` non-overlapping filled circles on a grid."""
    image = image.copy()
    height, width = image.shape[:2]
    cols = max(1, int(np.ceil(np.sqrt(count))))
    spacing_x = width // (cols + 1)
    spacing_y = height // (cols + 1)

    drawn = 0
    for row in range(cols):
        for col in range(cols):
            if drawn >= count:
                return image
            cx = spacing_x * (col + 1)
            cy = spacing_y * (row + 1)
            cv2.circle(image, (cx, cy), radius, color, thickness=-1)
            drawn += 1
    return image


@pytest.fixture
def blank_image() -> np.ndarray:
    return make_blank()
