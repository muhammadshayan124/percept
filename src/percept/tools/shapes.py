from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from percept.tools.base import ToolResult, VisionTool


class CountShapesTool(VisionTool):
    name = "count_shapes"
    description = (
        "Counts distinct solid shapes/blobs in the image (e.g. objects on a plain "
        "background, dots, cells). Returns a count and each shape's bounding box and "
        "area. Best for questions like 'how many objects are there'."
    )

    def __init__(self, min_area: int = 30):
        self._min_area = min_area

    def run(self, image: np.ndarray, **kwargs: Any) -> ToolResult:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        shapes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self._min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            shapes.append({"bbox": [int(x), int(y), int(w), int(h)], "area": float(area)})

        shapes.sort(key=lambda s: -s["area"])
        return ToolResult(
            summary=f"Found {len(shapes)} shape(s) with area >= {self._min_area}px.",
            data={"count": len(shapes), "shapes": shapes},
        )
