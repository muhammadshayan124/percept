from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from percept.tools.base import ToolResult, VisionTool


class CompareRegionsTool(VisionTool):
    name = "compare_regions"
    description = (
        "Compares the primary image against a second image of the same size and "
        "localizes the regions that differ, via absolute pixel difference. Use this "
        "for before/after comparisons or spot-the-difference questions. Requires a "
        "`other_image` kwarg (a second image array)."
    )

    def __init__(self, diff_threshold: int = 30, min_area: int = 40):
        self._diff_threshold = diff_threshold
        self._min_area = min_area

    def run(
        self, image: np.ndarray, other_image: np.ndarray | None = None, **kwargs: Any
    ) -> ToolResult:
        if other_image is None:
            raise ValueError("compare_regions requires an `other_image` kwarg")
        if image.shape != other_image.shape:
            raise ValueError(
                f"images must be the same shape to compare, "
                f"got {image.shape} vs {other_image.shape}"
            )

        gray_a = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(other_image, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray_a, gray_b)
        diff = cv2.GaussianBlur(diff, (5, 5), 0)
        _, mask = cv2.threshold(diff, self._diff_threshold, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self._min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            regions.append({"bbox": [int(x), int(y), int(w), int(h)], "area": float(area)})

        regions.sort(key=lambda r: -r["area"])
        changed_fraction = float(np.count_nonzero(mask) / mask.size)

        return ToolResult(
            summary=(
                f"Found {len(regions)} differing region(s), "
                f"{changed_fraction * 100:.2f}% of pixels changed."
            ),
            data={"regions": regions, "changed_fraction": changed_fraction},
        )
