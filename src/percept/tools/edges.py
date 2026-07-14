from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from percept.tools.base import ToolResult, VisionTool


class EdgeDensityTool(VisionTool):
    name = "edge_density"
    description = (
        "Runs Canny edge detection and reports what fraction of pixels are edges. Low "
        "density means a plain/uniform image (e.g. a blank wall, sky, or solid "
        "background); high density means a busy/textured/cluttered scene. Use this for "
        "questions about visual complexity or clutter."
    )

    def __init__(self, low_threshold: int = 50, high_threshold: int = 150):
        self._low = low_threshold
        self._high = high_threshold

    def run(self, image: np.ndarray, **kwargs: Any) -> ToolResult:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self._low, self._high)
        density = float(np.count_nonzero(edges) / edges.size)

        return ToolResult(
            summary=f"Edge density is {density * 100:.2f}% of pixels.",
            data={"density": density},
        )
