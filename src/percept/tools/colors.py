from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from percept.tools.base import ToolResult, VisionTool


class DominantColorsTool(VisionTool):
    name = "dominant_colors"
    description = (
        "Extracts the k most dominant colors in the image via k-means clustering on "
        "pixel RGB values, along with each color's share of the image. Use this for "
        "questions about color composition, palette, or which color dominates."
    )

    def __init__(self, k: int = 3, seed: int = 42):
        self._k = k
        self._seed = seed

    def run(self, image: np.ndarray, **kwargs: Any) -> ToolResult:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pixels = rgb.reshape(-1, 3).astype(np.float32)

        k = min(self._k, len(np.unique(pixels.astype(np.uint8), axis=0)))
        if k == 0:
            return ToolResult(summary="Image has no pixels to analyze.", data={"colors": []})

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
        cv2.setRNGSeed(self._seed)
        _, labels, centers = cv2.kmeans(
            pixels, k, None, criteria, attempts=3, flags=cv2.KMEANS_RANDOM_CENTERS
        )

        counts = np.bincount(labels.flatten(), minlength=k)
        order = np.argsort(-counts)

        colors = []
        for idx in order:
            r, g, b = centers[idx].astype(int).tolist()
            colors.append(
                {
                    "rgb": [int(r), int(g), int(b)],
                    "hex": f"#{r:02x}{g:02x}{b:02x}",
                    "share": float(counts[idx] / len(pixels)),
                }
            )

        top = colors[0]
        return ToolResult(
            summary=f"Dominant color is {top['hex']} ({top['share'] * 100:.1f}% of pixels).",
            data={"colors": colors},
        )
