from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from percept.tools.base import ToolResult, VisionTool


class SharpnessTool(VisionTool):
    name = "sharpness_score"
    description = (
        "Measures how in-focus/sharp the image is using the variance of the Laplacian "
        "(a standard blur-detection metric: higher variance means more high-frequency "
        "detail, i.e. a sharper image). Use this for questions about whether an image "
        "is blurry, out of focus, or low quality."
    )

    # Chosen empirically: real photos typically score in the hundreds to low
    # thousands; well below this range indicates visible blur.
    BLUR_THRESHOLD = 100.0

    def run(self, image: np.ndarray, **kwargs: Any) -> ToolResult:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        is_blurry = variance < self.BLUR_THRESHOLD

        return ToolResult(
            summary=(
                f"Laplacian variance is {variance:.1f} "
                f"({'likely blurry' if is_blurry else 'likely sharp'}, "
                f"threshold={self.BLUR_THRESHOLD})."
            ),
            data={"variance": variance, "is_blurry": is_blurry},
        )
