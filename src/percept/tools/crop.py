from __future__ import annotations

from typing import Any

import numpy as np

from percept.tools.base import ToolResult, VisionTool


class CropRegionTool(VisionTool):
    name = "crop_region"
    description = (
        "Crops the image to a bounding box, given `x`, `y`, `width`, and `height` "
        "kwargs (in pixels, origin top-left). Use this to focus a later tool call "
        "(e.g. sharpness_score or dominant_colors) on a specific part of the image "
        "identified by an earlier tool call's bounding box."
    )

    def run(
        self,
        image: np.ndarray,
        x: int = 0,
        y: int = 0,
        width: int | None = None,
        height: int | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        img_h, img_w = image.shape[:2]
        width = img_w - x if width is None else width
        height = img_h - y if height is None else height

        x0 = max(0, min(x, img_w))
        y0 = max(0, min(y, img_h))
        x1 = max(x0, min(x + width, img_w))
        y1 = max(y0, min(y + height, img_h))

        cropped = image[y0:y1, x0:x1]
        return ToolResult(
            summary=f"Cropped to region ({x0}, {y0}, {x1 - x0}x{y1 - y0}).",
            data={"bbox": [x0, y0, x1 - x0, y1 - y0], "image": cropped},
        )
