from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from percept.tools.base import ToolResult, VisionTool

_MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "face_detection_yunet_2023mar.onnx"


class DetectFacesTool(VisionTool):
    name = "detect_faces"
    description = (
        "Detects human faces using a lightweight DNN face detector (YuNet) and returns "
        "each face's bounding box plus a confidence score. Use this for questions about "
        "how many people/faces are in the image, or where they are."
    )

    def __init__(self, score_threshold: float = 0.7):
        if not _MODEL_PATH.exists():
            raise FileNotFoundError(
                f"face detection model not found at {_MODEL_PATH}. "
                "It should be bundled with the package under percept/data/."
            )
        self._detector = cv2.FaceDetectorYN_create(
            str(_MODEL_PATH), "", (0, 0), score_threshold=score_threshold
        )

    def run(self, image: np.ndarray, **kwargs: Any) -> ToolResult:
        height, width = image.shape[:2]
        self._detector.setInputSize((width, height))
        _, detections = self._detector.detect(image)

        faces = []
        if detections is not None:
            for row in detections:
                x, y, w, h = row[:4]
                score = float(row[14])
                faces.append(
                    {
                        "bbox": [int(x), int(y), int(w), int(h)],
                        "confidence": score,
                    }
                )

        return ToolResult(
            summary=f"Detected {len(faces)} face(s).",
            data={"count": len(faces), "faces": faces},
        )
