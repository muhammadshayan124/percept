"""Note on coverage: a synthetic image has no real face for the detector to
find, so these tests can only verify true-negative behavior and the response
shape, not detection accuracy on an actual face. Verify accuracy yourself
with real photos; see the README.
"""

import numpy as np
import pytest
from conftest import draw_circles, make_blank

from percept.tools.faces import DetectFacesTool


def test_blank_image_has_no_faces():
    result = DetectFacesTool().run(make_blank())
    assert result.data["count"] == 0
    assert result.data["faces"] == []


def test_non_face_shapes_are_not_detected_as_faces():
    image = draw_circles(make_blank(300, 300), count=4, radius=20)
    result = DetectFacesTool().run(image)
    assert result.data["count"] == 0


class _FakeYuNetDetector:
    """Stands in for cv2.FaceDetectorYN, whose methods are read-only on the
    real C-extension object and can't be monkeypatched directly.
    """

    def __init__(self, detections):
        self._detections = detections

    def setInputSize(self, size):
        pass

    def detect(self, image):
        return 1, self._detections


def test_result_shape_when_faces_are_found(monkeypatch):
    """Exercises the parsing path for a non-empty detection without depending
    on a real face actually being found, by stubbing the underlying detector.
    """
    tool = DetectFacesTool()
    fake_detection = np.array([[10.0, 20.0, 30.0, 40.0] + [0.0] * 10 + [0.91]], dtype=np.float32)
    monkeypatch.setattr(tool, "_detector", _FakeYuNetDetector(fake_detection))

    result = tool.run(make_blank(300, 300))

    assert result.data["count"] == 1
    face = result.data["faces"][0]
    assert face["bbox"] == [10, 20, 30, 40]
    assert face["confidence"] == pytest.approx(0.91)
