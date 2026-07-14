import cv2
import pytest
from conftest import make_blank

from percept.tools.compare import CompareRegionsTool


def test_identical_images_have_no_differing_regions():
    image = make_blank(200, 200)
    result = CompareRegionsTool().run(image, other_image=image.copy())

    assert result.data["regions"] == []
    assert result.data["changed_fraction"] == 0.0


def test_localizes_a_single_changed_region():
    before = make_blank(200, 200)
    after = before.copy()
    cv2.rectangle(after, (50, 50), (100, 100), (0, 0, 0), thickness=-1)

    result = CompareRegionsTool().run(before, other_image=after)

    assert len(result.data["regions"]) == 1
    x, y, w, h = result.data["regions"][0]["bbox"]
    # The drawn rectangle is 50x50 at (50,50); allow slack for blur/threshold.
    assert 40 <= w <= 60
    assert 40 <= h <= 60
    assert 40 <= x <= 60
    assert 40 <= y <= 60


def test_requires_other_image_kwarg():
    with pytest.raises(ValueError, match="other_image"):
        CompareRegionsTool().run(make_blank())


def test_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        CompareRegionsTool().run(make_blank(200, 200), other_image=make_blank(100, 100))
