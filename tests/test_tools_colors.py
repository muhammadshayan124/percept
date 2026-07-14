import pytest
from conftest import make_blank

from percept.tools.colors import DominantColorsTool


def test_single_color_image_reports_that_color():
    image = make_blank(100, 100, color=(0, 0, 255))  # BGR red
    result = DominantColorsTool(k=3).run(image)

    top = result.data["colors"][0]
    assert top["hex"] == "#ff0000"
    assert top["share"] > 0.99


def test_majority_color_ranks_first():
    image = make_blank(100, 100, color=(0, 255, 0))  # BGR green background
    image[0:20, 0:20] = (255, 0, 0)  # small blue patch

    result = DominantColorsTool(k=2).run(image)

    assert result.data["colors"][0]["hex"] == "#00ff00"
    assert result.data["colors"][0]["share"] > result.data["colors"][1]["share"]


def test_shares_sum_to_approximately_one():
    image = make_blank(80, 80, color=(0, 0, 255))
    image[:40, :] = (255, 0, 0)

    result = DominantColorsTool(k=2).run(image)

    total_share = sum(c["share"] for c in result.data["colors"])
    assert total_share == pytest.approx(1.0)
