import cv2
from conftest import draw_circles, make_blank

from percept.tools.sharpness import SharpnessTool


def _textured_image():
    """A blank image alone has zero variance regardless of blur, so give the
    sharpness detector real texture to lose when blurred.
    """
    return draw_circles(make_blank(200, 200), count=12, radius=8)


def test_blurring_reduces_sharpness_score():
    sharp = _textured_image()
    blurred = cv2.GaussianBlur(sharp, (15, 15), 0)
    tool = SharpnessTool()

    sharp_score = tool.run(sharp).data["variance"]
    blurred_score = tool.run(blurred).data["variance"]

    assert blurred_score < sharp_score


def test_heavily_blurred_image_is_flagged_blurry():
    image = cv2.GaussianBlur(_textured_image(), (25, 25), 0)
    result = SharpnessTool().run(image)
    assert result.data["is_blurry"] is True


def test_sharp_textured_image_is_not_flagged_blurry():
    result = SharpnessTool().run(_textured_image())
    assert result.data["is_blurry"] is False


def test_blank_image_has_zero_variance():
    result = SharpnessTool().run(make_blank())
    assert result.data["variance"] == 0.0
    assert result.data["is_blurry"] is True
