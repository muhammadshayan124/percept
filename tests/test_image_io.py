import cv2
import pytest
from conftest import make_blank

from percept.image_io import load_image


def test_loads_a_real_image_file(tmp_path):
    image = make_blank(50, 50, color=(0, 0, 255))
    path = tmp_path / "test.png"
    cv2.imwrite(str(path), image)

    loaded = load_image(path)

    assert loaded.shape == (50, 50, 3)
    assert tuple(loaded[0, 0]) == (0, 0, 255)


def test_raises_file_not_found_for_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "does_not_exist.png")


def test_raises_value_error_for_undecodable_file(tmp_path):
    path = tmp_path / "not_an_image.png"
    path.write_text("this is definitely not image data")

    with pytest.raises(ValueError, match="could not decode"):
        load_image(path)
