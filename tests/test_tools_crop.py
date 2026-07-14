from conftest import make_blank

from percept.tools.crop import CropRegionTool


def test_crops_to_requested_dimensions():
    image = make_blank(200, 200)
    result = CropRegionTool().run(image, x=10, y=20, width=50, height=30)

    assert result.data["image"].shape[:2] == (30, 50)
    assert result.data["bbox"] == [10, 20, 50, 30]


def test_crop_preserves_pixel_content():
    image = make_blank(100, 100, color=(0, 0, 0))
    image[10:20, 10:20] = (255, 255, 255)

    result = CropRegionTool().run(image, x=10, y=10, width=10, height=10)

    assert (result.data["image"] == 255).all()


def test_out_of_bounds_crop_is_clamped_not_errored():
    image = make_blank(50, 50)
    result = CropRegionTool().run(image, x=40, y=40, width=100, height=100)

    # Clamped to the remaining 10x10 region rather than raising or padding.
    assert result.data["image"].shape[:2] == (10, 10)


def test_default_crop_is_full_image():
    image = make_blank(60, 40)
    result = CropRegionTool().run(image)
    assert result.data["image"].shape[:2] == (40, 60)
