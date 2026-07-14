from conftest import draw_circles, make_blank

from percept.tools.shapes import CountShapesTool


def test_counts_known_number_of_circles():
    image = draw_circles(make_blank(300, 300), count=5, radius=20)
    tool = CountShapesTool()

    result = tool.run(image)

    assert result.data["count"] == 5
    assert len(result.data["shapes"]) == 5


def test_blank_image_has_no_shapes():
    tool = CountShapesTool()
    result = tool.run(make_blank())
    assert result.data["count"] == 0


def test_shapes_have_plausible_bounding_boxes():
    image = draw_circles(make_blank(200, 200), count=1, radius=25)
    tool = CountShapesTool()

    result = tool.run(image)

    shape = result.data["shapes"][0]
    x, y, w, h = shape["bbox"]
    # A filled circle of radius 25 should have a bounding box close to 50x50.
    assert 40 <= w <= 60
    assert 40 <= h <= 60


def test_min_area_filters_tiny_noise():
    image = draw_circles(make_blank(200, 200), count=3, radius=20)
    # Add a single tiny speck well below any reasonable min_area.
    image[5:7, 5:7] = (0, 0, 0)

    lenient_tool = CountShapesTool(min_area=1)
    strict_tool = CountShapesTool(min_area=50)

    assert lenient_tool.run(image).data["count"] == 4
    assert strict_tool.run(image).data["count"] == 3
