from conftest import draw_circles, make_blank

from percept.tools.edges import EdgeDensityTool


def test_blank_image_has_near_zero_edge_density():
    result = EdgeDensityTool().run(make_blank())
    assert result.data["density"] == 0.0


def test_textured_image_has_higher_density_than_blank():
    textured = draw_circles(make_blank(200, 200), count=10, radius=10)
    blank = make_blank(200, 200)

    textured_density = EdgeDensityTool().run(textured).data["density"]
    blank_density = EdgeDensityTool().run(blank).data["density"]

    assert textured_density > blank_density


def test_more_shapes_generally_increase_edge_density():
    few = draw_circles(make_blank(300, 300), count=2, radius=15)
    many = draw_circles(make_blank(300, 300), count=20, radius=15)

    few_density = EdgeDensityTool().run(few).data["density"]
    many_density = EdgeDensityTool().run(many).data["density"]

    assert many_density > few_density
