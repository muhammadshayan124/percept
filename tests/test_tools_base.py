import pytest

from percept.tools import CountShapesTool, DominantColorsTool, default_registry
from percept.tools.base import ToolRegistry


def test_registry_get_returns_tool_by_name():
    registry = ToolRegistry([CountShapesTool()])
    assert isinstance(registry.get("count_shapes"), CountShapesTool)


def test_registry_get_returns_none_for_unknown_tool():
    registry = ToolRegistry([CountShapesTool()])
    assert registry.get("nonexistent") is None


def test_registry_rejects_duplicate_names():
    with pytest.raises(ValueError, match="unique"):
        ToolRegistry([CountShapesTool(), CountShapesTool()])


def test_registry_catalog_includes_name_and_description():
    registry = ToolRegistry([CountShapesTool(), DominantColorsTool()])
    catalog = registry.catalog()

    names = {entry["name"] for entry in catalog}
    assert names == {"count_shapes", "dominant_colors"}
    for entry in catalog:
        assert entry["description"]


def test_default_registry_has_all_seven_tools():
    registry = default_registry()
    assert set(registry.names()) == {
        "count_shapes",
        "sharpness_score",
        "dominant_colors",
        "edge_density",
        "compare_regions",
        "crop_region",
        "detect_faces",
    }
