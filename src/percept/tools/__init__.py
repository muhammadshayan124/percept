from percept.tools.base import ToolRegistry, ToolResult, VisionTool
from percept.tools.colors import DominantColorsTool
from percept.tools.compare import CompareRegionsTool
from percept.tools.crop import CropRegionTool
from percept.tools.edges import EdgeDensityTool
from percept.tools.faces import DetectFacesTool
from percept.tools.shapes import CountShapesTool
from percept.tools.sharpness import SharpnessTool

__all__ = [
    "CompareRegionsTool",
    "CountShapesTool",
    "CropRegionTool",
    "DetectFacesTool",
    "DominantColorsTool",
    "EdgeDensityTool",
    "SharpnessTool",
    "ToolRegistry",
    "ToolResult",
    "VisionTool",
]


def default_registry() -> ToolRegistry:
    """The standard set of tools a percept agent has access to."""
    return ToolRegistry(
        [
            CountShapesTool(),
            SharpnessTool(),
            DominantColorsTool(),
            EdgeDensityTool(),
            CompareRegionsTool(),
            CropRegionTool(),
            DetectFacesTool(),
        ]
    )
