"""Common interface every vision tool implements, plus the registry the agent
picks tools from at planning time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ToolResult:
    """What a tool hands back to the agent loop as an observation."""

    summary: str
    """Short, LLM-readable description of what the tool found."""

    data: dict[str, Any] = field(default_factory=dict)
    """Structured data (counts, scores, coordinates) for programmatic use."""


class VisionTool(ABC):
    """One callable vision operation."""

    name: str
    description: str
    """Shown to the LLM when deciding which tool to call -- keep it precise
    about what the tool measures and when it's the right choice."""

    @abstractmethod
    def run(self, image: np.ndarray, **kwargs: Any) -> ToolResult:
        """Run the tool against a BGR image (OpenCV convention) and return a
        result. Implementations should not raise on malformed input where a
        sensible empty/zero result is possible; reserve exceptions for
        programmer errors (wrong image shape, missing required kwargs).
        """


class ToolRegistry:
    """Maps tool names to instances and renders the catalog the agent's
    planning prompt shows the LLM.
    """

    def __init__(self, tools: list[VisionTool]):
        by_name = {tool.name: tool for tool in tools}
        if len(by_name) != len(tools):
            raise ValueError("tool names must be unique")
        self._tools = by_name

    def get(self, name: str) -> VisionTool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def catalog(self) -> list[dict[str, str]]:
        return [{"name": t.name, "description": t.description} for t in self._tools.values()]
