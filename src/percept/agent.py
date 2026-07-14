"""The ReAct-style planning loop: ask the LLM what to do, run the tool it
picks, feed the observation back, repeat until it gives a final answer or
the iteration budget runs out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from percept.llm import FinalAnswerDecision, LLMClient, ToolCallDecision
from percept.tools.base import ToolRegistry


@dataclass(frozen=True)
class AgentStep:
    tool: str
    args: dict[str, Any]
    observation: str
    data: dict[str, Any]


@dataclass(frozen=True)
class AgentTrace:
    task: str
    steps: list[AgentStep]
    final_answer: str | None
    iterations: int


class AgentIterationLimitError(RuntimeError):
    """Raised when the agent exhausts its iteration budget without reaching
    a final answer. Carries the partial trace so callers can inspect what
    was tried.
    """

    def __init__(self, trace: AgentTrace):
        super().__init__(f"agent did not reach a final answer within {trace.iterations} iterations")
        self.trace = trace


class VisionAgent:
    def __init__(self, llm: LLMClient, registry: ToolRegistry, max_iterations: int = 8):
        self._llm = llm
        self._registry = registry
        self._max_iterations = max_iterations

    def run(self, image: np.ndarray, task: str) -> AgentTrace:
        steps: list[AgentStep] = []
        history: list[dict[str, Any]] = []
        catalog = self._registry.catalog()

        for iteration in range(1, self._max_iterations + 1):
            decision = self._llm.decide(task, catalog, history)

            if isinstance(decision, FinalAnswerDecision):
                return AgentTrace(
                    task=task, steps=steps, final_answer=decision.answer, iterations=iteration
                )

            step = self._execute(decision, image)
            steps.append(step)
            history.append({"tool": step.tool, "args": step.args, "observation": step.observation})

        raise AgentIterationLimitError(
            AgentTrace(task=task, steps=steps, final_answer=None, iterations=self._max_iterations)
        )

    def _execute(self, decision: ToolCallDecision, image: np.ndarray) -> AgentStep:
        tool = self._registry.get(decision.tool_name)
        if tool is None:
            available = ", ".join(self._registry.names())
            return AgentStep(
                tool=decision.tool_name,
                args=decision.tool_args,
                observation=(
                    f"Error: unknown tool {decision.tool_name!r}. Available tools: {available}."
                ),
                data={},
            )

        try:
            result = tool.run(image, **decision.tool_args)
        except Exception as exc:
            return AgentStep(
                tool=decision.tool_name,
                args=decision.tool_args,
                observation=f"Error running {decision.tool_name}: {exc}",
                data={},
            )

        return AgentStep(
            tool=decision.tool_name,
            args=decision.tool_args,
            observation=result.summary,
            data=result.data,
        )
