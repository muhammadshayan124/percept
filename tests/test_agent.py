import pytest
from conftest import draw_circles, make_blank

from percept.agent import AgentIterationLimitError, VisionAgent
from percept.llm import FinalAnswerDecision, LLMClient, ToolCallDecision
from percept.tools import CountShapesTool, ToolRegistry
from percept.tools.base import VisionTool


class _ScriptedLLM(LLMClient):
    """Returns a fixed sequence of decisions, one per call, regardless of
    what task/catalog/history it's given -- lets tests assert on the agent
    loop's control flow without a real model.
    """

    def __init__(self, decisions):
        self._decisions = list(decisions)
        self.calls = []

    def decide(self, task, tool_catalog, history):
        self.calls.append({"task": task, "history": list(history)})
        return self._decisions.pop(0)


class _AlwaysFailsTool(VisionTool):
    name = "always_fails"
    description = "a tool that always raises, for testing error recovery"

    def run(self, image, **kwargs):
        raise RuntimeError("simulated tool failure")


def test_agent_returns_final_answer_after_one_tool_call():
    llm = _ScriptedLLM(
        [
            ToolCallDecision(thought="count first", tool_name="count_shapes", tool_args={}),
            FinalAnswerDecision(thought="done", answer="There are 3 shapes."),
        ]
    )
    registry = ToolRegistry([CountShapesTool()])
    agent = VisionAgent(llm, registry)

    image = draw_circles(make_blank(200, 200), count=3, radius=15)
    trace = agent.run(image, "How many shapes are there?")

    assert trace.final_answer == "There are 3 shapes."
    assert trace.iterations == 2
    assert len(trace.steps) == 1
    assert trace.steps[0].tool == "count_shapes"
    assert trace.steps[0].data["count"] == 3


def test_agent_answers_immediately_with_no_tool_calls():
    llm = _ScriptedLLM([FinalAnswerDecision(thought="obvious", answer="It's a blank image.")])
    agent = VisionAgent(llm, ToolRegistry([CountShapesTool()]))

    trace = agent.run(make_blank(), "Describe this image.")

    assert trace.final_answer == "It's a blank image."
    assert trace.steps == []
    assert trace.iterations == 1


def test_agent_feeds_tool_observation_back_into_next_call():
    llm = _ScriptedLLM(
        [
            ToolCallDecision(thought="count", tool_name="count_shapes", tool_args={}),
            FinalAnswerDecision(thought="done", answer="ok"),
        ]
    )
    agent = VisionAgent(llm, ToolRegistry([CountShapesTool()]))

    agent.run(make_blank(), "task")

    second_call_history = llm.calls[1]["history"]
    assert len(second_call_history) == 1
    assert second_call_history[0]["tool"] == "count_shapes"
    assert "shape(s)" in second_call_history[0]["observation"]


def test_agent_recovers_from_unknown_tool_name():
    llm = _ScriptedLLM(
        [
            ToolCallDecision(thought="oops", tool_name="not_a_real_tool", tool_args={}),
            FinalAnswerDecision(thought="fallback", answer="I could not use that tool."),
        ]
    )
    agent = VisionAgent(llm, ToolRegistry([CountShapesTool()]))

    trace = agent.run(make_blank(), "task")

    assert trace.final_answer == "I could not use that tool."
    assert "unknown tool" in trace.steps[0].observation.lower()


def test_agent_recovers_from_tool_exception():
    llm = _ScriptedLLM(
        [
            ToolCallDecision(thought="try it", tool_name="always_fails", tool_args={}),
            FinalAnswerDecision(thought="fallback", answer="That tool failed."),
        ]
    )
    agent = VisionAgent(llm, ToolRegistry([_AlwaysFailsTool()]))

    trace = agent.run(make_blank(), "task")

    assert trace.final_answer == "That tool failed."
    assert "simulated tool failure" in trace.steps[0].observation


def test_agent_raises_when_iteration_budget_is_exhausted():
    llm = _ScriptedLLM(
        [ToolCallDecision(thought="loop", tool_name="count_shapes", tool_args={}) for _ in range(5)]
    )
    agent = VisionAgent(llm, ToolRegistry([CountShapesTool()]), max_iterations=3)

    with pytest.raises(AgentIterationLimitError) as exc_info:
        agent.run(make_blank(), "task")

    trace = exc_info.value.trace
    assert trace.final_answer is None
    assert trace.iterations == 3
    assert len(trace.steps) == 3
