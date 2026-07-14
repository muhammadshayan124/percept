import pytest

from percept.llm import FinalAnswerDecision, ToolCallDecision, build_planning_prompt, parse_decision


def test_parses_tool_call_decision():
    text = '{"thought": "let me count", "action": "count_shapes", "action_input": {}}'
    decision = parse_decision(text)

    assert isinstance(decision, ToolCallDecision)
    assert decision.thought == "let me count"
    assert decision.tool_name == "count_shapes"
    assert decision.tool_args == {}


def test_parses_final_answer_decision():
    text = '{"thought": "done", "final_answer": "There are 3 objects."}'
    decision = parse_decision(text)

    assert isinstance(decision, FinalAnswerDecision)
    assert decision.answer == "There are 3 objects."


def test_parses_tool_call_with_args():
    text = (
        '{"thought": "zoom in", "action": "crop_region", '
        '"action_input": {"x": 10, "y": 20, "width": 50, "height": 50}}'
    )
    decision = parse_decision(text)

    assert isinstance(decision, ToolCallDecision)
    assert decision.tool_args == {"x": 10, "y": 20, "width": 50, "height": 50}


def test_tolerates_markdown_code_fence():
    text = '```json\n{"thought": "ok", "final_answer": "yes"}\n```'
    decision = parse_decision(text)
    assert isinstance(decision, FinalAnswerDecision)


def test_tolerates_surrounding_prose():
    text = (
        "Sure, here is my decision:\n"
        '{"thought": "ok", "action": "sharpness_score", "action_input": {}}\n'
        "Let me know if that works."
    )
    decision = parse_decision(text)
    assert isinstance(decision, ToolCallDecision)
    assert decision.tool_name == "sharpness_score"


def test_raises_on_no_json_found():
    with pytest.raises(ValueError, match="no JSON object"):
        parse_decision("I don't know what to do.")


def test_raises_on_missing_action_and_final_answer():
    with pytest.raises(ValueError, match="neither"):
        parse_decision('{"thought": "hmm"}')


def test_build_planning_prompt_includes_task_and_tools():
    prompt = build_planning_prompt(
        task="How many objects?",
        tool_catalog=[{"name": "count_shapes", "description": "counts things"}],
        history=[],
    )
    assert "How many objects?" in prompt
    assert "count_shapes" in prompt
    assert "counts things" in prompt
    assert "(none yet)" in prompt


def test_build_planning_prompt_includes_history():
    prompt = build_planning_prompt(
        task="task",
        tool_catalog=[],
        history=[{"tool": "count_shapes", "args": {}, "observation": "Found 3 shapes."}],
    )
    assert "count_shapes" in prompt
    assert "Found 3 shapes." in prompt
