import json

import cv2
from click.testing import CliRunner
from conftest import draw_circles, make_blank

import percept.cli as cli_module
from percept.llm import FinalAnswerDecision, LLMUnavailableError, ToolCallDecision


class _FakeLLM:
    def __init__(self, decisions):
        self._decisions = list(decisions)

    def decide(self, task, tool_catalog, history):
        return self._decisions.pop(0)


def _write_test_image(tmp_path):
    image = draw_circles(make_blank(200, 200), count=4, radius=15)
    path = tmp_path / "test.png"
    cv2.imwrite(str(path), image)
    return path


def test_run_command_prints_steps_and_answer(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cli_module,
        "get_llm_client",
        lambda provider, **kw: _FakeLLM(
            [
                ToolCallDecision(thought="count", tool_name="count_shapes", tool_args={}),
                FinalAnswerDecision(thought="done", answer="There are 4 shapes."),
            ]
        ),
    )
    image_path = _write_test_image(tmp_path)

    runner = CliRunner()
    result = runner.invoke(cli_module.main, ["run", str(image_path), "How many shapes?"])

    assert result.exit_code == 0, result.output
    assert "count_shapes" in result.output
    assert "There are 4 shapes." in result.output


def test_run_command_writes_json_trace(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cli_module,
        "get_llm_client",
        lambda provider, **kw: _FakeLLM([FinalAnswerDecision(thought="ok", answer="Blank image.")]),
    )
    image_path = _write_test_image(tmp_path)
    json_path = tmp_path / "trace.json"

    runner = CliRunner()
    result = runner.invoke(
        cli_module.main,
        ["run", str(image_path), "Describe it.", "--json-out", str(json_path)],
    )

    assert result.exit_code == 0, result.output
    data = json.loads(json_path.read_text())
    assert data["final_answer"] == "Blank image."
    assert data["iterations"] == 1


def test_run_command_exits_nonzero_when_provider_unavailable(tmp_path, monkeypatch):
    def _raise(provider, **kw):
        raise LLMUnavailableError("ANTHROPIC_API_KEY is not set")

    monkeypatch.setattr(cli_module, "get_llm_client", _raise)
    image_path = _write_test_image(tmp_path)

    runner = CliRunner()
    result = runner.invoke(cli_module.main, ["run", str(image_path), "task"])

    assert result.exit_code == 1
    assert "unavailable" in result.output


class _DiesOnFirstCallLLM:
    """Reproduces providers like Ollama, whose connectivity check happens
    lazily on the first decide() call rather than at construction (unlike
    the API-key-based providers, which fail fast in get_llm_client itself).
    """

    def decide(self, task, tool_catalog, history):
        raise LLMUnavailableError("could not reach Ollama at http://localhost:11434: timed out")


def test_run_command_exits_cleanly_when_llm_fails_mid_loop_not_at_construction(
    tmp_path, monkeypatch
):
    """Regression test: get_llm_client() succeeding doesn't guarantee the
    client stays reachable -- LLMUnavailableError raised from inside the
    agent loop must be caught too, not just around client construction.
    """
    monkeypatch.setattr(cli_module, "get_llm_client", lambda provider, **kw: _DiesOnFirstCallLLM())
    image_path = _write_test_image(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        cli_module.main, ["run", str(image_path), "task", "--provider", "ollama"]
    )

    # CliRunner reports our own sys.exit(1) as a SystemExit "exception"; that's
    # the expected clean-exit path. Anything else means the LLMUnavailableError
    # propagated unhandled instead of being caught.
    if result.exception is not None:
        assert isinstance(result.exception, SystemExit), (
            f"unhandled exception: {result.exception!r}"
        )
    assert result.exit_code == 1, result.output
    assert "unavailable" in result.output


def test_run_command_exits_nonzero_when_iteration_budget_exhausted(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cli_module,
        "get_llm_client",
        lambda provider, **kw: _FakeLLM(
            [
                ToolCallDecision(thought="loop", tool_name="count_shapes", tool_args={})
                for _ in range(5)
            ]
        ),
    )
    image_path = _write_test_image(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        cli_module.main, ["run", str(image_path), "task", "--max-iterations", "2"]
    )

    assert result.exit_code == 1
    assert "No final answer reached." in result.output
