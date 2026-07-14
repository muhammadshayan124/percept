import cv2
from conftest import draw_circles, make_blank
from fastapi.testclient import TestClient

import percept.api as api_module
from percept.api import app
from percept.llm import FinalAnswerDecision, LLMUnavailableError, ToolCallDecision

client = TestClient(app)


class _FakeLLM:
    def __init__(self, decisions):
        self._decisions = list(decisions)

    def decide(self, task, tool_catalog, history):
        return self._decisions.pop(0)


def _image_bytes():
    image = draw_circles(make_blank(200, 200), count=4, radius=15)
    ok, buf = cv2.imencode(".png", image)
    assert ok
    return buf.tobytes()


def test_list_tools():
    response = client.get("/tools")
    assert response.status_code == 200
    names = {t["name"] for t in response.json()}
    assert "count_shapes" in names
    assert "detect_faces" in names


def test_run_tool_directly_without_any_llm():
    response = client.post(
        "/tools/count_shapes/run",
        files={"image": ("test.png", _image_bytes(), "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["data"]["count"] == 4


def test_run_tool_rejects_unknown_tool_name():
    response = client.post(
        "/tools/not_a_real_tool/run",
        files={"image": ("test.png", _image_bytes(), "image/png")},
    )
    assert response.status_code == 404


def test_run_tool_rejects_undecodable_upload():
    response = client.post(
        "/tools/count_shapes/run",
        files={"image": ("test.png", b"not an image", "image/png")},
    )
    assert response.status_code == 400


def test_analyze_runs_full_agent_loop(monkeypatch):
    monkeypatch.setattr(
        api_module,
        "get_llm_client",
        lambda provider, **kw: _FakeLLM(
            [
                ToolCallDecision(thought="count", tool_name="count_shapes", tool_args={}),
                FinalAnswerDecision(thought="done", answer="There are 4 shapes."),
            ]
        ),
    )

    response = client.post(
        "/analyze",
        data={"task": "How many shapes?", "provider": "anthropic"},
        files={"image": ("test.png", _image_bytes(), "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["final_answer"] == "There are 4 shapes."
    assert body["steps"][0]["tool"] == "count_shapes"
    assert body["steps"][0]["data"]["count"] == 4


def test_analyze_returns_503_when_provider_unavailable(monkeypatch):
    def _raise(provider, **kw):
        raise LLMUnavailableError("no API key")

    monkeypatch.setattr(api_module, "get_llm_client", _raise)

    response = client.post(
        "/analyze",
        data={"task": "task", "provider": "anthropic"},
        files={"image": ("test.png", _image_bytes(), "image/png")},
    )

    assert response.status_code == 503


class _DiesOnFirstCallLLM:
    """Reproduces providers like Ollama, whose connectivity check happens
    lazily on the first decide() call rather than at construction.
    """

    def decide(self, task, tool_catalog, history):
        raise LLMUnavailableError("could not reach Ollama at http://localhost:11434: timed out")


def test_analyze_returns_503_when_llm_fails_mid_loop_not_at_construction(monkeypatch):
    """Regression test: get_llm_client() succeeding doesn't guarantee the
    client stays reachable -- LLMUnavailableError raised from inside the
    agent loop must produce a clean 503, not an unhandled 500.
    """
    monkeypatch.setattr(api_module, "get_llm_client", lambda provider, **kw: _DiesOnFirstCallLLM())

    response = client.post(
        "/analyze",
        data={"task": "task", "provider": "ollama"},
        files={"image": ("test.png", _image_bytes(), "image/png")},
    )

    assert response.status_code == 503


def test_analyze_returns_partial_trace_when_iterations_exhausted(monkeypatch):
    monkeypatch.setattr(
        api_module,
        "get_llm_client",
        lambda provider, **kw: _FakeLLM(
            [
                ToolCallDecision(thought="loop", tool_name="count_shapes", tool_args={})
                for _ in range(10)
            ]
        ),
    )

    response = client.post(
        "/analyze",
        data={"task": "task", "provider": "anthropic", "max_iterations": 2},
        files={"image": ("test.png", _image_bytes(), "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["final_answer"] is None
    assert len(body["steps"]) == 2
