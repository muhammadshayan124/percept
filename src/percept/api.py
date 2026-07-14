"""FastAPI app exposing the agent and individual vision tools over HTTP.

Tool endpoints work with no LLM configured at all (useful for trying the real
CV tools directly); the /analyze endpoint needs an LLM provider available.
"""

from __future__ import annotations

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from percept.agent import AgentIterationLimitError, VisionAgent
from percept.llm import LLMUnavailableError, get_llm_client
from percept.tools import default_registry

app = FastAPI(title="percept", description="An autonomous vision agent with real CV tools")

_registry = default_registry()


async def _read_image(file: UploadFile) -> np.ndarray:
    raw = await file.read()
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "could not decode uploaded file as an image")
    return image


class StepResponse(BaseModel):
    tool: str
    args: dict
    observation: str
    data: dict


class TraceResponse(BaseModel):
    task: str
    iterations: int
    final_answer: str | None
    steps: list[StepResponse]


@app.get("/tools")
def list_tools() -> list[dict[str, str]]:
    return _registry.catalog()


# B008: FastAPI's DI reads File()/Form() defaults at decoration time, so the
# call-in-default pattern bugbear normally flags is required here, not a bug.
@app.post("/tools/{tool_name}/run")
async def run_tool(tool_name: str, image: UploadFile = File(...)) -> dict:  # noqa: B008
    tool = _registry.get(tool_name)
    if tool is None:
        raise HTTPException(404, f"unknown tool {tool_name!r}. Available: {_registry.names()}")

    img = await _read_image(image)
    try:
        result = tool.run(img)
    except Exception as exc:
        raise HTTPException(400, f"tool failed: {exc}") from exc

    return {"summary": result.summary, "data": result.data}


@app.post("/analyze", response_model=TraceResponse)
async def analyze(
    task: str = Form(...),
    provider: str = Form("anthropic"),
    model: str | None = Form(None),
    max_iterations: int = Form(8),
    image: UploadFile = File(...),  # noqa: B008 -- see run_tool's comment above
) -> TraceResponse:
    img = await _read_image(image)

    kwargs = {"model": model} if model else {}
    try:
        llm = get_llm_client(provider, **kwargs)
    except LLMUnavailableError as exc:
        raise HTTPException(503, f"provider {provider!r} unavailable: {exc}") from exc

    agent = VisionAgent(llm, _registry, max_iterations=max_iterations)

    try:
        trace = agent.run(img, task)
    except AgentIterationLimitError as exc:
        trace = exc.trace
    except LLMUnavailableError as exc:
        raise HTTPException(503, f"provider {provider!r} unavailable: {exc}") from exc

    return TraceResponse(
        task=trace.task,
        iterations=trace.iterations,
        final_answer=trace.final_answer,
        steps=[
            StepResponse(
                tool=s.tool, args=s.args, observation=s.observation, data=_json_safe(s.data)
            )
            for s in trace.steps
        ],
    )


def _json_safe(data: dict) -> dict:
    """Tool data dicts may carry numpy arrays (e.g. crop_region's cropped
    image); strip anything that isn't JSON-serializable rather than crashing
    the response.
    """
    return {k: v for k, v in data.items() if not isinstance(v, np.ndarray)}
