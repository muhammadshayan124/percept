# percept

An autonomous, ReAct-style agent that answers questions about an image by choosing and chaining real computer-vision tools, instead of just describing what a single model saw.

```
$ percept run photo.png "How many objects are in this image?" --provider ollama --model qwen2.5:7b

[1] count_shapes({}) -> Found 4 shape(s) with area >= 30px.
[2] edge_density({}) -> Edge density is 0.67% of pixels.
[3] dominant_colors({'k': 5}) -> Dominant color is #ffffff (90.6% of pixels).

Answer: There are 4 objects in the image.
```
*(real output from a local `qwen2.5:7b` run, unedited — the model chose to double-check with two extra tools before answering, which is realistic agent behavior, not scripted)*

## Why

Most "vision agent" demos wrap a single vision-language model call and call it a day. `percept` is closer to how a real analyst would work an image: pick a tool for the question at hand (count shapes? check sharpness? compare two images? crop and zoom?), read what it reports back, and decide whether that's enough or another tool is needed. The planning model never sees raw pixels — it only sees tool descriptions and structured observations, which makes the reasoning traceable and the tools independently testable.

## Tools

| Tool | What it does |
|---|---|
| `count_shapes` | Contour-based blob/object counting |
| `sharpness_score` | Laplacian-variance blur detection |
| `dominant_colors` | k-means color palette extraction |
| `edge_density` | Canny edge density (scene complexity/clutter) |
| `compare_regions` | Localizes differing regions between two images |
| `crop_region` | Crops to a bounding box, e.g. to zoom into an earlier tool's result |
| `detect_faces` | Face detection via YuNet (a small ONNX DNN model, bundled with the package) |

All seven are real OpenCV/classical-CV operations, not stubs — see [Tests](#tests) for how each is verified against synthetic images with known ground truth.

**A deliberate scoping choice:** these tools are fast, dependency-light, and need no GPU or model download beyond the ~230KB face detector bundled in the package. That was a real tradeoff against heavier deep-learning object detectors (YOLO, Faster R-CNN); for a tool-use agent, reliability and speed matter more than state-of-the-art accuracy on any one tool. `VisionTool` is a small interface (see `tools/base.py`), so swapping in a heavier detector later is a contained change.

## Install

```bash
pip install percept                # core tools + CLI + API, no LLM SDK required
pip install "percept[anthropic]"   # + Anthropic provider
pip install "percept[openai]"      # + OpenAI provider
pip install "percept[gemini]"      # + Gemini provider
# Ollama needs no extra dependency -- just a running local server
```

## Quick start

```bash
export ANTHROPIC_API_KEY=...
percept run photo.jpg "Is this photo blurry?" --provider anthropic
```

Or fully local with [Ollama](https://ollama.com):

```bash
ollama pull qwen2.5:7b
percept run photo.jpg "How many faces are in this photo?" --provider ollama --model qwen2.5:7b
```

Add `--json-out trace.json` to save the full step-by-step trace, including every tool's structured data (bounding boxes, scores, colors), not just the final answer.

## Try the tools without any LLM

Every tool works standalone, no API key or model needed:

```python
from percept.image_io import load_image
from percept.tools import CountShapesTool

image = load_image("photo.jpg")
result = CountShapesTool().run(image)
print(result.summary, result.data)
```

Or over HTTP:

```bash
uvicorn percept.api:app --reload
curl -X POST localhost:8000/tools/sharpness_score/run -F "image=@photo.jpg"
```

## As a service

```bash
curl -X POST localhost:8000/analyze \
  -F "task=How many objects are on the table?" \
  -F "provider=anthropic" \
  -F "image=@photo.jpg"
```

Returns the full trace (every tool call, its observation, and the final answer) as JSON.

## Adding a tool

Implement `VisionTool` (`tools/base.py`): a `name`, a `description` the planning model reads to decide when to use it, and a `run(image, **kwargs) -> ToolResult`. Register it in `tools/__init__.py`'s `default_registry()`.

## Tests

64 tests. Every CV tool is checked against synthetically generated images with exactly known ground truth (draw N circles, assert the counter finds N; blur a textured image, assert sharpness score drops; draw a rectangle diff, assert `compare_regions` localizes it) — not just "does it run without crashing." The agent loop is tested against a scripted fake LLM covering the full decision tree: normal completion, immediate answers, unknown tool names, tool exceptions, and iteration-budget exhaustion, all without needing API credits to run in CI.

```bash
pip install -e ".[dev]"
pytest
ruff check src tests
ruff format src tests
```

## Attribution

The bundled face detector (`percept/data/face_detection_yunet_2023mar.onnx`) is YuNet from the [OpenCV Zoo](https://github.com/opencv/opencv_zoo), licensed Apache 2.0.
