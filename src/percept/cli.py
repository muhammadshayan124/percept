from __future__ import annotations

import json
import sys

import click

from percept.agent import AgentIterationLimitError, VisionAgent
from percept.image_io import load_image
from percept.llm import LLMUnavailableError, get_llm_client
from percept.tools import default_registry


@click.group()
def main() -> None:
    """percept: an autonomous agent that answers questions about an image by
    calling real computer-vision tools."""


@main.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.argument("task")
@click.option(
    "--provider", default="anthropic", show_default=True, help="anthropic, openai, gemini, ollama."
)
@click.option("--model", default=None, help="Override the provider's default model.")
@click.option(
    "--max-iterations", default=8, show_default=True, help="Tool-call budget before giving up."
)
@click.option(
    "--json-out", type=click.Path(), default=None, help="Write the full trace as JSON here."
)
def run(
    image_path: str,
    task: str,
    provider: str,
    model: str | None,
    max_iterations: int,
    json_out: str | None,
) -> None:
    """Run the agent on IMAGE_PATH with a natural-language TASK.

    Example: percept run photo.jpg "How many objects are on the table?"
    """
    image = load_image(image_path)
    kwargs = {"model": model} if model else {}
    try:
        llm = get_llm_client(provider, **kwargs)
    except LLMUnavailableError as exc:
        click.echo(f"provider {provider!r} unavailable: {exc}", err=True)
        sys.exit(1)

    agent = VisionAgent(llm, default_registry(), max_iterations=max_iterations)

    try:
        trace = agent.run(image, task)
    except AgentIterationLimitError as exc:
        click.echo(str(exc), err=True)
        trace = exc.trace
    except LLMUnavailableError as exc:
        click.echo(f"provider {provider!r} unavailable: {exc}", err=True)
        sys.exit(1)

    for i, step in enumerate(trace.steps, start=1):
        click.echo(f"[{i}] {step.tool}({step.args}) -> {step.observation}")

    if trace.final_answer:
        click.echo(f"\nAnswer: {trace.final_answer}")
    else:
        click.echo("\nNo final answer reached.", err=True)

    if json_out:
        payload = {
            "task": trace.task,
            "iterations": trace.iterations,
            "final_answer": trace.final_answer,
            "steps": [
                {"tool": s.tool, "args": s.args, "observation": s.observation, "data": s.data}
                for s in trace.steps
            ],
        }
        with open(json_out, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        click.echo(f"wrote {json_out}")

    if not trace.final_answer:
        sys.exit(1)


if __name__ == "__main__":
    main()
