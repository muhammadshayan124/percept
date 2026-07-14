"""Pluggable LLM client for the agent's planning step.

Each call gets the task, the tool catalog, and the transcript so far, and
must return either a tool call to make or a final answer. Providers are
lazily imported so the base package has no hard dependency on any single
LLM SDK -- install `percept[anthropic]`, `percept[openai]`, or
`percept[gemini]` for the one you want, or set OLLAMA_HOST to use a local
model with no extra dependency at all.
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import httpx


class LLMUnavailableError(RuntimeError):
    """Raised when a provider can't run right now (missing SDK, missing API key)."""


@dataclass(frozen=True)
class ToolCallDecision:
    thought: str
    tool_name: str
    tool_args: dict[str, Any]


@dataclass(frozen=True)
class FinalAnswerDecision:
    thought: str
    answer: str


AgentDecision = ToolCallDecision | FinalAnswerDecision


SYSTEM_PROMPT = """You are a vision analysis agent. You answer questions about an image by \
calling tools that inspect it, one at a time, then reasoning over what they report back.

Respond with a single JSON object and nothing else. To call a tool:
{"thought": "why you're calling this tool", "action": "tool_name", "action_input": {...}}

When you have enough information to answer the user's task, respond instead with:
{"thought": "your reasoning", "final_answer": "the answer, written for the end user"}

Only call tools from the provided catalog, with kwargs matching what each tool documents.
"""


def build_planning_prompt(
    task: str, tool_catalog: list[dict[str, str]], history: list[dict[str, Any]]
) -> str:
    catalog_text = "\n".join(f"- {t['name']}: {t['description']}" for t in tool_catalog)
    history_text = "\n".join(
        f"[{i + 1}] Called {step['tool']}({step['args']}) -> {step['observation']}"
        for i, step in enumerate(history)
    )
    return (
        f"Task: {task}\n\n"
        f"Available tools:\n{catalog_text}\n\n"
        f"History so far:\n{history_text if history_text else '(none yet)'}\n\n"
        "What do you do next?"
    )


def parse_decision(text: str) -> AgentDecision:
    """Extract the JSON decision object from an LLM response, tolerating
    markdown code fences and surrounding prose.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object found in LLM response: {text!r}")

    payload = json.loads(match.group(0))
    thought = payload.get("thought", "")

    if "final_answer" in payload:
        return FinalAnswerDecision(thought=thought, answer=payload["final_answer"])
    if "action" in payload:
        return ToolCallDecision(
            thought=thought,
            tool_name=payload["action"],
            tool_args=payload.get("action_input", {}) or {},
        )
    raise ValueError(f"LLM response has neither 'action' nor 'final_answer': {payload!r}")


class LLMClient(ABC):
    @abstractmethod
    def decide(
        self, task: str, tool_catalog: list[dict[str, str]], history: list[dict[str, Any]]
    ) -> AgentDecision:
        """Ask the model what to do next given the task, tool catalog, and history so far."""


class AnthropicClient(LLMClient):
    def __init__(self, model: str = "claude-opus-4-8", api_key: str | None = None):
        try:
            import anthropic
        except ImportError as exc:
            raise LLMUnavailableError(
                "anthropic is not installed. Install it with: pip install 'percept[anthropic]'"
            ) from exc

        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise LLMUnavailableError("ANTHROPIC_API_KEY is not set")

        self._client = anthropic.Anthropic(api_key=key)
        self._model = model

    def decide(self, task, tool_catalog, history) -> AgentDecision:
        prompt = build_planning_prompt(task, tool_catalog, history)
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        return parse_decision(text)


class OpenAIClient(LLMClient):
    def __init__(self, model: str = "gpt-4o", api_key: str | None = None):
        try:
            import openai
        except ImportError as exc:
            raise LLMUnavailableError(
                "openai is not installed. Install it with: pip install 'percept[openai]'"
            ) from exc

        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise LLMUnavailableError("OPENAI_API_KEY is not set")

        self._client = openai.OpenAI(api_key=key)
        self._model = model

    def decide(self, task, tool_catalog, history) -> AgentDecision:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_planning_prompt(task, tool_catalog, history)},
            ],
        )
        return parse_decision(response.choices[0].message.content)


class GeminiClient(LLMClient):
    def __init__(self, model: str = "gemini-2.5-pro", api_key: str | None = None):
        try:
            from google import genai
        except ImportError as exc:
            raise LLMUnavailableError(
                "google-genai is not installed. Install it with: pip install 'percept[gemini]'"
            ) from exc

        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise LLMUnavailableError("GEMINI_API_KEY is not set")

        self._client = genai.Client(api_key=key)
        self._model = model

    def decide(self, task, tool_catalog, history) -> AgentDecision:
        prompt = f"{SYSTEM_PROMPT}\n\n{build_planning_prompt(task, tool_catalog, history)}"
        response = self._client.models.generate_content(model=self._model, contents=prompt)
        return parse_decision(response.text)


class OllamaClient(LLMClient):
    # Generous: a cold model load alone can take 20-30s on CPU before the
    # server even starts evaluating the prompt.
    DEFAULT_TIMEOUT_SECONDS = 180.0

    def __init__(
        self, model: str = "llama3.1", host: str | None = None, timeout: float | None = None
    ):
        self._host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self._model = model
        self._timeout = timeout or self.DEFAULT_TIMEOUT_SECONDS

    def decide(self, task, tool_catalog, history) -> AgentDecision:
        prompt = f"{SYSTEM_PROMPT}\n\n{build_planning_prompt(task, tool_catalog, history)}"
        try:
            response = httpx.post(
                f"{self._host}/api/generate",
                json={"model": self._model, "prompt": prompt, "stream": False},
                timeout=self._timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(f"could not reach Ollama at {self._host}: {exc}") from exc

        return parse_decision(response.json()["response"])


def get_llm_client(provider: str, **kwargs: Any) -> LLMClient:
    providers = {
        "anthropic": AnthropicClient,
        "openai": OpenAIClient,
        "gemini": GeminiClient,
        "ollama": OllamaClient,
    }
    cls = providers.get(provider)
    if cls is None:
        raise ValueError(f"unknown provider {provider!r}; choose from {sorted(providers)}")
    return cls(**kwargs)
