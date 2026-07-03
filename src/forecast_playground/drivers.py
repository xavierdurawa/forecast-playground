"""Model drivers: one uniform interface over provider chat APIs.

The agent loop is provider-agnostic; each driver hides one API's quirks and returns
a normalized :class:`Turn`. Two are provided:

- :class:`OpenAIDriver` — speaks the OpenAI chat-completions API, which is the de
  facto standard: OpenAI, vLLM, Ollama, Together, Groq, DeepSeek, Mistral, and most
  local servers all expose it. Point ``base_url`` at any of them; no new dependency
  beyond the ``openai`` SDK, no per-provider adapter.
- :class:`AnthropicDriver` — the native Anthropic / Bedrock Messages API, so an
  existing ``AnthropicBedrock`` setup keeps working unchanged.

A driver converts the harness's neutral tool defs + message history into the
provider's shape, calls the model, and parses the reply into a ``Turn``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

# A tool the model asked to call, normalized across providers.
@dataclass
class ToolInvocation:
    id: str
    name: str
    args: dict[str, Any]


@dataclass
class Turn:
    """One model response, normalized.

    Attributes:
        text: Any assistant text in the reply.
        tool_calls: Tools the model wants run this turn (empty if it's done).
        stop_reason: Provider stop reason, normalized to include ``"tool_use"`` when
            tools were requested and ``"stop"``/``"length"`` otherwise.
        raw: The provider's raw assistant message, appended verbatim to history so
            the follow-up request is well-formed.
    """

    text: str
    tool_calls: list[ToolInvocation]
    stop_reason: str
    raw: Any = None
    meta: dict[str, Any] = field(default_factory=dict)


class Driver(Protocol):
    """Uniform model interface used by the agent loop."""

    def start(self, question: str) -> list[Any]:
        """Return the initial message history for ``question``."""
        ...

    def step(
        self,
        model: str,
        system: str,
        messages: list[Any],
        tools: list[dict] | None,
        max_tokens: int,
    ) -> Turn:
        """Call the model once; append the assistant reply to ``messages`` in place."""
        ...

    def add_tool_results(
        self, messages: list[Any], results: list[tuple[ToolInvocation, str]]
    ) -> None:
        """Append tool results to ``messages`` in the provider's expected shape."""
        ...

    def add_user(self, messages: list[Any], text: str) -> None:
        """Append a plain user message (used to force a final answer)."""
        ...


# ---------------------------------------------------------------------------
# OpenAI-compatible driver (the universal path)
# ---------------------------------------------------------------------------

class OpenAIDriver:
    """Driver for any OpenAI-compatible chat-completions endpoint.

    Args:
        client: An ``openai.OpenAI`` client. If None, one is built from ``base_url``
            / ``api_key`` (or the standard ``OPENAI_*`` env vars).
        base_url: Point at a non-OpenAI server (vLLM/Ollama/Together/...). Optional.
        api_key: Overrides the env var. Local servers often accept any string.
    """

    def __init__(self, client: Any = None, base_url: str | None = None, api_key: str | None = None):
        if client is None:
            from openai import OpenAI

            kwargs: dict[str, Any] = {}
            if base_url:
                kwargs["base_url"] = base_url
            if api_key:
                kwargs["api_key"] = api_key
            client = OpenAI(**kwargs)
        self._client = client

    def _tools(self, tools: list[dict] | None) -> list[dict] | None:
        # Harness tool defs are {name, description, parameters}; OpenAI wants them
        # wrapped as {"type": "function", "function": {...}}.
        if not tools:
            return None
        return [{"type": "function", "function": t} for t in tools]

    def start(self, question: str) -> list[Any]:
        return [{"role": "user", "content": question}]

    def step(self, model, system, messages, tools, max_tokens) -> Turn:
        # System prompt is prepended each call; cheap and keeps drivers stateless.
        full = [{"role": "system", "content": system}, *messages]
        resp = self._client.chat.completions.create(
            model=model,
            messages=full,
            tools=self._tools(tools),
            max_tokens=max_tokens,
        )
        choice = resp.choices[0]
        msg = choice.message
        calls: list[ToolInvocation] = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolInvocation(id=tc.id, name=tc.function.name, args=args))
        stop = "tool_use" if calls else (choice.finish_reason or "stop")
        # Append the assistant message verbatim (as a dict) for the next request.
        messages.append(msg.model_dump(exclude_none=True))
        return Turn(text=msg.content or "", tool_calls=calls, stop_reason=stop, raw=msg)

    def add_tool_results(self, messages, results) -> None:
        for inv, result in results:
            messages.append(
                {"role": "tool", "tool_call_id": inv.id, "content": result}
            )

    def add_user(self, messages, text) -> None:
        messages.append({"role": "user", "content": text})


# ---------------------------------------------------------------------------
# Anthropic / Bedrock driver (native path, preserves existing setup)
# ---------------------------------------------------------------------------

class AnthropicDriver:
    """Driver for the native Anthropic / AnthropicBedrock Messages API.

    Args:
        client: An ``anthropic.Anthropic`` or ``AnthropicBedrock`` client. If None,
            an ``AnthropicBedrock`` is created (uses ambient AWS creds).
        aws_region: Region for the default Bedrock client.
    """

    def __init__(self, client: Any = None, aws_region: str = "us-west-2"):
        if client is None:
            from anthropic import AnthropicBedrock

            client = AnthropicBedrock(aws_region=aws_region)
        self._client = client

    def _tools(self, tools: list[dict] | None) -> list[dict]:
        # Harness tool defs use OpenAI-style "parameters"; Anthropic wants
        # "input_schema". Same content, different key.
        out = []
        for t in tools or []:
            out.append(
                {
                    "name": t["name"],
                    "description": t["description"],
                    "input_schema": t.get("parameters", t.get("input_schema", {})),
                }
            )
        return out

    def start(self, question: str) -> list[Any]:
        return [{"role": "user", "content": question}]

    def step(self, model, system, messages, tools, max_tokens) -> Turn:
        kwargs: dict[str, Any] = dict(
            model=model, max_tokens=max_tokens, system=system, messages=messages
        )
        if tools:
            kwargs["tools"] = self._tools(tools)
        resp = self._client.messages.create(**kwargs)
        text = "\n".join(
            b.text for b in resp.content if getattr(b, "type", None) == "text"
        )
        calls = [
            ToolInvocation(id=b.id, name=b.name, args=dict(b.input))
            for b in resp.content
            if getattr(b, "type", None) == "tool_use"
        ]
        stop = "tool_use" if resp.stop_reason == "tool_use" else resp.stop_reason
        messages.append({"role": "assistant", "content": resp.content})
        return Turn(text=text, tool_calls=calls, stop_reason=stop, raw=resp)

    def add_tool_results(self, messages, results) -> None:
        blocks = [
            {"type": "tool_result", "tool_use_id": inv.id, "content": result}
            for inv, result in results
        ]
        messages.append({"role": "user", "content": blocks})

    def add_user(self, messages, text) -> None:
        messages.append({"role": "user", "content": text})
