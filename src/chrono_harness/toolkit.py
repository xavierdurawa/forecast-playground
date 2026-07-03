"""Toolkit: bind sources + a Clock into model-callable tools and dispatch calls.

This is the layer that turns the harness into something a model can actually use:
it exposes each source as a search-style tool with a clean JSON schema, injects the
Clock (so the model never sets its own as-of date), executes tool calls by name, and
records every call for friction analysis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from .cache import ResultCache
from .clock import Clock, LookaheadError
from .sources.base import Source
from .tools.sandbox import run_python

# Anthropic/OpenAI share this {name, description, input_schema/parameters} shape.
ToolDef = dict[str, Any]


@dataclass
class ToolCall:
    """A record of one tool invocation (for tracing harness friction)."""

    name: str
    args: dict[str, Any]
    ok: bool
    result_chars: int
    error: str | None = None
    cached: bool = False


@dataclass
class Toolkit:
    """Binds sources + extras to one Clock and dispatches model tool calls.

    Args:
        clock: The as-of Clock injected into every retrieval (model can't set it).
        sources: Source adapters, each exposed as ``<source>_search``.
        enable_python: Expose the ``run_python`` compute tool.
        max_result_chars: Hard cap on any single tool result, as a backstop against
            context overflow when a source returns several/large documents.
        cache: Optional ResultCache. Retrieval is deterministic given (tool, args,
            as_of), so results are cached to avoid re-fetching within/across runs.
            Note: ``run_python`` is never cached (it may be nondeterministic).
    """

    clock: Clock
    sources: list[Source]
    enable_python: bool = True
    max_result_chars: int = 20000
    cache: ResultCache | None = None
    calls: list[ToolCall] = field(default_factory=list)

    def _source_tool_name(self, src: Source) -> str:
        # "wikipedia:en" -> "wikipedia_search"; keep it a valid identifier.
        base = src.name.split(":")[0]
        return f"{base}_search"

    def _tool_specs(self) -> list[tuple[str, str, dict[str, Any]]]:
        """Canonical (name, description, json-schema-properties) per tool.

        One source of truth; the Anthropic/OpenAI emitters format it differently.
        """
        specs: list[tuple[str, str, dict[str, Any]]] = []
        for src in self.sources:
            specs.append(
                (
                    self._source_tool_name(src),
                    f"Search the {src.name} source for information available as of "
                    f"the (hidden) forecast date. Guarantee: {src.guarantee.value}. "
                    f"Returns documents that existed at or before that date.",
                    {
                        "query": {
                            "type": "string",
                            "description": "Search query / article title / URL / token id.",
                        }
                    },
                )
            )
        if self.enable_python:
            specs.append(
                (
                    "run_python",
                    run_python.__doc__.split("\n", 1)[0],
                    {"code": {"type": "string", "description": "Python source to run."}},
                )
            )
        return specs

    def anthropic_tools(self) -> list[ToolDef]:
        """Tool definitions in Anthropic's tool-use format (name/description/input_schema)."""
        return [
            {
                "name": name,
                "description": desc,
                "input_schema": {
                    "type": "object",
                    "properties": props,
                    "required": list(props),
                },
            }
            for name, desc, props in self._tool_specs()
        ]

    def anthropic_tools_as_openai(self) -> list[ToolDef]:
        """Same tools in OpenAI/verifiers format (name/description/parameters)."""
        return [
            {
                "name": name,
                "description": desc,
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": list(props),
                },
            }
            for name, desc, props in self._tool_specs()
        ]

    def _dispatch_map(self) -> dict[str, Callable[[dict[str, Any]], str]]:
        m: dict[str, Callable[[dict[str, Any]], str]] = {}
        for src in self.sources:
            m[self._source_tool_name(src)] = lambda a, s=src: self._run_source(s, a)
        if self.enable_python:
            m["run_python"] = lambda a: run_python(a.get("code", ""))
        return m

    def _run_source(self, src: Source, args: dict[str, Any]) -> str:
        docs = src.fetch(args.get("query", ""), self.clock)
        if not docs:
            return f"(no results from {src.name} as of {self.clock.as_of.date()})"
        # Concatenate documents with their timestamps so the model sees recency.
        parts = []
        for d in docs:
            head = f"[{d.source} | {d.timestamp.date().isoformat()}]"
            parts.append(f"{head}\n{d.content}")
        out = "\n\n".join(parts)
        # Backstop against context overflow from multi/large documents.
        if len(out) > self.max_result_chars:
            out = out[: self.max_result_chars] + "\n…[result truncated]"
        return out

    def call(self, name: str, args: dict[str, Any]) -> str:
        """Execute a tool by name, recording the call. Returns a string result.

        A LookaheadError is converted to an error result rather than raised, so a
        misbehaving source surfaces as a visible failure instead of crashing a run.
        """
        dispatch = self._dispatch_map()
        if name not in dispatch:
            rec = ToolCall(name, args, ok=False, result_chars=0, error="unknown tool")
            self.calls.append(rec)
            return f"ERROR: unknown tool {name!r}"

        # run_python may be nondeterministic (time, randomness) so it is never cached.
        cacheable = self.cache is not None and name != "run_python"
        cache_key = None
        if cacheable:
            cache_key = json.dumps(
                {"tool": name, "args": args, "as_of": self.clock.as_of.isoformat()},
                sort_keys=True,
            )
            hit = self.cache.get(cache_key)
            if hit is not None:
                self.calls.append(
                    ToolCall(name, args, ok=True, result_chars=len(hit), cached=True)
                )
                return hit
        try:
            result = dispatch[name](args)
            self.calls.append(ToolCall(name, args, ok=True, result_chars=len(result)))
            if cacheable:
                self.cache.put(cache_key, result)
            return result
        except LookaheadError as e:
            self.calls.append(ToolCall(name, args, ok=False, result_chars=0, error=str(e)))
            return f"ERROR (lookahead blocked): {e}"
        except Exception as e:  # network, parse, etc. — keep the run alive
            self.calls.append(
                ToolCall(name, args, ok=False, result_chars=0, error=repr(e))
            )
            return f"ERROR: {e!r}"
