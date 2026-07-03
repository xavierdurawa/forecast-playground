"""Turn plain Python tool functions into OpenAI-style tool schemas.

The ``{name, description, parameters}`` JSON-Schema form is the cross-provider
lingua franca (OpenAI, Anthropic, vLLM all consume it). Writing tools as plain
typed functions means the schema — and downstream verifiers/MCP surfaces — come
nearly for free.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, get_type_hints

# Maps Python types to JSON-Schema primitive types.
_JSON_TYPES: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _json_type(py_type: Any) -> str:
    return _JSON_TYPES.get(py_type, "string")


def function_to_tool_def(func: Callable[..., Any]) -> dict[str, Any]:
    """Build an OpenAI-style tool definition from a function's signature + docstring.

    The first line of the docstring becomes the tool description. Parameters named
    ``clock`` (or annotated as a Clock) are treated as harness-injected and are
    *excluded* from the model-facing schema — the model never sets its own as-of date.
    """
    hints = get_type_hints(func)
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""
    description = doc.split("\n", 1)[0].strip()

    properties: dict[str, Any] = {}
    required: list[str] = []
    for pname, param in sig.parameters.items():
        if pname in ("self", "clock", "kwargs") or param.kind in (
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        ):
            continue
        properties[pname] = {"type": _json_type(hints.get(pname, str))}
        if param.default is inspect.Parameter.empty:
            required.append(pname)

    return {
        "name": func.__name__,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


def tools_to_openai_schema(funcs: list[Callable[..., Any]]) -> list[dict[str, Any]]:
    """Map a list of tool functions to a list of OpenAI-style tool definitions."""
    return [function_to_tool_def(f) for f in funcs]
