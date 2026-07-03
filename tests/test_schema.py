"""Tests for OpenAI-style tool schema generation."""

from forecast_playground import tools_to_openai_schema
from forecast_playground.clock import Clock


def sample_tool(title: str, limit: int = 5, clock: Clock = None) -> str:
    """Look up a thing by title.

    More detail here that should not appear in the description.
    """
    return ""


def test_schema_shape_and_clock_excluded():
    (defn,) = tools_to_openai_schema([sample_tool])
    assert defn["name"] == "sample_tool"
    assert defn["description"] == "Look up a thing by title."
    props = defn["parameters"]["properties"]
    # clock is harness-injected and must not be model-facing.
    assert "clock" not in props
    assert props["title"]["type"] == "string"
    assert props["limit"]["type"] == "integer"
    # Only the argument without a default is required.
    assert defn["parameters"]["required"] == ["title"]
