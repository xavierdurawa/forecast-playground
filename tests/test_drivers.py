"""Offline tests for the model drivers' format conversion (no network)."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from forecast_playground.drivers import AnthropicDriver, OpenAIDriver

_TOOLS = [{"name": "wiki_search", "description": "d", "parameters": {"type": "object", "properties": {}}}]


def test_openai_driver_wraps_tools_and_parses_calls():
    # Fake OpenAI client returning one tool call.
    tool_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="wiki_search", arguments=json.dumps({"query": "x"})),
    )
    msg = MagicMock()
    msg.content = "thinking"
    msg.tool_calls = [tool_call]
    msg.model_dump.return_value = {"role": "assistant", "content": "thinking"}
    resp = SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="tool_calls")])

    client = MagicMock()
    client.chat.completions.create.return_value = resp
    drv = OpenAIDriver(client=client)

    messages = drv.start("Q?")
    turn = drv.step("gpt-x", "sys", messages, _TOOLS, 1000)

    # Tools were wrapped in {"type":"function","function":{...}}.
    sent_tools = client.chat.completions.create.call_args.kwargs["tools"]
    assert sent_tools[0]["type"] == "function"
    assert sent_tools[0]["function"]["name"] == "wiki_search"
    # System prepended.
    assert client.chat.completions.create.call_args.kwargs["messages"][0]["role"] == "system"
    # Tool call parsed.
    assert turn.stop_reason == "tool_use"
    assert turn.tool_calls[0].name == "wiki_search"
    assert turn.tool_calls[0].args == {"query": "x"}


def test_openai_driver_final_answer():
    msg = MagicMock()
    msg.content = "PROBABILITY: 0.3"
    msg.tool_calls = None
    msg.model_dump.return_value = {"role": "assistant", "content": "PROBABILITY: 0.3"}
    resp = SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")])
    client = MagicMock()
    client.chat.completions.create.return_value = resp

    drv = OpenAIDriver(client=client)
    turn = drv.step("gpt-x", "sys", drv.start("Q?"), None, 1000)
    assert turn.tool_calls == []
    assert "0.3" in turn.text


def test_anthropic_driver_renames_parameters_to_input_schema():
    block = SimpleNamespace(type="text", text="hi")
    resp = SimpleNamespace(content=[block], stop_reason="end_turn")
    client = MagicMock()
    client.messages.create.return_value = resp

    drv = AnthropicDriver(client=client)
    drv.step("claude-x", "sys", drv.start("Q?"), _TOOLS, 1000)

    sent_tools = client.messages.create.call_args.kwargs["tools"]
    assert "input_schema" in sent_tools[0] and "parameters" not in sent_tools[0]
    assert sent_tools[0]["name"] == "wiki_search"
