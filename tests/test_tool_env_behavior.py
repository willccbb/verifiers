"""Tests covering ToolEnv and StatefulToolEnv helper behaviours."""

from __future__ import annotations

import json
from typing import Any

import pytest
from datasets import Dataset
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function as ChatCompletionMessageToolCallFunction,
)

from verifiers.envs.stateful_tool_env import StatefulToolEnv
from verifiers.envs.tool_env import ToolEnv
from verifiers.parsers.parser import Parser
from verifiers.rubrics.rubric import Rubric


class _BaseToolEnv(ToolEnv):
    async def env_response(self, messages, state, **kwargs):  # type: ignore[override]
        return [], state


class _DummyStatefulToolEnv(StatefulToolEnv):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.update_invocations: list[tuple[str, dict[str, Any]]] = []

    def update_tool_args(self, tool_name, tool_args, messages, state, **kwargs):  # type: ignore[override]
        # Track calls and inject stateful arguments
        captured = dict(tool_args)
        self.update_invocations.append((tool_name, captured))
        updated = dict(tool_args)
        if tool_name == "_dummy_tool":
            updated["injected"] = state["injected"]
        return updated


def _make_dataset() -> Dataset:
    return Dataset.from_dict(
        {
            "question": [""],
            "answer": [""],
        }
    )


def _make_env(env_cls, tools=None, **kwargs):
    return env_cls(
        client=None,
        model="test-model",
        dataset=_make_dataset(),
        parser=Parser(),
        rubric=Rubric(),
        tools=tools,
        **kwargs,
    )


def _dummy_tool(x: int, injected: str = "") -> str:
    return f"{x}:{injected}"


@pytest.mark.asyncio
async def test_tool_env_remove_tool_updates_internal_state():
    env = _make_env(_BaseToolEnv, tools=[], max_turns=1)

    env.add_tool(_dummy_tool)
    assert _dummy_tool in env.tools
    assert "_dummy_tool" in env.tool_map

    env.remove_tool(_dummy_tool)
    assert _dummy_tool not in env.tools
    assert "_dummy_tool" not in env.tool_map
    assert all(
        tool_spec["function"]["name"] != "_dummy_tool" for tool_spec in env.oai_tools
    )

    # Ensure max_turns enforcement short-circuits completion detection
    messages = [{"role": "assistant", "content": "done"}]
    state = {"turn": 1}
    env.max_turns = 1
    assert await env.is_completed(messages, state)


@pytest.mark.asyncio
async def test_stateful_tool_env_update_receives_tool_name_and_cleanup():
    env = _make_env(_DummyStatefulToolEnv, tools=[], max_turns=2)
    env.add_tool(_dummy_tool, args_to_skip=["injected"])

    tool_call = ChatCompletionMessageToolCall(
        id="call-1",
        function=ChatCompletionMessageToolCallFunction(
            name="_dummy_tool", arguments=json.dumps({"x": 1})
        ),
        type="function",
    )
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [tool_call],
        }
    ]
    state = {"injected": "secret"}

    tool_messages, _ = await env.env_response(messages, state)

    assert env.update_invocations == [("_dummy_tool", {"x": 1})]
    assert tool_messages == [
        {"role": "tool", "content": "1:secret", "tool_call_id": "call-1"}
    ]

    env.remove_tool(_dummy_tool)
    assert "_dummy_tool" not in env.skipped_args
    assert all(
        tool_spec["function"]["name"] != "_dummy_tool" for tool_spec in env.oai_tools
    )
