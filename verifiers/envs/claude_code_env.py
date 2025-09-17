"""
Claude Code environment for single-shot tasks.
Multi shot tasks will need to have a different implementation and possible inherit or share functionality with `multiturn_env.py`
"""

import json
from typing import Any, Callable
from claude_code_sdk import query, ClaudeCodeOptions
from claude_code_sdk.types import (
    ResultMessage,
    SystemMessage,
    AssistantMessage,
    UserMessage,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
)
from claude_code_sdk.types import Message as AnthropicMessage
from verifiers.envs.environment import Environment
from verifiers.utils.async_utils import maybe_await
from verifiers.types import Info, Messages, SamplingArgs, State


class ClaudeCodeEnv(Environment):
    def __init__(
        self,
        max_turns=-1,
        tools: list[Callable] | None = None,
        system_prompt: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tools = tools
        self.max_turns = max_turns
        self.system_prompt = system_prompt

    async def setup_state(self, state: State, **kwargs) -> State:
        #   TODO: maybe implement state setup? maybe the claude options?
        return state

    async def anthropic_to_oai_completion(
        self, anthropic_msgs: list[AnthropicMessage]
    ) -> Messages:
        oai_messages = []

        for msg in anthropic_msgs:
            if isinstance(msg, UserMessage):
                # Convert user message content
                if isinstance(msg.content, str):
                    oai_messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg.content, list):
                    # Handle content blocks in user messages
                    text_parts = []
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            text_parts.append(block.text)
                        elif isinstance(block, ToolResultBlock):
                            # OpenAI expects tool results in separate messages
                            oai_messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": block.tool_use_id,
                                    "content": json.dumps(block.content)
                                    if block.content
                                    else "",
                                }
                            )

                    # Add user text if any
                    if text_parts:
                        oai_messages.append(
                            {"role": "user", "content": "\n".join(text_parts)}
                        )

            elif isinstance(msg, AssistantMessage):
                # Convert assistant message content
                text_parts = []
                tool_calls = []

                if hasattr(msg, "content") and msg.content:
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            text_parts.append(block.text)
                        elif isinstance(block, ToolUseBlock):
                            tool_calls.append(
                                {
                                    "id": block.id,
                                    "type": "function",
                                    "function": {
                                        "name": block.name,
                                        "arguments": json.dumps(block.input),
                                    },
                                }
                            )
                        elif isinstance(block, ToolResultBlock):
                            # Tool results in assistant messages are unusual but handle them
                            if block.content:
                                text_parts.append(f"Tool result: {block.content}")

                # Build message with proper typing
                oai_message: dict[str, Any] = {
                    "role": "assistant",
                    "content": "\n".join(text_parts) if text_parts else "",
                }

                if tool_calls:
                    oai_message["tool_calls"] = tool_calls

                oai_messages.append(oai_message)

            elif isinstance(msg, SystemMessage):
                # Skip init and metadata messages, they don't map to OpenAI format
                continue

            elif isinstance(msg, ResultMessage):
                # Skip result messages as they contain metadata, not conversation content
                if msg.result:
                    oai_message = {"role": "assistant", "content": msg.result}
                    oai_messages.append(oai_message)

        return oai_messages

    async def rollout(
        self,
        client,
        model: str,
        prompt: Messages,
        answer: str = "",
        task: str = "default",
        info: Info | None = None,
        sampling_args: SamplingArgs | None = None,
        **kwargs,
    ) -> tuple[Messages, State]:
        """
        Run rollout on Claude Code for a given prompt.
        Returns a tuple of (completion, state).
        """
        info = info or {}
        if isinstance(prompt, str):
            rollout = prompt
        else:
            rollout = str(prompt[-1]["content"]) if "content" in prompt[-1] else ""
        state = {
            "prompt": prompt,
            "completion": [],  # maybe we dont need this
            "task": task,
            "info": info,
            "tools": [],
            "responses": [],
            "turn": 0,
            "mcp_servers": [],
            "claude_code_options": {"permission_mode": "bypassPermissions"},
        }
        state = await maybe_await(self.setup_state, state, **kwargs)
        options = ClaudeCodeOptions(
            **state["claude_code_options"], system_prompt=self.system_prompt
        )
        messages = list()
        async for message in query(prompt=rollout, options=options):
            if isinstance(message, SystemMessage) and message.subtype == "init":
                state["tools"] = message.data["tools"]
                state["mcp_servers"] = message.data["mcp_servers"]
            if isinstance(message, ResultMessage):
                state["result_msg"] = message.result
                state["success"] = not message.is_error
            messages.append(message)
        oai_messages = await self.anthropic_to_oai_completion(anthropic_msgs=messages)
        return oai_messages, state
