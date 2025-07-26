"""Utility functions for working with types and gradual migration to Pydantic."""

from typing import Any, Dict, List, Union
from verifiers.types import ChatMessage


def ensure_chat_message(msg: Union[Dict[str, Any], ChatMessage]) -> ChatMessage:
    """Convert dict or ChatMessage to ChatMessage instance.
    
    This utility helps with gradual migration from dict-based messages to Pydantic models.
    """
    if isinstance(msg, dict):
        return ChatMessage.from_dict(msg)
    return msg


def ensure_chat_messages(msgs: List[Union[Dict[str, Any], ChatMessage]]) -> List[ChatMessage]:
    """Convert list of dicts or ChatMessages to list of ChatMessage instances."""
    return [ensure_chat_message(msg) for msg in msgs]


def create_user_message(content: str) -> ChatMessage:
    """Convenience function to create a user message."""
    return ChatMessage(role="user", content=content)


def create_assistant_message(content: str, tool_calls=None) -> ChatMessage:
    """Convenience function to create an assistant message."""
    return ChatMessage(role="assistant", content=content, tool_calls=tool_calls)


def create_system_message(content: str) -> ChatMessage:
    """Convenience function to create a system message."""
    return ChatMessage(role="system", content=content)


def create_tool_message(content: str, tool_call_id: str) -> ChatMessage:
    """Convenience function to create a tool message."""
    return ChatMessage(role="tool", content=content, tool_call_id=tool_call_id)