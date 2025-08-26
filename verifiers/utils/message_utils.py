from collections.abc import Iterable
from typing import cast

from verifiers.types import ChatMessage, Messages


def sanitize_object(obj: object):
    """
    Recursively convert Pydantic/OpenAI SDK objects to plain Python types
    (dict/list/str/bool/number). Leaves primitives unchanged.
    """
    if isinstance(obj, (str, bytes, bytearray, int, float, bool)) or obj is None:
        return obj
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        obj = dump()
    if isinstance(obj, dict):
        return {k: sanitize_object(v) for k, v in obj.items()}
    # check if obj is iterable
    if isinstance(obj, Iterable):
        return [sanitize_object(x) for x in obj]
    return obj


def sanitize_chat_message(message: ChatMessage):
    """
    input: chat message (dict or object)
    output: chat message (dict)
    """
    # TODO: debug for multimodal messages; content can get consumed as an iterator
    new_message = {}
    dump = getattr(message, "model_dump", None)
    if callable(dump):
        new_message = dump()
        return new_message
    assert isinstance(message, dict)
    assert isinstance(new_message, dict)
    new_message["role"] = message["role"]
    if "content" in message and message["content"]:
        content = message["content"]
        if isinstance(content, str):
            new_message["content"] = content
        else:
            new_message["content"] = []
            parts = list(content) if not isinstance(content, str) else content
            for c in parts:
                if isinstance(c, str):
                    new_message["content"].append(c)
                else:
                    new_message["content"].append(sanitize_object(c))
    if "tool_calls" in message and message["tool_calls"]:
        tool_calls = list(message["tool_calls"])
        new_message["tool_calls"] = [
            sanitize_object(tool_call) for tool_call in tool_calls
        ]
    return new_message


def sanitize_messages(messages: Messages) -> str | list:
    """
    input: list of dicts or Pydantic models, or str
    output: list of dicts, or str
    """
    if isinstance(messages, str):
        return messages
    sanitized_list = [sanitize_chat_message(m) for m in list(messages)]
    return sanitized_list


def content_to_printable(content: object) -> str:
    """
    Render content to readable text, handling multimodal lists.
    - Text parts: return their text
    - Image-like parts: return "[image]"
    Falls back to str(content).
    """
    print(str(content)[:100])
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if "type" in content and content["type"] == "text":
            return content["text"]
        if "type" in content and content["type"] in {
            "image_url",
            "input_image",
            "image",
        }:
            return "[image]"
    if isinstance(content, (list, tuple)):
        out = []
        for x in content:
            out.append(content_to_printable(x))
        return "\n\n".join(out)
    return str(content)


def message_to_printable(message: ChatMessage) -> ChatMessage:
    new_message = {}
    new_message["role"] = message["role"]
    new_message["content"] = []
    if "tool_calls" in message and message["tool_calls"]:
        new_message["tool_calls"] = message["tool_calls"]
    content = message.get("content")
    if content is None:
        return cast(ChatMessage, new_message)
    if isinstance(content, str):
        new_message["content"].append(content)
    else:
        for c in content:
            if isinstance(c, str):
                new_message["content"].append(c)
            else:
                c_dict = dict(c)
                if c_dict["type"] == "text":
                    new_message["content"].append(c_dict["text"])
                elif c_dict["type"] == "image_url":
                    new_message["content"].append("[image]")
    print(new_message["content"])
    new_message["content"] = "\n\n".join(new_message["content"])
    return cast(ChatMessage, new_message)


def messages_to_printable(messages: Messages) -> Messages:
    if isinstance(messages, str):
        return messages
    return [message_to_printable(m) for m in messages]


def cleanup_message(message: ChatMessage) -> ChatMessage:
    new_message = {}
    new_message["role"] = message["role"]
    new_message["content"] = []
    content = message.get("content")
    if content is None:
        return cast(ChatMessage, new_message)
    if isinstance(content, str):
        new_message["content"] = content
    else:
        for c in content:
            new_c = c.copy()
            c_dict = dict(c)
            if "image_url" in c_dict and "type" in c_dict and c_dict["type"] == "text":
                new_c.pop("image_url")
                new_message["content"].append(new_c)
            elif (
                "image_url" in c_dict
                and "type" in c_dict
                and c_dict["type"] == "image_url"
            ):
                new_c.pop("text")
                new_message["content"].append(new_c)
            else:
                new_message["content"].append(new_c)
    return cast(ChatMessage, new_message)


def cleanup_messages(messages: Messages) -> Messages:
    if isinstance(messages, str):
        return messages
    new_messages = []
    for m in messages:
        new_messages.append(cleanup_message(m))
    return new_messages
