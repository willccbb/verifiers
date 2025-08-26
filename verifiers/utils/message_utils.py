from collections.abc import Iterable

from verifiers.types import ChatMessage, Message, Messages


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


def message_to_printable(message: Message):
    """
    Render message content to readable text, handling multimodal lists.
    - Text parts: return their text
    - Image-like parts: return "[image]"
    Falls back to str(content).
    """
    if isinstance(message, str):
        return message
    out_msg = {}
    out_msg["role"] = message["role"]
    if "tool_calls" in message:
        out_msg["tool_calls"] = []
        for tool_call in message["tool_calls"]:
            out_msg["tool_calls"].append(dict(tool_call))
    if "content" in message:
        out_msg["content"] = ""
        if isinstance(message["content"], str):
            out_msg["content"] = message["content"]
        elif message["content"] is None:
            out_msg["content"] = ""
        else:
            old_parts = []
            parts = []
            for content in message["content"]:
                old_parts.append(content)
                if content["type"] == "text":
                    parts.append(content["text"])
                elif content["type"] == "image_url":
                    parts.append("[image]")
            out_msg["content"] = "\n\n".join(parts)
    return out_msg
