from __future__ import annotations

import inspect
import re
from typing import Any, Literal, Union, get_args, get_origin

from verifiers.types import (
    ChatCompletionToolParam,
    FunctionParameters,
    JsonPrimitive,
)

_JSON_PRIMITIVE_MAP: dict[type, JsonPrimitive] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _get_json_type(annotation: Any) -> tuple[JsonPrimitive, list[Any] | None]:
    """Return the JSON Schema type name and optional enum values for *annotation*.

    The second element is a list of literal values if *annotation* is a typing.Literal.
    """
    origin = get_origin(annotation)

    if origin is Literal:
        # Treat Literal values as strings/numbers depending on their Python type.
        literal_values = list(get_args(annotation))
        if not literal_values:
            return "string", None
        first_value = literal_values[0]
        json_type = _JSON_PRIMITIVE_MAP.get(type(first_value), "string")
        return json_type, literal_values

    if origin is Union:
        # If Optional[T] or Union[T, None], return type of T.
        args = [a for a in get_args(annotation) if a is not type(None)]  # noqa: E721, we really need NoneType
        if len(args) == 1:
            return _get_json_type(args[0])

    # Normal (non-parameterised) annotation
    json_type = _JSON_PRIMITIVE_MAP.get(annotation, "string")
    return json_type, None


_PARAM_RE = re.compile(r"^\s*(\w+)\s*\(([^)]*)\):\s*(.*)$")


def _parse_docstring(func: Any) -> tuple[str, dict[str, str]]:
    """Extract the short description and parameter descriptions from *func*'s docstring.

    Returns
    -------
    (summary, param_descriptions)
    """
    doc = inspect.getdoc(func) or ""
    if not doc:
        return "", {}

    lines = doc.splitlines()
    # First non-empty line is the summary.
    summary = next((line.strip() for line in lines if line.strip()), "")

    param_descs: dict[str, str] = {}
    # Try to locate an "Args:" or "Parameters:" block.
    try:
        block_idx = next(
            i
            for i, line in enumerate(lines)
            if line.strip().lower() in {"args:", "arguments:", "parameters:"}
        )
    except StopIteration:
        return summary, param_descs

    for raw in lines[block_idx + 1 :]:
        if not raw.strip():
            # Stop once we hit a blank line, assuming end of args block.
            break
        match = _PARAM_RE.match(raw)
        if match:
            name, _type, desc = match.groups()
            param_descs[name] = desc.strip()
        else:
            # Continuation lines – append to last parameter if any.
            if param_descs and raw.startswith(" " * 4):
                last_key = next(reversed(param_descs))
                param_descs[last_key] += " " + raw.strip()
            else:
                break  # End of recognised param section
    return summary, param_descs


def _is_required(annotation: Any) -> bool:
    """True if *annotation* is not Optional/Union[..., None]."""
    origin = get_origin(annotation)
    if origin is Union:
        return type(None) not in get_args(annotation)
    return True


def convert_func_to_oai_tool(func: Any) -> ChatCompletionToolParam:
    """Convert *func* to an OpenAI function-calling tool schema.

    The returned mapping matches the structure expected in the `tools` list
    of the OpenAI ChatCompletion API.
    """
    if not callable(func):
        raise TypeError("Expected a callable object")

    signature = inspect.signature(func)
    summary, param_descs = _parse_docstring(func)

    if not summary:
        summary = f"Auto-generated description for `{func.__name__}`."  # basic fallback

    # Resolve postponed annotations so we properly interpret Literal and others
    try:
        resolved_hints = inspect.get_annotations(func, eval_str=True)
    except AttributeError:  # Fallback for older Python versions
        from typing import get_type_hints

        resolved_hints = get_type_hints(func)

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in signature.parameters.items():
        if name == "self":
            continue  # Ignore instance methods' self parameter

        annotation = resolved_hints.get(
            name,
            param.annotation
            if param.annotation is not inspect.Parameter.empty
            else str,
        )
        json_type, enum_vals = _get_json_type(annotation)

        prop_schema: dict[str, Any] = {
            "type": json_type,
        }
        if enum_vals is not None:
            prop_schema["enum"] = enum_vals

        # Description: prefer docstring info, else fallback to generic text.
        if name in param_descs:
            prop_schema["description"] = param_descs[name]
        else:
            prop_schema.setdefault(
                "description",
                f"Parameter `{name}` of type {json_type}.",
            )

        properties[name] = prop_schema

        # Consider parameter required if no default or default is inspect._empty
        if param.default is inspect.Parameter.empty and _is_required(annotation):
            required.append(name)

    parameters_schema: FunctionParameters = {
        "type": "object",
        "properties": properties,
    }
    if required:
        parameters_schema["required"] = required

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": summary,
            "parameters": parameters_schema,
        },
    }
