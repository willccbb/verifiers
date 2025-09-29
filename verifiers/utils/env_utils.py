import importlib
import inspect
import logging
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal

from openai import BadRequestError
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion import Choice as ChatCompletionChoice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion import Completion
from openai.types.completion_choice import CompletionChoice
from openai.types.completion_usage import CompletionUsage

from verifiers.types import MessageType, ModelResponse

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from verifiers.envs.environment import Environment


@dataclass(slots=True)
class ContextLengthErrorData:
    """Structured context length metadata extracted from provider errors."""

    source: Literal["openai", "vllm"]
    message: str
    details: dict[str, int]


# OpenAI responses surface context length errors with the following wording.
# Captured format from live errors recorded in litellm's proxy tests:
# https://github.com/BerriAI/litellm/blob/09a69ad4/tests/proxy_unit_tests/test_proxy_exception_mapping.py#L30-L39
_OPENAI_CONTEXT_LENGTH_RE = re.compile(
    r"This model's maximum context length is (?P<limit>\d+) tokens\. "
    r"However, you requested (?P<requested>\d+) tokens "
    r"\((?P<prompt_tokens>\d+) in the messages, (?P<completion_tokens>\d+) "
    r"in the completion\)\. Please reduce the length of the messages or completion\."
)

# vLLM raises ValueErrors with these exact messages when validating prompts. See:
# https://github.com/vllm-project/vllm/blob/9cfa5486/vllm/entrypoints/openai/serving_engine.py#L642-L673
_VLLM_PROMPT_ONLY_RE = re.compile(
    r"This model's maximum context length is (?P<limit>\d+) tokens\. "
    r"However, your request has (?P<prompt_tokens>\d+) input tokens\. "
    r"Please reduce the length of the input messages\."
)
_VLLM_OPERATION_INPUT_RE = re.compile(
    r"This model's maximum context length is (?P<limit>\d+) tokens\. "
    r"However, you requested (?P<prompt_tokens>\d+) tokens in the input for [^.]+\. "
    r"Please reduce the length of the input\."
)
_VLLM_MAX_TOKENS_RE = re.compile(
    r"'max_tokens' or 'max_completion_tokens' is too large: (?P<completion_tokens>\d+)\. "
    r"This model's maximum context length is (?P<limit>\d+) tokens and your request has "
    r"(?P<prompt_tokens>\d+) input tokens \((?P=completion_tokens) > (?P=limit) - (?P=prompt_tokens)\)\."
)


def load_environment(env_id: str, **env_args) -> "Environment":
    logger = logging.getLogger("verifiers.utils.env_utils")
    logger.info(f"Loading environment: {env_id}")

    module_name = env_id.replace("-", "_")
    try:
        module = importlib.import_module(module_name)

        if not hasattr(module, "load_environment"):
            raise AttributeError(
                f"Module '{module_name}' does not have a 'load_environment' function. "
                f"This usually means there's a package name collision. Please either:\n"
                f"1. Rename your environment (e.g. suffix with '-env')\n"
                f"2. Remove unneeded files with the same name\n"
                f"3. Check that you've installed the correct environment package"
            )

        env_load_func: Callable[..., "Environment"] = getattr(
            module, "load_environment"
        )
        sig = inspect.signature(env_load_func)
        defaults_info = []
        for param_name, param in sig.parameters.items():
            if param.default != inspect.Parameter.empty:
                if isinstance(param.default, (dict, list)):
                    defaults_info.append(f"{param_name}={param.default}")
                elif isinstance(param.default, str):
                    defaults_info.append(f"{param_name}='{param.default}'")
                else:
                    defaults_info.append(f"{param_name}={param.default}")
            else:
                defaults_info.append(f"{param_name}=<required>")

        if defaults_info:
            logger.debug(f"Environment defaults: {', '.join(defaults_info)}")

        if env_args:
            provided_params = set(env_args.keys())
        else:
            provided_params = set()

        all_params = set(sig.parameters.keys())
        default_params = all_params - provided_params

        if provided_params:
            provided_values = []
            for param_name in provided_params:
                provided_values.append(f"{param_name}={env_args[param_name]}")
            logger.info(f"Using provided args: {', '.join(provided_values)}")

        if default_params:
            default_values = []
            for param_name in default_params:
                param = sig.parameters[param_name]
                if param.default != inspect.Parameter.empty:
                    if isinstance(param.default, str):
                        default_values.append(f"{param_name}='{param.default}'")
                    else:
                        default_values.append(f"{param_name}={param.default}")
            if default_values:
                logger.info(f"Using default args: {', '.join(default_values)}")

        env_instance: "Environment" = env_load_func(**env_args)

        logger.info(f"Successfully loaded environment '{env_id}'")

        return env_instance

    except ImportError as e:
        logger.error(
            f"Failed to import environment module {module_name} for env_id {env_id}: {str(e)}"
        )
        raise ValueError(
            f"Could not import '{env_id}' environment. Ensure the package for the '{env_id}' environment is installed."
        ) from e
    except Exception as e:
        logger.error(
            f"Failed to load environment {env_id} with args {env_args}: {str(e)}"
        )
        raise RuntimeError(f"Failed to load environment '{env_id}': {str(e)}") from e


def infer_provider_name(client: "AsyncOpenAI") -> str:
    base_url = getattr(client, "base_url", "")
    base_url_str = str(base_url)
    if "api.openai.com" in base_url_str:
        return "OpenAI"
    if any(
        host in base_url_str for host in ("localhost", "127.0.0.1", "0.0.0.0", "vllm")
    ):
        return "vLLM"
    return "OpenAI-compatible endpoint"


def _match_openai_context_length(message: str) -> dict[str, int] | None:
    match = _OPENAI_CONTEXT_LENGTH_RE.fullmatch(message)
    if not match:
        return None
    details = {key: int(value) for key, value in match.groupdict().items()}
    limit = details["limit"]
    requested = details["requested"]
    over = requested - limit
    if over >= 0:
        details["over"] = over
    return details


def _match_vllm_context_length(message: str) -> dict[str, int] | None:
    for pattern in (_VLLM_PROMPT_ONLY_RE, _VLLM_OPERATION_INPUT_RE):
        match = pattern.fullmatch(message)
        if not match:
            continue
        details = {key: int(value) for key, value in match.groupdict().items()}
        prompt_tokens = details["prompt_tokens"]
        limit = details["limit"]
        details.setdefault("completion_tokens", 0)
        details["over"] = max(prompt_tokens - limit, 0)
        return details

    match = _VLLM_MAX_TOKENS_RE.fullmatch(message)
    if not match:
        return None
    details = {key: int(value) for key, value in match.groupdict().items()}
    limit = details["limit"]
    prompt_tokens = details["prompt_tokens"]
    completion_tokens = details["completion_tokens"]
    details["requested"] = prompt_tokens + completion_tokens
    details["over"] = max(details["requested"] - limit, 0)
    return details


def _iter_message_candidates(error: Exception) -> list[str]:
    candidates: list[str] = []
    if isinstance(error, BadRequestError):
        body = getattr(error, "body", None)
        if isinstance(body, dict):
            error_block = body.get("error")
            if isinstance(error_block, dict):
                raw_message = error_block.get("message")
                if isinstance(raw_message, str):
                    candidates.append(raw_message)
        candidates.append(str(error))
    else:
        candidates.append(str(error))
    return [candidate for candidate in candidates if candidate]


def extract_context_length_error(error: Exception) -> ContextLengthErrorData | None:
    for message in _iter_message_candidates(error):
        openai_details = _match_openai_context_length(message)
        if openai_details is not None:
            return ContextLengthErrorData("openai", message, openai_details)
        vllm_details = _match_vllm_context_length(message)
        if vllm_details is not None:
            return ContextLengthErrorData("vllm", message, vllm_details)
    return None


def format_context_length_warning(
    provider: str, model: str, details: dict[str, int]
) -> str:
    fragments: list[str] = []
    requested = details.get("requested")
    prompt_tokens = details.get("prompt_tokens")
    completion_tokens = details.get("completion_tokens")
    limit = details.get("limit")
    over = details.get("over")

    if requested is not None:
        fragments.append(f"requested {requested} tokens")
    elif prompt_tokens is not None:
        fragments.append(f"prompt {prompt_tokens} tokens")
    if limit is not None:
        fragments.append(f"limit {limit}")
    if over is not None and over > 0:
        fragments.append(f"over by {over} tokens")
    if completion_tokens:
        fragments.append(f"{completion_tokens} reserved for completion")

    if not fragments:
        fragments.append("input prompt was too long")

    fragments.append("returning synthetic length-finish response")
    return f"Context length exceeded for model '{model}' via {provider} - " + "; ".join(
        fragments
    )


def build_context_length_stub_response(
    message_type: MessageType, model: str, details: dict[str, int]
) -> ModelResponse:
    completion_tokens = int(details.get("completion_tokens", 0))
    prompt_tokens = details.get("prompt_tokens")
    if prompt_tokens is None:
        requested = details.get("requested")
        if requested is not None:
            prompt_tokens = requested - completion_tokens
        else:
            prompt_tokens = details.get("limit", 0)
    prompt_tokens = max(int(prompt_tokens), 0)
    total_tokens = prompt_tokens + completion_tokens
    usage = CompletionUsage(
        completion_tokens=completion_tokens,
        prompt_tokens=prompt_tokens,
        total_tokens=total_tokens,
    )
    created_ts = int(time.time())
    if message_type == "chat":
        message = ChatCompletionMessage(role="assistant", content="")
        choice = ChatCompletionChoice(
            finish_reason="length",
            index=0,
            message=message,
        )
        return ChatCompletion(
            id="context_length_guardrail",
            choices=[choice],
            created=created_ts,
            model=model,
            object="chat.completion",
            usage=usage,
        )
    choice = CompletionChoice(finish_reason="length", index=0, text="")
    return Completion(
        id="context_length_guardrail",
        choices=[choice],
        created=created_ts,
        model=model,
        object="text_completion",
        usage=usage,
    )
