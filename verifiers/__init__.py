import logging
import sys
from typing import Any, Callable, Dict, List, Literal, Optional, TypedDict, Union

from openai.types.chat.chat_completion import ChatCompletion
from openai.types.completion import Completion

# typing aliases
MessageType = Literal["chat", "completion"]
ModelResponse = Union[Completion, ChatCompletion, None]
ChatMessageField = Literal["role", "content"]
ChatMessage = Dict[ChatMessageField, str]
Message = Union[str, ChatMessage]
Messages = Union[str, List[ChatMessage]]
Info = Dict[str, Any]
State = Dict[str, Any]
SamplingArgs = Dict[str, Any]
RewardFunc = Callable[..., float]

from typing import Dict, List


class GenerateInputs(TypedDict):
    prompt: List[Messages]
    answer: Optional[List[str]]
    info: Optional[List[Dict]]
    task: Optional[List[str]]
    completion: Optional[List[Messages]]


GenerateOutputs = Dict[str, Any]


class ProcessedOutputs(TypedDict):
    prompt_ids: List[int]
    prompt_mask: List[int]
    completion_ids: List[int]
    completion_mask: List[int]
    rewards: List[float]


try:
    import torch._dynamo  # type: ignore

    torch._dynamo.config.suppress_errors = True  # type: ignore
except ImportError:
    pass

try:
    from .utils.logging_utils import print_prompt_completions_sample, setup_logging  # noqa: F401

    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False

from .envs.env_group import EnvGroup
from .envs.environment import Environment
from .envs.multiturn_env import MultiTurnEnv
from .envs.singleturn_env import SingleTurnEnv
from .envs.tool_env import ToolEnv
from .parsers.parser import Parser
from .parsers.think_parser import ThinkParser
from .parsers.xml_parser import XMLParser
from .rubrics.judge_rubric import JudgeRubric
from .rubrics.rubric import Rubric
from .rubrics.rubric_group import RubricGroup
from .rubrics.tool_rubric import ToolRubric
from .utils.data_utils import extract_boxed_answer, extract_hash_answer, load_example_dataset

# Conditional import based on trl availability
try:
    import trl  # noqa: F401

    from .trainers import GRPOConfig, GRPOTrainer, grpo_defaults, lora_defaults  # noqa: F401
    from .utils.model_utils import get_model, get_model_and_tokenizer, get_tokenizer  # noqa: F401

    _HAS_TRL = True
except ImportError:
    _HAS_TRL = False

__version__ = "0.1.0"


# Setup default logging configuration
def setup_logging(
    level: str = "INFO",
    log_format: Optional[str] = None,
    date_format: Optional[str] = None,
) -> None:
    """
    Setup basic logging configuration for the verifiers package.

    Args:
        level: The logging level to use. Defaults to "INFO".
        log_format: Custom log format string. If None, uses default format.
        date_format: Custom date format string. If None, uses default format.
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    if date_format is None:
        date_format = "%Y-%m-%d %H:%M:%S"

    # Create a StreamHandler that writes to stderr
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(fmt=log_format, datefmt=date_format))

    # Get the root logger for the verifiers package
    logger = logging.getLogger("verifiers")
    logger.setLevel(level.upper())
    logger.addHandler(handler)

    # Prevent the logger from propagating messages to the root logger
    logger.propagate = False


setup_logging()

__all__ = [
    "Parser",
    "ThinkParser",
    "XMLParser",
    "Rubric",
    "JudgeRubric",
    "RubricGroup",
    "ToolRubric",
    "Environment",
    "MultiTurnEnv",
    "SingleTurnEnv",
    "ToolEnv",
    "EnvGroup",
    "extract_boxed_answer",
    "extract_hash_answer",
    "load_example_dataset",
    "setup_logging",
]

# Add trainer exports only if trl is available
if _HAS_TRL:
    __all__.extend(
        [
            "get_model",
            "get_tokenizer",
            "get_model_and_tokenizer",
            "GRPOTrainer",
            "GRPOConfig",
            "grpo_defaults",
            "lora_defaults",
        ]
    )

if _HAS_RICH:
    __all__.extend(
        [
            "print_prompt_completions_sample",
        ]
    )
