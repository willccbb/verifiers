from typing import Callable
RewardFunc = Callable[..., float]

try:
    import torch._dynamo # type: ignore
    torch._dynamo.config.suppress_errors = True # type: ignore
except ImportError:
    pass

try:
    from .utils.logging_utils import setup_logging
    from .utils.logging_utils import print_prompt_completions_sample
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False

from .utils.data_utils import extract_boxed_answer, extract_hash_answer, load_example_dataset

from .parsers.parser import Parser
from .parsers.think_parser import ThinkParser
from .parsers.xml_parser import XMLParser

from .rubrics.rubric import Rubric
from .rubrics.judge_rubric import JudgeRubric
from .rubrics.rubric_group import RubricGroup

from .envs.environment import Environment
from .envs.multiturn_env import MultiTurnEnv
from .envs.singleturn_env import SingleTurnEnv
from .envs.tool_env import ToolEnv
from .envs.env_group import EnvGroup

# Conditional import based on trl availability
try:
    import trl # type: ignore
    from .utils.model_utils import get_model, get_tokenizer, get_model_and_tokenizer
    from .trainers import GRPOTrainer, GRPOConfig, grpo_defaults, lora_defaults
    _HAS_TRL = True
except ImportError:
    _HAS_TRL = False

__version__ = "0.1.0"

# Setup default logging configuration
setup_logging()

__all__ = [
    "Parser",
    "ThinkParser",
    "XMLParser",
    "Rubric",
    "JudgeRubric",
    "RubricGroup",
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
    __all__.extend([
        "get_model",
        "get_tokenizer",
        "get_model_and_tokenizer",
        "GRPOTrainer",
        "GRPOConfig",
        "grpo_defaults",
        "lora_defaults",
    ])

if _HAS_RICH:
    __all__.extend([
        "print_prompt_completions_sample",
    ])