from typing import Callable
RewardFunc = Callable[..., float]

import torch._dynamo
torch._dynamo.config.suppress_errors = True # type: ignore

from .utils.logging_utils import setup_logging, print_prompt_completions_sample
from .utils.data_utils import extract_boxed_answer, extract_hash_answer, load_example_dataset
from .utils.model_utils import get_model, get_tokenizer, get_model_and_tokenizer

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

from .trainers import GRPOTrainer, GRPOConfig, grpo_defaults, lora_defaults

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
    "GRPOTrainer",
    "GRPOConfig",
    "get_model",
    "get_tokenizer",
    "get_model_and_tokenizer",
    "grpo_defaults",
    "lora_defaults",
    "extract_boxed_answer",
    "extract_hash_answer",
    "load_example_dataset",
    "setup_logging",
    "print_prompt_completions_sample",
]