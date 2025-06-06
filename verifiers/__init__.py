from typing import Callable
RewardFunc = Callable[..., float]

import torch._dynamo
torch._dynamo.config.suppress_errors = True # type: ignore

from .parsers.parser import Parser
from .parsers.xml_parser import XMLParser

from .rubrics.rubric import Rubric
from .rubrics.judge_rubric import JudgeRubric
from .rubrics.rubric_group import RubricGroup

from .envs.environment import Environment
from .envs.multiturn_env import MultiTurnEnv
from .envs.singleturn_env import SingleTurnEnv
from .envs.tool_env import ToolEnv

from .utils.logging_utils import setup_logging, print_prompt_completions_sample
from .trainers.grpo_trainer import GRPOTrainer
from .trainers.grpo_config import GRPOConfig
from .utils.data_utils import extract_boxed_answer, extract_hash_answer, load_example_dataset
from .utils.model_utils import get_model, get_tokenizer, get_model_and_tokenizer
from .utils.config_utils import grpo_defaults, lora_defaults

__version__ = "0.1.0"

# Setup default logging configuration
setup_logging()

__all__ = [
    "Parser",
    "XMLParser",
    "Rubric",
    "JudgeRubric",
    "RubricGroup",
    "Environment",
    "MultiTurnEnv",
    "SingleTurnEnv",
    "ToolEnv",
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