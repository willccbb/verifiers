from typing import Callable, Union
from transformers import PreTrainedModel 
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

import torch._dynamo
torch._dynamo.config.suppress_errors = True # type: ignore

from .envs.environment import Environment
from .envs.code_env import CodeEnv
from .envs.doublecheck_env import DoubleCheckEnv
from .envs.singleturn_env import SingleTurnEnv
from .envs.simple_env import SimpleEnv
from .envs.tool_env import ToolEnv
from .parsers.xml_parser import XMLParser
from .rubrics.rubric import Rubric
from .trainers.grpo_env_trainer import GRPOEnvTrainer
from .utils.data_utils import extract_boxed_answer, extract_hash_answer, preprocess_dataset
from .utils.model_utils import get_model, get_tokenizer, get_model_and_tokenizer
from .utils.config_utils import grpo_defaults, lora_defaults
from .utils.logging_utils import setup_logging, print_prompt_completions_sample

__version__ = "0.1.0"

# Setup default logging configuration
setup_logging()

__all__ = [
    "Environment",
    "CodeEnv",
    "DoubleCheckEnv",
    "SingleTurnEnv",
    "SimpleEnv",
    "ToolEnv",
    "Rubric",
    "GRPOEnvTrainer",
    "XMLParser",
    "get_model",
    "get_tokenizer",
    "get_model_and_tokenizer",
    "grpo_defaults",
    "lora_defaults",
    "extract_boxed_answer",
    "extract_hash_answer",
    "preprocess_dataset",
    "setup_logging",
    "print_prompt_completions_sample",
]