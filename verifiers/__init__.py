from typing import Callable
from transformers import PreTrainedModel # type: ignore 
RewardFunc = Callable[..., float]

import torch._dynamo
torch._dynamo.config.suppress_errors = True # type: ignore


from .parsers.parser import Parser
from .parsers.xml_parser import XMLParser
from .rubrics.rubric import Rubric

from .envs.environment import Environment
from .envs.multiturn_env import MultiTurnEnv

from .envs.code_env import CodeEnv
from .envs.doublecheck_env import DoubleCheckEnv
from .envs.singleturn_env import SingleTurnEnv
from .envs.simple_env import SimpleEnv
from .envs.tool_env import ToolEnv
from .trainers.grpo_env_trainer import GRPOEnvTrainer
from .utils.data_utils import extract_boxed_answer, extract_hash_answer, preprocess_dataset
from .utils.model_utils import get_model, get_tokenizer, get_model_and_tokenizer
from .utils.config_utils import grpo_defaults, lora_defaults, GRPOEnvConfig
from .utils.logging_utils import setup_logging, print_prompt_completions_sample

__version__ = "0.1.0"

# Setup default logging configuration
setup_logging()

__all__ = [
    "Parser",
    "XMLParser",
    "Rubric",
    "Environment",
    "MultiTurnEnv",
    "CodeEnv",
    "DoubleCheckEnv",
    "SingleTurnEnv",
    "SimpleEnv",
    "ToolEnv",
    "GRPOEnvTrainer",
    "GRPOEnvConfig",
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