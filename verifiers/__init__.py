from typing import Callable, Union
from transformers import PreTrainedModel
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

from .envs.environment import Environment
from .envs.code_env import CodeEnv
from .envs.doublecheck_env import DoubleCheckEnv
from .envs.simple_env import SimpleEnv
from .envs.tool_env import ToolEnv
from .trainers.grpo_env_trainer import GRPOEnvTrainer
from .utils.data_utils import extract_boxed_answer, extract_hash_answer, preprocess_dataset
from .utils.model_utils import get_model, get_tokenizer, get_model_and_tokenizer
from .utils.config_utils import get_default_grpo_config
from .utils.logging_utils import setup_logging, print_prompt_completions_sample


__version__ = "0.1.0"

# Setup default logging configuration
setup_logging()

__all__ = [
    "Environment",
    "CodeEnv",
    "DoubleCheckEnv",
    "SimpleEnv",
    "ToolEnv",
    "GRPOEnvTrainer",
    "get_model",
    "get_tokenizer",
    "get_model_and_tokenizer",
    "get_default_grpo_config",
    "extract_boxed_answer",
    "extract_hash_answer",
    "preprocess_dataset",
    "setup_logging",
    "print_prompt_completions_sample",
]