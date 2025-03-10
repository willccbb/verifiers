from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence, Union
import logging

from datasets import Dataset
from trl.trainer.grpo_trainer import RewardFunc
from ..imports import LLM, SamplingParams  # type: ignore

class Environment(ABC):

    def __init__(self, **kwargs: Any):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.logger = logging.getLogger(f"verifiers.envs.{self.__class__.__name__}")
        self.tokenizer = None
        self.dataset = None
        self.eval_dataset = None

    @abstractmethod
    def get_dataset(self, **kwargs: Any) -> Dataset | None:
        pass

    @abstractmethod
    def get_eval_dataset(self, **kwargs: Any) -> Dataset | None:
        pass

    @abstractmethod
    def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
        pass
    
    @abstractmethod
    def generate(self,
                 prompts: List[List[Dict[str, Any]]],
                 llm: LLM,
                 sampling_params: SamplingParams,
                 **kwargs: Any) -> Dict[str, List[Sequence[int]] | List[str] | List[List[Dict[str, Any]]]]:
        pass
