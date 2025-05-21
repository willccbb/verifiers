from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence, Callable
import logging

from datasets import Dataset
from openai import OpenAI

from verifiers import Parser, Rubric
from ..imports import SamplingParams  # type: ignore

class Environment(ABC):

    def __init__(self,
                 dataset: Dataset | None = None,
                 eval_dataset: Dataset | None = None,
                 parser: Parser = Parser(),
                 rubric: Rubric = Rubric(),
                 **kwargs: Any):
        #self.client = None
        #self.model = None
        self.dataset = dataset
        self.eval_dataset = eval_dataset
        self.parser = parser
        self.rubric = rubric
        self.logger = logging.getLogger(f"verifiers.envs.{self.__class__.__name__}")
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        if self.dataset is None and self.eval_dataset is None:
            raise ValueError("Either dataset or eval_dataset must be provided")
        

    def get_dataset(self, **kwargs: Any) -> Dataset | None:
        return self.dataset

    def get_eval_dataset(self, **kwargs: Any) -> Dataset | None:
        return self.eval_dataset
 
    @abstractmethod
    def generate(self,
                 prompts: List[List[Dict[str, Any]]],
                 client: OpenAI,
                 sampling_params: SamplingParams,
                 **kwargs: Any) -> Dict[str, Any]:
        pass
