from abc import ABC
from typing import List, Dict, Callable
import logging

from verifiers.trainers.grpo_env_trainer import RewardFunc

class Rubric(ABC):
    def __init__(self, 
                 funcs: List[Callable] = [],
                 weights: List[float] = [],
                 **kwargs):
        self.logger = logging.getLogger(f"verifiers.parsers.{self.__class__.__name__}")
        self.parser = None
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.reward_funcs = funcs
        self.reward_weights = weights

    def get_assistant_messages(self, trajectory: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Helper function to extract assistant messages from a trajectory."""
        return [msg for msg in trajectory if msg['role'] == 'assistant']

    def exact_answer_reward_func(self, completions, answer, **kwargs) -> List[float]:
        """Reward function that checks if the final answer matches the expected answer."""
        responses = [self.parser.get_last_answer(c) for c in completions]
        return [1.0 if str(r) == str(a) else 0.0 for r, a in zip(responses, answer)]

    def int_answer_reward_func(self, completions, answer, **kwargs) -> List[float]:
        """Reward function that checks if the final answer is an integer."""
        responses = [self.parser.get_last_answer(c) for c in completions]
        return [1.0 if str(r).isdigit() else 0.0 for r in responses]

    def get_reward_funcs(self) -> List[RewardFunc]:
        return self.reward_funcs

    def get_reward_weights(self) -> List[float]:
        return self.reward_weights
