from abc import ABC
from typing import List, Dict, Callable
import logging

from verifiers.trainers.grpo_env_trainer import RewardFunc

class Rubric(ABC):
    """
    Rubric class for reward functions.

    Each reward function takes:
    - prompt: List[Dict[str, str]] | str 
    - completion: List[Dict[str, str]] | str
    - answer: Any (metadata for scoring)
    - task (optional): str (type of task)
    - **kwargs: additional kwargs

    Returns:
    - float | List[float] | Dict[str, float]
    """

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

    def exact_answer_reward_func(self, completion, answer, **kwargs) -> List[float]:
        """Reward function that checks if the final answer matches the expected answer."""
        responses = [self.parser.get_final_answer(c) for c in completions]
        return [1.0 if str(r) == str(a) else 0.0 for r, a in zip(responses, answer)]

    def int_answer_reward_func(self, completion, answer, **kwargs) -> List[float]:
        """Reward function that checks if the final answer is an integer."""
        responses = [self.parser.get_final_answer(c) for c in completions]
        return [1.0 if str(r).isdigit() else 0.0 for r in responses]

    def get_reward_funcs(self) -> List[RewardFunc]:
        return self.reward_funcs

    def get_reward_weights(self) -> List[float]:
        return self.reward_weights

    def evaluate_rollout(self,
                         prompt,
                         completion,
                         answer,
                         task,
                         max_workers: int = 3,
                         **kwargs) -> Dict[str, float]:
        """
        Evaluate all reward functions in parallel with ThreadPoolExecutor.

        Args:
            prompt: List[Dict[str, str]] | str
            completion: List[Dict[str, str]] | str
            answer: Any
            task: str
            max_workers: int
            **kwargs: additional kwargs

        Returns:
            Dict[str, float]: Dictionary of reward function names and their scores.
        """
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(func, prompt, completion, answer, task, **kwargs) for func in self.get_reward_funcs()]
            rewards = [future.result() for future in futures]

        # zip rewards with reward functions and weights
        weighted_rewards = {
            func.__name__: reward * weight 
            for func, reward, weight in zip(
                self.get_reward_funcs(),
                rewards,
                self.get_reward_weights()
            )
        }
        return weighted_rewards

    def evaluate_rollout_group(self,
                               prompts: List[List[Dict[str, str]] | str],
                               completions: List[List[Dict[str, str]] | str],
                               answers: List[Any],
                               tasks: List[str],
                               max_workers: int = 12,
                               **kwargs) -> List[Dict[str, float]]:
        """
        Evaluate a group of rollouts.
        
        Default behavior:
        - evaluate each rollout independently with ThreadPoolExecutor
        - return list of dictionaries of reward function names and their scores

        Potential overrides:
        - inter-group comparisons (voting, ranking, Elo, etc.)
        - scores computed using global state stored in Rubric class
        """
        with ThreadPoolExecutor(max_workers=max_workers) as executor: