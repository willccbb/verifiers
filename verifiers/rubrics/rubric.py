import asyncio
import inspect
import logging
from typing import Dict, List, Union

from verifiers.parsers.parser import Parser
from verifiers.types import ChatMessage, Info, RewardFunc, State


class Rubric:
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

    def __init__(
        self,
        funcs: List[RewardFunc] = [],
        weights: List[float] = [],
        parser: Parser = Parser(),
        parallelize_scoring: bool = True,
        **kwargs,
    ):
        self.logger = logging.getLogger(f"verifiers.rubrics.{self.__class__.__name__}")
        self.parser = parser
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.reward_funcs = funcs
        self.reward_weights = weights
        if not self.reward_weights:
            self.reward_weights = [1.0] * len(self.reward_funcs)
        self.parallelize_scoring = parallelize_scoring

    def get_reward_func_names(self) -> List[str]:
        return [func.__name__ for func in self.reward_funcs]

    def get_reward_funcs(self) -> List[RewardFunc]:
        return self.reward_funcs  # type: ignore

    def get_reward_weights(self) -> List[float]:
        return self.reward_weights  # type: ignore

    def add_reward_func(self, func: RewardFunc, weight: float = 1.0):
        self.reward_funcs.append(func)
        self.reward_weights.append(weight)

    async def call_reward_func(
        self,
        func: RewardFunc,
        parser: Parser,
        prompt: Union[str, List[ChatMessage]],
        completion: Union[str, List[ChatMessage]],
        answer: str,
        state: State,
        task: str = "default",
        info: Info = {},
        **kwargs,
    ) -> float:
        """
        Invoke `func` with only the required arguments.

        Example:
        ```
        def func(completion, answer, **kwargs):
            ...
        ``
        """
        sig = inspect.signature(func)

        common = dict(
            parser=parser,
            prompt=prompt,
            completion=completion,
            answer=answer,
            state=state,
            task=task,
            info=info,
        )
        ans = 0.0
        merged = {**common, **kwargs}
        if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
            try:
                ans = func(**merged)
            except Exception as e:
                self.logger.error(f"Error calling reward function {func.__name__}: {e}")
                ans = 0.0
        else:
            allowed = {k: v for k, v in merged.items() if k in sig.parameters}
            try:
                ans = func(**allowed)
            except Exception as e:
                self.logger.error(f"Error calling reward function {func.__name__}: {e}")
                ans = 0.0
        return ans

    async def score_rollout(
        self,
        prompt: Union[str, List[ChatMessage]],
        completion: Union[str, List[ChatMessage]],
        answer: str,
        state: State,
        task: str = "default",
        info: Info = {},
        **kwargs,
    ) -> Dict[str, float]:
        """
        Evaluate all reward functions asynchronously for a single rollout.
        """
        if self.parallelize_scoring:
            score_tasks = [
                self.call_reward_func(
                    func=func,
                    parser=self.parser,
                    prompt=prompt,
                    completion=completion,
                    answer=answer,
                    state=state,
                    task=task,
                    info=info,
                    **kwargs,
                )
                for func in self.get_reward_funcs()
            ]
            reward_scores = await asyncio.gather(*score_tasks)
        else:
            reward_scores = []
            for func in self.get_reward_funcs():
                score = await self.call_reward_func(
                    func=func,
                    parser=self.parser,
                    prompt=prompt,
                    completion=completion,
                    answer=answer,
                    state=state,
                    task=task,
                    info=info,
                    **kwargs,
                )
                reward_scores.append(score)
        rewards = {
            func.__name__: reward
            for func, reward in zip(self.get_reward_funcs(), reward_scores)
        }
        rewards["reward"] = sum(
            [
                reward * weight
                for reward, weight in zip(reward_scores, self.get_reward_weights())
            ]
        )
        return rewards

    async def score_rollouts(
        self,
        prompts: List[Union[str, List[ChatMessage]]],
        completions: List[Union[str, List[ChatMessage]]],
        answers: List[str],
        states: List[State],
        tasks: List[str],
        infos: List[Info] = [],
        **kwargs,
    ) -> Dict[str, List[float]]:
        """
        Compute reward scores for a group of rollouts.

        Default behavior:
        - evaluate each rollout asynchronously
        - return list of dictionaries of reward function names and their scores

        Potential overrides:
        - inter-group comparisons (voting, ranking, Elo, etc.)
        - scores computed using global state stored in Rubric class
        """
        from tqdm.asyncio import tqdm_asyncio

        rollout_tasks = [
            self.score_rollout(*pcasti, **kwargs)
            for pcasti in zip(prompts, completions, answers, states, tasks, infos)
        ]
        rewards = await tqdm_asyncio.gather(
            *rollout_tasks,
            total=len(prompts),
            desc=f"Evaluating {len(prompts)} rollouts",
        )

        # Handle empty rewards list
        if not rewards:
            # Return empty dict with keys for each reward function
            reward_func_names = self.get_reward_func_names()
            return {name: [] for name in reward_func_names + ["reward"]}

        return {k: [item[k] for item in rewards] for k in rewards[0]}
