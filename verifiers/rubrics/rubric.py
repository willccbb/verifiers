import asyncio
from asyncio import Semaphore
import concurrent.futures

import inspect
import logging
from typing import List, Dict, Any, Union

from verifiers import RewardFunc
from verifiers.parsers import Parser


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

    def __init__(self, 
                 funcs: List[RewardFunc] = [],
                 weights: List[float] = [],
                 parser: Parser = Parser(),
                 **kwargs):
        self.logger = logging.getLogger(f"verifiers.rubrics.{self.__class__.__name__}")
        self.parser = parser
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.reward_funcs = funcs
        self.reward_weights = weights
        if not self.reward_weights:
            self.reward_weights = [1.0] * len(self.reward_funcs)

    def get_reward_func_names(self) -> List[str]:
        return [func.__name__ for func in self.reward_funcs]

    def get_reward_funcs(self) -> List[RewardFunc]:
        return self.reward_funcs # type: ignore

    def get_reward_weights(self) -> List[float]:
        return self.reward_weights # type: ignore

    def add_reward_func(self, func: RewardFunc, weight: float = 1.0):
        self.reward_funcs.append(func)
        self.reward_weights.append(weight)

    def _call_reward_func(self,
                          func: RewardFunc,
                          prompt: Union[str, List[Dict[str, Any]]],
                          completion: Union[str, List[Dict[str, Any]]],
                          answer: Any,
                          state: Dict[str, Any],
                          task: str | None,
                          **kwargs) -> float:
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
            prompt=prompt,
            completion=completion,
            answer=answer,
            state=state,
            task=task,
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
    
    async def score_rollout(self,
                            prompt: Union[str, List[Dict[str, Any]]],
                            completion: Union[str, List[Dict[str, Any]]],
                            answer: Any,
                            state: Dict[str, Any],
                            task: str | None = None,
                            **kwargs) -> Dict[str, float]:
        """
        Evaluate all reward functions asynchronously for a single rollout.
        """
        futures = [
            asyncio.to_thread(
                self._call_reward_func,
                func,
                prompt,
                completion,
                answer,
                state,
                task=task,
                **kwargs
            )
            for func in self.get_reward_funcs()
        ]
        reward_scores = await asyncio.gather(*futures)
        rewards = {func.__name__: reward for func, reward in zip(self.get_reward_funcs(), reward_scores)}
        rewards['reward'] = sum([reward * weight for reward, weight in zip(reward_scores, self.get_reward_weights())])
        return rewards

    async def _score_single(self, semaphore, *pcast, **kw):
        async with semaphore:
            return await self.score_rollout(*pcast, **kw)

    async def _score_all(
            self, prompts, completions, answers, states, tasks,
            max_concurrent: int = 128,
            **kwargs) -> Dict[str, List[float]]:
        from tqdm.asyncio import tqdm_asyncio
        semaphore = Semaphore(max_concurrent)
        rollout_tasks = [
            self._score_single(semaphore, *pcast, **kwargs)
            for pcast in zip(prompts, completions, answers, states, tasks)
        ]
        rewards = await tqdm_asyncio.gather(
            *rollout_tasks,
            total=len(prompts),
            desc=f"Evaluating {len(prompts)} rollouts"
        )
        return {k: [item[k] for item in rewards] for k in rewards[0]}
    
    def score_rollouts(self,
                       prompts: List[Union[str, List[Dict[str, Any]]]],
                       completions: List[Union[str, List[Dict[str, Any]]]],
                       answers: List[Any],
                       states: List[Dict[str, Any]],
                       tasks: List[str],
                       max_concurrent: int = 128,
                       **kwargs) -> Dict[str, List[float]]:
        """
        Compute reward scores for a group of rollouts.
        
        Default behavior:
        - evaluate each rollout asynchronously 
        - return list of dictionaries of reward function names and their scores

        Potential overrides:
        - inter-group comparisons (voting, ranking, Elo, etc.)
        - scores computed using global state stored in Rubric class
        """
        # Set up custom executor for the event loop if needed
        def setup_executor(loop):
            if loop._default_executor is None:
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent)
                loop.set_default_executor(executor)
        
        coro = self._score_all(
            prompts, completions, answers, states, tasks,
            max_concurrent=max_concurrent,
            **kwargs
        )
        try:
            # Create new event loop with custom executor
            loop = asyncio.new_event_loop()
            setup_executor(loop)
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()
                asyncio.set_event_loop(None)
        except RuntimeError:
            # Jupyter notebook or existing event loop
            import nest_asyncio 
            nest_asyncio.apply()
            loop = asyncio.get_running_loop()
            setup_executor(loop)
            return loop.run_until_complete(coro)