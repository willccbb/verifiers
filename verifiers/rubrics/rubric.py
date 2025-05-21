import asyncio
from asyncio import Semaphore

import inspect
import logging
from typing import List, Dict, Any

from verifiers import RewardFunc

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
                 **kwargs):
        self.logger = logging.getLogger(f"verifiers.parsers.{self.__class__.__name__}")
        self.parser = None
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.reward_funcs = funcs
        self.reward_weights = weights
        if not self.reward_weights:
            self.reward_weights = [1.0] * len(self.reward_funcs)

    def get_reward_funcs(self) -> List[RewardFunc]:
        return self.reward_funcs # type: ignore

    def get_reward_weights(self) -> List[float]:
        return self.reward_weights # type: ignore

    def _call_reward_func(self,
                          func: RewardFunc,
                          prompt: List[Dict[str, str]] | str,
                          completion: List[Dict[str, str]] | str,
                          answer: Any,
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
            task=task,
        )
        merged = {**common, **kwargs}
        if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
            return func(**merged)
        allowed = {k: v for k, v in merged.items() if k in sig.parameters}
        return func(**allowed)
    
    async def _score_rollout(self,
                             prompt,
                             completion,
                             answer,
                             task: str | None = None,
                             apply_weights: bool = True,
                             **kwargs) -> Dict[str, float]:
        """
        Evaluate all reward functions asynchronously.

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
        futures = [
            asyncio.to_thread(
                self._call_reward_func,
                func,
                prompt,
                completion,
                answer,
                task=task,
                **kwargs
            )
            for func in self.get_reward_funcs()
        ]
        rewards = await asyncio.gather(*futures)
        # zip rewards with reward functions and weights
        weighted_rewards = {
            func.__name__: reward * weight if weight and apply_weights else reward
            for func, reward, weight in zip(
                self.get_reward_funcs(),
                rewards,
                self.get_reward_weights()
            )
        }
        # add total reward
        total_reward = sum([r for r in weighted_rewards.values()])
        weighted_rewards['reward'] = total_reward
        return weighted_rewards 

    # async def _evaluate_group_async(self,
    #                                 prompts: List[List[Dict[str, str]] | str],
    #                                 completions: List[List[Dict[str, str]] | str],
    #                                 answers: List[Any],
    #                                 tasks: List[str],
    #                                 max_conc: int = 32,
    #                                 apply_weights: bool = True,
    #                                 **kwargs) -> List[Dict[str, float]]:
        
        

    #     """

    #     pattern to use
    #                     semaphore = Semaphore(max_concurrent)
                
    #             # Process all examples concurrently
    #             tasks = [process_example(example, semaphore) for example in eval_dataset]
    #             results = await tqdm_asyncio.gather(
    #                 *tasks,
    #                 total=len(eval_dataset),
    #                 desc=f"Evaluating {len(eval_dataset)} examples"
    #             )
                
        
    #     """

    #     sem = asyncio.Semaphore(max_conc)
    #     async def _one(prompt, completion, answer, task):
    #         async with sem:     
    #             return await self._evaluate_rollout(
    #                 prompt, completion, answer, task, 
    #                 apply_weights=apply_weights, **kwargs)
            
    #     eval_tasks = [
    #         _one(p, c, a, t)
    #         for p, c, a, t in zip(prompts, completions, answers, tasks)
    #     ]
    #     return await asyncio.gather(*eval_tasks)

    def score_rollout_group(self,
                            prompts: List[List[Dict[str, str]] | str],
                            completions: List[List[Dict[str, str]] | str],
                            answers: List[Any],
                            tasks: List[str | None],
                            max_conc: int = 32,
                            apply_weights: bool = True,
                            **kwargs) -> List[Dict[str, float]]:
        """
        Evaluate a group of rollouts.
        
        Default behavior:
        - evaluate each rollout asynchronously 
        - return list of dictionaries of reward function names and their scores

        Potential overrides:
        - inter-group comparisons (voting, ranking, Elo, etc.)
        - scores computed using global state stored in Rubric class
        """

        async def score_rollout(prompt,     
                                completion,
                                answer,
                                task,
                                semaphore,
                                **kwargs):
            async with semaphore:
                return await self._score_rollout(
                    prompt, completion, answer, task,
                    apply_weights=apply_weights, **kwargs)

        async def score_all():
            from tqdm.asyncio import tqdm_asyncio
            semaphore = Semaphore(max_conc)
            rollout_tasks = [
                score_rollout(p, c, a, t, semaphore, **kwargs)
                for p, c, a, t in zip(prompts, completions, answers, tasks)
            ]
            return await tqdm_asyncio.gather(
                *rollout_tasks,
                total=len(prompts),
                desc=f"Evaluating {len(prompts)} rollouts"
            )

        # Run the async evaluation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(score_all())
        finally:
            loop.close()

        return results  