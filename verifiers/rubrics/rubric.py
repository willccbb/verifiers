import asyncio
import inspect
import logging
import time
from typing import Callable

from verifiers.parsers.parser import Parser
from verifiers.types import (
    Info,
    Messages,
    RewardFunc,
    RolloutScore,
    RolloutScores,
    State,
)
from verifiers.utils.async_utils import maybe_await


def standardize_groups(rewards: list[float], group_size: int) -> list[float]:
    """
    Standardize rewards within groups by subtracting mean and dividing by std.

    Each group will have mean ≈ 0 and std ≈ 1 after transformation.
    If a group has std = 0 (all same values), returns 0.0 for that group.

    Args:
        rewards: List of rewards to transform
        group_size: Number of consecutive rewards per group

    Returns:
        List of standardized rewards
    """
    n_groups = len(rewards) // group_size
    result = []

    for i in range(n_groups):
        start_idx = i * group_size
        end_idx = start_idx + group_size
        group = rewards[start_idx:end_idx]

        mean = sum(group) / len(group)
        variance = sum((r - mean) ** 2 for r in group) / len(group)
        std = variance**0.5

        if std < 1e-10:  # Avoid division by zero
            standardized = [0.0] * group_size
        else:
            standardized = [(r - mean) / std for r in group]

        result.extend(standardized)

    return result


def normalize_groups(rewards: list[float], group_size: int) -> list[float]:
    """
    Normalize rewards within groups using min-max normalization.

    Each group will have values in range [0, 1] after transformation.
    If a group has min = max (all same values), returns 0.0 for that group.

    Args:
        rewards: List of rewards to transform
        group_size: Number of consecutive rewards per group

    Returns:
        List of normalized rewards
    """
    n_groups = len(rewards) // group_size
    result = []

    for i in range(n_groups):
        start_idx = i * group_size
        end_idx = start_idx + group_size
        group = rewards[start_idx:end_idx]

        min_val = min(group)
        max_val = max(group)
        range_val = max_val - min_val

        if range_val < 1e-10:  # Avoid division by zero
            normalized = [0.0] * group_size
        else:
            normalized = [(r - min_val) / range_val for r in group]

        result.extend(normalized)

    return result


def rank_groups(rewards: list[float], group_size: int) -> list[float]:
    """
    Convert rewards to ranks within groups.

    Rank 0 is the lowest reward, rank (group_size - 1) is the highest.
    Ties are broken by position (earlier positions get lower ranks).

    Args:
        rewards: List of rewards to transform
        group_size: Number of consecutive rewards per group

    Returns:
        List of rank-based rewards
    """
    n_groups = len(rewards) // group_size
    result = []

    for i in range(n_groups):
        start_idx = i * group_size
        end_idx = start_idx + group_size
        group = rewards[start_idx:end_idx]

        # Create list of (value, original_index) pairs
        indexed_group = [(val, idx) for idx, val in enumerate(group)]
        # Sort by value (ascending), then by index for tie-breaking
        sorted_group = sorted(indexed_group, key=lambda x: (x[0], x[1]))

        # Assign ranks
        ranks = [0.0] * group_size
        for rank, (val, orig_idx) in enumerate(sorted_group):
            ranks[orig_idx] = float(rank)

        result.extend(ranks)

    return result


class Rubric:
    """
    Rubric class for reward functions.

    Each reward function takes:
    - prompt: list[dict[str, str]] | str
    - completion: list[dict[str, str]] | str
    - answer: Any (metadata for scoring)
    - task (optional): str (type of task)
    - **kwargs: additional kwargs

    Returns:
    - float | list[float] | dict[str, float]
    """

    def __init__(
        self,
        funcs: list[RewardFunc] | None = None,
        weights: list[float] | None = None,
        parser: Parser | None = None,
        parallelize_scoring: bool = True,
        **kwargs,
    ):
        self.logger = logging.getLogger(f"verifiers.rubrics.{self.__class__.__name__}")

        self.reward_funcs = funcs or []
        self.reward_weights = weights or []
        self.parser = parser or Parser()

        for key, value in kwargs.items():
            setattr(self, key, value)
        if not self.reward_weights:
            self.reward_weights = [1.0] * len(self.reward_funcs)
        self.parallelize_scoring = parallelize_scoring
        # class objects for reward functions
        self.class_objects = {}
        if self.parser:
            self.class_objects["parser"] = self.parser

    def get_reward_func_names(self) -> list[str]:
        return [func.__name__ for func in self.reward_funcs]

    def get_reward_funcs(self) -> list[RewardFunc]:
        return self.reward_funcs  # type: ignore

    def get_reward_weights(self) -> list[float]:
        return self.reward_weights  # type: ignore

    def add_reward_func(self, func: RewardFunc, weight: float = 1.0):
        self.reward_funcs.append(func)
        self.reward_weights.append(weight)

    async def call_reward_func(
        self,
        func: RewardFunc,
        prompt: Messages,
        completion: Messages,
        answer: str,
        state: State,
        task: str = "default",
        info: Info | None = None,
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
        info = info or {}
        sig = inspect.signature(func)

        common = dict(
            prompt=prompt,
            completion=completion,
            answer=answer,
            state=state,
            task=task,
            info=info,
        )
        common.update(self.class_objects)
        merged = {**common, **kwargs}
        if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
            try:
                ans = float(await maybe_await(func, **merged))
            except Exception as e:
                self.logger.error(f"Error calling reward function {func.__name__}: {e}")
                ans = 0.0
        else:
            allowed = {k: v for k, v in merged.items() if k in sig.parameters}
            try:
                ans = float(await maybe_await(func, **allowed))
            except Exception as e:
                self.logger.error(f"Error calling reward function {func.__name__}: {e}")
                ans = 0.0
        return ans

    async def score_rollout_with_semaphore(
        self,
        semaphore: asyncio.Semaphore,
        *args,
        **kwargs,
    ) -> RolloutScore:
        """
        Score a rollout with a semaphore.
        """
        async with semaphore:
            return await self.score_rollout(*args, **kwargs)

    async def score_rollout(
        self,
        prompt: Messages,
        completion: Messages,
        answer: str,
        state: State,
        task: str = "default",
        info: Info | None = None,
        **kwargs,
    ) -> RolloutScore:
        """
        Evaluate all reward functions asynchronously for a single rollout.
        """
        # start timer
        start_time = time.time()
        if self.parallelize_scoring:
            score_tasks = [
                self.call_reward_func(
                    func=func,
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
                    prompt=prompt,
                    completion=completion,
                    answer=answer,
                    state=state,
                    task=task,
                    info=info,
                    **kwargs,
                )
                reward_scores.append(score)
        rewards = RolloutScore(
            metrics={
                func.__name__: reward
                for func, reward in zip(self.get_reward_funcs(), reward_scores)
            },
            reward=sum(
                [
                    reward * weight
                    for reward, weight in zip(reward_scores, self.get_reward_weights())
                ]
            ),
        )
        end_time = time.time()
        state["timing"]["scoring_ms"] = (end_time - start_time) * 1000
        state["timing"]["total_ms"] += state["timing"]["scoring_ms"]
        return rewards

    async def score_rollouts(
        self,
        prompts: list[Messages],
        completions: list[Messages],
        answers: list[str],
        states: list[State],
        tasks: list[str],
        infos: list[Info],
        max_concurrent: int = -1,
        group_size: int | None = None,
        group_transform: str | Callable[[list[float]], list[float]] | None = None,
        **kwargs,
    ) -> RolloutScores:
        """
        Compute reward scores for a group of rollouts.

        Default behavior:
        - evaluate each rollout asynchronously
        - return list of dictionaries of reward function names and their scores

        Group-level transformations:
        - If group_size is specified, rollouts are grouped and group_transform is applied
        - Built-in transforms: "standardize", "normalize", "rank"
        - Custom transforms can be provided as callables

        Args:
            prompts: List of prompts
            completions: List of completions
            answers: List of answers
            states: List of states
            tasks: List of tasks
            infos: List of infos
            max_concurrent: Maximum number of concurrent scoring operations
            group_size: Number of consecutive rollouts per group (for group-level transforms)
            group_transform: Transformation to apply within groups. Can be:
                - "standardize": subtract mean, divide by std
                - "normalize": min-max normalization to [0, 1]
                - "rank": convert to ranks (0 to group_size-1)
                - callable: custom function taking list[float] and returning list[float]
            **kwargs: Additional arguments passed to reward functions

        Returns:
            RolloutScores with rewards and metrics
        """
        from tqdm.asyncio import tqdm_asyncio

        # Validate group_size if provided
        if group_size is not None:
            if len(prompts) % group_size != 0:
                raise ValueError(
                    f"Number of rollouts ({len(prompts)}) must be divisible by group_size ({group_size})"
                )

        if max_concurrent > 0:
            semaphore = asyncio.Semaphore(max_concurrent)
            rollout_tasks = [
                self.score_rollout_with_semaphore(semaphore, *pcasti, **kwargs)
                for pcasti in zip(prompts, completions, answers, states, tasks, infos)
            ]
        else:
            rollout_tasks = [
                self.score_rollout(*pcasti, **kwargs)
                for pcasti in zip(prompts, completions, answers, states, tasks, infos)
            ]

        rewards = await tqdm_asyncio.gather(
            *rollout_tasks,
            total=len(prompts),
            desc=f"Evaluating {len(prompts)} rollouts",
        )

        if not rewards:
            reward_func_names = self.get_reward_func_names()
            return RolloutScores(
                reward=[],
                metrics={name: [] for name in reward_func_names},
            )

        # Extract rewards and metrics
        reward_values = [reward.reward for reward in rewards]
        metrics_dict = {
            k: [item.metrics[k] for item in rewards] for k in rewards[0].metrics
        }

        # Apply group-level transformation if specified
        if group_size is not None and group_transform is not None:
            # Determine which transform function to use
            if isinstance(group_transform, str):
                if group_transform == "standardize":

                    def transform_fn(vals: list[float]) -> list[float]:
                        return standardize_groups(vals, group_size)

                elif group_transform == "normalize":

                    def transform_fn(vals: list[float]) -> list[float]:
                        return normalize_groups(vals, group_size)

                elif group_transform == "rank":

                    def transform_fn(vals: list[float]) -> list[float]:
                        return rank_groups(vals, group_size)

                else:
                    raise ValueError(
                        f"Unknown group_transform: {group_transform}. "
                        f"Must be 'standardize', 'normalize', 'rank', or a callable."
                    )
            elif callable(group_transform):
                # Custom transform function - apply to each group
                def transform_fn(vals: list[float]) -> list[float]:
                    n_groups = len(vals) // group_size
                    result = []
                    for i in range(n_groups):
                        start_idx = i * group_size
                        end_idx = start_idx + group_size
                        group = vals[start_idx:end_idx]
                        transformed_group = group_transform(group)
                        result.extend(transformed_group)
                    return result

            else:
                raise ValueError(
                    f"group_transform must be a string or callable, got {type(group_transform)}"
                )

            # Apply transformation to main reward
            reward_values = transform_fn(reward_values)

            # Apply transformation to each metric
            for metric_name in metrics_dict:
                metrics_dict[metric_name] = transform_fn(metrics_dict[metric_name])

        return RolloutScores(
            reward=reward_values,
            metrics=metrics_dict,
        )
