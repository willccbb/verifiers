from typing import Callable

from verifiers.rubrics.rubric import Rubric
from verifiers.types import (
    Info,
    Messages,
    RewardFunc,
    RolloutScore,
    RolloutScores,
    State,
)


class RubricGroup(Rubric):
    """
    Class for aggregating multiple rubrics.
    """

    def __init__(self, rubrics: list[Rubric], **kwargs):
        if not rubrics:
            raise ValueError("RubricGroup must have at least one rubric")
        super().__init__(**kwargs)
        self.rubrics = rubrics
        self.logger.info(f"Initialized RubricGroup with {len(rubrics)} rubrics")

    def get_reward_func_names(self) -> list[str]:
        names = []
        for rubric in self.rubrics:
            names.extend(rubric.get_reward_func_names())
        return names

    def get_reward_funcs(self) -> list[RewardFunc]:
        funcs = []
        for rubric in self.rubrics:
            funcs.extend(rubric.get_reward_funcs())
        return funcs

    def get_reward_weights(self) -> list[float]:
        weights = []
        for rubric in self.rubrics:
            weights.extend(rubric.get_reward_weights())
        return weights

    def add_reward_func(self, func: RewardFunc, weight: float = 1.0):
        assert len(self.rubrics) > 0, "RubricGroup must have at least one rubric"
        self.logger.warning("Adding reward function to the first rubric in the group.")
        self.rubrics[0].add_reward_func(func, weight)

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
        total_reward = 0.0
        aggregated_metrics: dict[str, float] = {}
        for rubric in self.rubrics:
            score = await rubric.score_rollout(
                prompt,
                completion,
                answer,
                state,
                task,
                info,
                **kwargs,
            )
            total_reward += score.reward
            for key, value in score.metrics.items():
                aggregated_metrics[key] = aggregated_metrics.get(key, 0.0) + value
        return RolloutScore(reward=total_reward, metrics=aggregated_metrics)

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
        Run all rubrics sequentially and return the aggregated scores.

        Reward functions with the same name are summed up.

        Note: group_transform is applied AFTER aggregating all rubric scores.
        If you need different group transforms per rubric, use individual rubrics.
        """
        all_scores = RolloutScores(
            reward=[],
            metrics={},
        )
        for rubric in self.rubrics:
            # Don't apply group_transform yet - do it after aggregating all rubrics
            rubric_scores = await rubric.score_rollouts(
                prompts,
                completions,
                answers,
                states,
                tasks,
                infos,
                max_concurrent,
                group_size=None,  # Don't transform yet
                group_transform=None,  # Don't transform yet
                **kwargs,
            )
            # aggregate reward (element-wise sum across rubrics)
            if not all_scores.reward:
                all_scores.reward = rubric_scores.reward
            else:
                all_scores.reward = [
                    a + b for a, b in zip(all_scores.reward, rubric_scores.reward)
                ]
            for key, value in rubric_scores.metrics.items():
                if key in all_scores.metrics:
                    # element-wise sum
                    all_scores.metrics[key] = [
                        a + b for a, b in zip(all_scores.metrics[key], value)
                    ]
                else:
                    all_scores.metrics[key] = value

        # Apply group transformation to aggregated scores if requested
        if group_size is not None and group_transform is not None:
            # Import transformation functions
            from verifiers.rubrics.rubric import (
                normalize_groups,
                rank_groups,
                standardize_groups,
            )

            # Validate group_size
            if len(prompts) % group_size != 0:
                raise ValueError(
                    f"Number of rollouts ({len(prompts)}) must be divisible by group_size ({group_size})"
                )

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
            all_scores.reward = transform_fn(all_scores.reward)

            # Apply transformation to each metric
            for metric_name in all_scores.metrics:
                all_scores.metrics[metric_name] = transform_fn(
                    all_scores.metrics[metric_name]
                )

        return all_scores
