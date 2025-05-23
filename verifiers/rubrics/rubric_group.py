from typing import List, Dict, Any

from verifiers import RewardFunc
from verifiers.rubrics.rubric import Rubric


class RubricGroup(Rubric):
    """
    Class for aggregating multiple rubrics.
    """
    def __init__(self, rubrics: List[Rubric], **kwargs):
        self.rubrics = rubrics
        assert len(rubrics) > 0, "RubricGroup must have at least one rubric"
        super().__init__(**kwargs)
        self.logger.info(f"Initialized RubricGroup with {len(rubrics)} rubrics")

    def get_reward_func_names(self) -> List[str]:
        names = []
        for rubric in self.rubrics:
            names.extend(rubric.get_reward_func_names())
        return names

    def get_reward_funcs(self) -> List[RewardFunc]:
        funcs = []
        for rubric in self.rubrics:
            funcs.extend(rubric.get_reward_funcs())
        return funcs

    def get_reward_weights(self) -> List[float]:
        weights = []
        for rubric in self.rubrics:
            weights.extend(rubric.get_reward_weights())
        return weights
    
    def add_reward_func(self, func: RewardFunc, weight: float = 1.0):
        assert len(self.rubrics) > 0, "RubricGroup must have at least one rubric"
        self.logger.warning("Adding reward function to the first rubric in the group.")
        self.rubrics[0].add_reward_func(func, weight)

    def score_rollouts(self,
                       prompts: List[List[Dict[str, str]] | str],
                       completions: List[List[Dict[str, str]] | str],
                       answers: List[Any],
                       states: List[Dict[str, Any]],
                       tasks: List[str],
                       max_concurrent: int = 32,
                       **kwargs) -> Dict[str, List[float]]:
        """
        Run all rubrics sequentially and return the aggregated scores.

        Reward functions with the same name are summed up.
        """
        all_scores = {} 
        for rubric in self.rubrics:
            rubric_scores = rubric.score_rollouts(
                prompts, completions, answers, states, tasks,
                max_concurrent=max_concurrent, **kwargs)
            for key, value in rubric_scores.items():
                if key in all_scores:
                    # element-wise sum
                    all_scores[key] = [a + b for a, b in zip(all_scores[key], value)]
                else:
                    all_scores[key] = value
        return all_scores