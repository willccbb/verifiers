from typing import List

from verifiers import RewardFunc, Parser
from verifiers.parsers import XMLParser
from verifiers.rubrics import Rubric


class MathRubric(Rubric):
    def __init__(self,
                 funcs: List[RewardFunc] = [],
                 weights: List[float] = [],
                 parser: Parser = XMLParser(fields=["think", "answer"])):
        super().__init__(funcs=funcs, weights=weights, parser=parser)
        self.add_reward_func(self.correct_answer_reward_func)
        self.add_reward_func(self.parser.get_format_reward_func(), weight=0.2)

    def correct_answer_reward_func(self, completion, answer, **kwargs) -> float:
        """Reward function that checks if the final answer matches the expected answer."""
        try:
            from math_verify import parse, verify # type: ignore
            response = self.parser.parse_answer(completion)
            return 1.0 if verify(parse(answer), parse(response)) else 0.0
        except Exception:
            return 0.0