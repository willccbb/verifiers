from typing import List

from verifiers.parsers.parser import Parser
from verifiers.parsers.xml_parser import XMLParser
from verifiers.rubrics.rubric import Rubric
from verifiers.types import RewardFunc


class MathRubric(Rubric):
    def __init__(
        self,
        funcs: List[RewardFunc] = [],
        weights: List[float] = [],
        parser: Parser = XMLParser(fields=["think", "answer"]),
    ):
        super().__init__(funcs=funcs, weights=weights, parser=parser)
        self.add_reward_func(self.correct_answer_reward_func)
        self.add_reward_func(self.parser.get_format_reward_func(), weight=0.2)

    def correct_answer_reward_func(self, completion, answer, **kwargs) -> float:
        """Reward function that checks if the final answer matches the expected answer."""
        try:
            from verifiers.rubrics.utils.math_utils import (
                grade_answer_mathd,
                grade_answer_sympy,
            )

            response = self.parser.parse_answer(completion) or ""
            return (
                1.0
                if grade_answer_mathd(response, answer)
                or grade_answer_sympy(response, answer)
                else 0.0
            )
        except Exception as e:
            self.logger.error("Please install math_verify to use this reward function.")
            raise e
