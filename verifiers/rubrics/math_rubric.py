from verifiers.parsers import XMLParser
from verifiers.rubrics import Rubric


class MathRubric(Rubric):
    def __init__(self, parser: XMLParser | None = None):
        self.parser = parser if parser is not None else XMLParser(fields=["think", "answer"])
        self.reward_funcs = [
            self.exact_answer_reward_func,
            self.int_answer_reward_func,
            self.parser.get_xml_reward_func(),
            self.parser.get_format_reward_func()
        ]
        self.reward_weights = [
            1.0,
            0.5,
            0.25,
            0.25
        ]

    def exact_answer_reward_func(self, completion, answer, **kwargs) -> List[float]:
        """Reward function that checks if the final answer matches the expected answer."""
        responses = [self.parser.get_final_answer(c) for c in completions]
        return [1.0 if str(r) == str(a) else 0.0 for r, a in zip(responses, answer)]

    def int_answer_reward_func(self, completion, answer, **kwargs) -> List[float]:
        """Reward function that checks if the final answer is an integer."""
        responses = [self.parser.get_final_answer(c) for c in completions]
        return [1.0 if str(r).isdigit() else 0.0 for r in responses]