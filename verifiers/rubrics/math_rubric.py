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

