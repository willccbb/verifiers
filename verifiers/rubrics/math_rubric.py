from verifiers.parsers import XMLParser
from verifiers.rubrics import Rubric


class MathRubric(Rubric):
    def __init__(self):
        self.parser = XMLParser(fields=["reasoning", "answer"])
        self.reward_funcs = [
            self.exact_answer_reward_func,
            self.int_answer_reward_func,
            self.parser.get_xml_reward_func(),
            self.parser.get_format_reward_func()
        ]

