from openai import OpenAI

from verifiers.rubrics.rubric import Rubric


class JudgeRubric(Rubric):
    def __init__(self,
                 judge_client: OpenAI,
                 judge_model: str,
                 judge_prompt: str,
                 **kwargs):
        super().__init__(**kwargs)
        self.add_reward_func(self.judge_reward_func)

    def judge_reward_func(self, completion, **kwargs) -> float:
        return 1.0 if completion['judge'] == 'correct' else 0.0