from openai import OpenAI

from verifiers.parsers import Parser
from verifiers.rubrics.rubric import Rubric

DEFAULT_JUDGE_PROMPT = """Given a ground truth answer \
and a response, determine if the response is correct.

Ground truth answer:
```
{answer}
```

Response:
```
{response}
```

Respond either "yes" or "no" only."""

class JudgeRubric(Rubric):
    def __init__(self,
                 judge_client: OpenAI = OpenAI(),
                 judge_model: str = "gpt-4.1-mini",
                 judge_prompt: str = DEFAULT_JUDGE_PROMPT,
                 parser: Parser = Parser(),
                 **kwargs):
        super().__init__(**kwargs)
        self.judge_client = judge_client
        self.judge_model = judge_model
        self.judge_prompt = judge_prompt
        self.parser = parser
        self.add_reward_func(self.judge_reward_func)

    def judge_reward_func(self, completion, answer, **kwargs) -> float:
        response = self.parser.parse_answer(completion)
        # check which fields are present in judge prompt template
        if '{answer}' in self.judge_prompt:
            prompt = self.judge_prompt.format(answer=answer, response=response)
        else:
            prompt = self.judge_prompt.format(response=response)
        judge_response = self.judge_client.chat.completions.create(
            model=self.judge_model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
        )
        judge_response = str(judge_response.choices[0].message.content)
        return 1.0 if 'yes' in judge_response.lower() else 0.0