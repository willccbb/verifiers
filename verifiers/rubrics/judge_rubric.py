from typing import Callable

from openai import OpenAI

from verifiers import Parser, Rubric

DEFAULT_JUDGE_PROMPT = """Given a ground truth answer \
and a response, determine if the response is correct.

Question:
```
{question}
```

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
    def __init__(
        self,
        parser: Parser = Parser(),
        judge_client: OpenAI | None = None,
        judge_model: str = "gpt-4.1-nano",
        judge_sampling_args: dict = {},
        judge_prompt: str = DEFAULT_JUDGE_PROMPT,
        judge_score_fn: Callable[[str], float] = lambda x: 1.0
        if "yes" in x.lower()
        else 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.parser = parser
        self.judge_client = judge_client if judge_client is not None else OpenAI()
        self.judge_model = judge_model
        self.judge_prompt = judge_prompt
        self.judge_sampling_args = judge_sampling_args
        self.judge_score_fn = judge_score_fn
        self.add_reward_func(self.judge_reward_func)

    def judge(self, prompt, completion, answer, **kwargs) -> str:
        if isinstance(prompt, list):
            question = prompt[-1]["content"]
        else:
            question = prompt
        response = self.parser.parse_answer(completion)
        prompt = self.judge_prompt.format(
            question=question, answer=answer, response=response
        )
        judge_response = self.judge_client.chat.completions.create(
            model=self.judge_model,
            messages=[{"role": "user", "content": prompt}],
            **self.judge_sampling_args,
        )
        judge_response = str(judge_response.choices[0].message.content)
        return judge_response

    def judge_reward_func(self, prompt, completion, answer, **kwargs) -> float:
        judge_response = self.judge(prompt, completion, answer, **kwargs)
        return self.judge_score_fn(judge_response)
