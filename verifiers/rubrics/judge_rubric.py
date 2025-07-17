from openai import OpenAI

from verifiers.parsers.parser import Parser
from verifiers.rubrics.rubric import Rubric

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
        parallelize_scoring: bool = False,
        judge_client: OpenAI | None = None,
        judge_model: str = "gpt-4.1-nano",
        judge_sampling_args: dict = {},
        judge_prompt: str = DEFAULT_JUDGE_PROMPT,
        **kwargs,
    ):
        super().__init__(
            parser=parser, parallelize_scoring=parallelize_scoring, **kwargs
        )
        self.parser = parser
        self.judge_client = judge_client if judge_client is not None else OpenAI()
        self.judge_model = judge_model
        self.judge_prompt = judge_prompt
        self.judge_sampling_args = judge_sampling_args

    def judge(self, prompt, completion, answer, state, **kwargs) -> str:
        if "judge_response" in state:
            return state["judge_response"]
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
        state["judge_response"] = judge_response
        return judge_response
