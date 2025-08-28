from typing import Any

from openai import AsyncOpenAI, OpenAI

from verifiers.parsers.parser import Parser
from verifiers.rubrics.rubric import Rubric
from verifiers.samplers import Sampler
from verifiers.types import Messages, State

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
        parser: Parser | None = None,
        parallelize_scoring: bool = False,
        judge_client: OpenAI | AsyncOpenAI | None = None,
        judge_model: str = "gpt-4.1-nano",
        judge_sampling_args: dict[str, Any] | None = None,
        judge_prompt: str = DEFAULT_JUDGE_PROMPT,
        judge_sampler: Sampler | None = None,
        **kwargs,
    ):
        super().__init__(
            parser=parser, parallelize_scoring=parallelize_scoring, **kwargs
        )

        if judge_sampler is not None:
            self.judge_sampler = judge_sampler
        else:
            from verifiers.samplers import OpenAISampler

            self.judge_sampler = OpenAISampler(client=judge_client, model=judge_model)

        self.judge_client = judge_client if judge_client is not None else AsyncOpenAI()
        self.judge_model = judge_model
        self.judge_prompt = judge_prompt
        self.judge_sampling_args = judge_sampling_args or {}
        self.class_objects = {
            "parser": self.parser,
            "judge": self.judge,
            "judge_client": self.judge_client,
            "judge_model": self.judge_model,
            "judge_prompt": self.judge_prompt,
            "judge_sampling_args": self.judge_sampling_args,
        }

    async def judge(
        self,
        prompt: Messages,
        completion: Messages,
        answer: str,
        state: State,
        **kwargs,
    ) -> str:
        if isinstance(prompt, list):
            last_msg = prompt[-1]
            if isinstance(last_msg, dict) and "content" in last_msg:
                question = str(last_msg["content"])
            else:
                question = ""
        else:
            question = str(prompt)
        response = self.parser.parse_answer(completion)
        judge_prompt = self.judge_prompt.format(
            question=question, answer=answer, response=response
        )
        cached = state.get("judge_response")
        if isinstance(cached, dict) and judge_prompt in cached:
            return cached[judge_prompt]

        judge_args = dict(self.judge_sampling_args or {})
        judge_args = {k: v for k, v in judge_args.items() if v is not None}

        judge_message = await self.judge_sampler.sample(
            messages=[{"role": "user", "content": judge_prompt}], **judge_args
        )
        judge_response = str(judge_message["content"])
        if not isinstance(cached, dict):
            cached = {}
        cached[judge_prompt] = judge_response
        state["judge_response"] = cached
        return judge_response
