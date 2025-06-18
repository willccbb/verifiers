import os
from openai import OpenAI

from verifiers.parsers import Parser
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
    def __init__(self,
                 judge_client: OpenAI | None = None,
                 judge_model: str = "gpt-4.1-nano",
                 judge_prompt: str = DEFAULT_JUDGE_PROMPT,
                 parser: Parser = Parser(),
                 **kwargs):
        super().__init__(**kwargs)
        self.judge_client = judge_client if judge_client is not None else OpenAI()
        self.judge_model = judge_model
        self.judge_prompt = judge_prompt
        self.parser = parser
        self.add_reward_func(self.judge_reward_func)

    def judge_reward_func(self, prompt, completion, answer, **kwargs) -> float:
        response = self.parser.parse_answer(completion)
        # check which fields are present in judge prompt template
        # get question from answer:
        if isinstance(prompt, list):
            question = prompt[-1]['content']
        else:
            question = prompt
        if isinstance(completion, list):
            response = completion[-1]['content']
        else:
            response = completion
        prompt = self.judge_prompt.format(question=question, answer=answer, response=response)
        judge_response = self.judge_client.chat.completions.create(
            model=self.judge_model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
        )
        judge_response = str(judge_response.choices[0].message.content)
        if 'yes' in judge_response.lower():
            return 1.0
        elif 'no' in judge_response.lower():
            return 0.0
        else:
            # extract float from judge_response
            return float(judge_response)
    


    