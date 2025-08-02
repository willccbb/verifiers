import os

from datasets import load_dataset
from openai import OpenAI

import verifiers as vf


def load_environment(
    num_train_examples: int = 1000,
    num_eval_examples: int = 100,
    judge_model: str = "gpt-4.1-mini",
    judge_base_url: str = "https://api.openai.com/v1",
    judge_api_key_var: str = "OPENAI_API_KEY",
    use_think: bool = True,
    **kwargs,
) -> vf.Environment:
    dataset = load_dataset("lucadiliello/hotpotqa", split="train").map(
        lambda x: {
            "question": x["question"],
            "answer": x["answers"][0],
        }
    )
    dataset = dataset.select(range(num_train_examples))  # type: ignore
    eval_dataset = load_dataset("lucadiliello/hotpotqa", split="validation").map(
        lambda x: {
            "question": x["question"],
            "answer": x["answers"][0],
        }
    )
    eval_dataset = eval_dataset.select(range(num_eval_examples))  # type: ignore

    if use_think:
        parser = vf.ThinkParser()
    else:
        parser = vf.Parser()
    judge_client = OpenAI(
        base_url=judge_base_url, api_key=os.getenv(judge_api_key_var, "EMPTY")
    )
    rubric = vf.JudgeRubric(
        parser=parser,
        judge_model=judge_model,
        judge_client=judge_client,
    )

    def correct_answer(prompt, completion, answer, state) -> float:
        judge_response = rubric.judge(prompt, completion, answer, state)
        return 1.0 if "yes" in judge_response.lower() else 0.0

    rubric.add_reward_func(correct_answer)
    return vf.SingleTurnEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        parser=parser,
        rubric=rubric,
    )
