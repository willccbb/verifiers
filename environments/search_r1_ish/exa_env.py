import os

from datasets import load_dataset
from openai import OpenAI
from exa_py import Exa

import verifiers as vf
from verifiers.rubrics.judge_rubric import JudgeRubric

def load_environment(
    exa_api_key: str,
    judge_api_key_var: str = "OPENAI_API_KEY",
    judge_model: str = "gpt-4.1-mini",
    judge_base_url: str | None = None,
    max_turns=10,
    max_search_results=25,
) -> vf.Environment:

    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
    dataset = dataset.map(lambda _: {"task": "hotpot_qa"})
    eval_dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    eval_dataset = dataset.map(lambda _: {"task": "hotpot_qa"})

    exa = Exa(api_key=exa_api_key)

    def exa_search(query: str, num_results: int = 1) -> list[dict[str, str | None]]:
        """ Searches on Exa 
        Args:
            query (str): Search query
        Returns:
            list[dict]: Search results containing the title, text, author (if any), published date (if any) and url.
        """
        full_res = exa.search_and_contents(query, text=True, type='auto').results
        return [dict(
            title=r.title,
            url=r.url,
            published_date=r.published_date,
            author=r.author,
            text=r.text,
        ) for r in full_res[:min(num_results, max_search_results)]]

    tools = [exa_search]
    parser = vf.ThinkParser()
    vf_env = vf.ToolEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        parser=parser,
        tools=tools,
        max_turns=max_turns,
    )

    judge_client = OpenAI(base_url=judge_base_url, api_key=os.getenv(judge_api_key_var))
    judge_rubric = JudgeRubric(
        judge_client=judge_client, judge_model=judge_model, parser=vf_env.parser
    )
    vf_env.rubric = vf.RubricGroup(rubrics=[judge_rubric, vf_env.rubric])

    return vf_env
