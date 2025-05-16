from datasets import load_dataset
import verifiers as vf

dataset = load_dataset('agentlans/wikipedia-paragraphs').map(lambda x: {'question': x['text'], 'answer': x['text'][::-1]})
parser = vf.XMLParser(['think', 'answer'], answer_field="answer")
system_prompt = f"""Respond in the following format:
{parser.get_format_str()}

Reverse the given text character-by-character."""

def lcs_reward_func(completions, answer, **kwargs) -> list[float]:
    """
    LCS ratio of the reversed prompt and the parsed completion.
    """
    def lcs_ratio(x: str, y: str) -> float:
        """
        Return the longest common subsequence ratio of x and y.
        """
        from difflib import SequenceMatcher
        return SequenceMatcher(None, x, y).ratio()
    responses = [parser.parse_answer(c) or '' for c in completions]
    return [lcs_ratio(r, a) for r, a in zip(responses, answer)]

rubric = vf.Rubric(funcs=[
	lcs_reward_func,
	parser.get_format_reward_func(),
], weights=[1.0, 0.2])

vf_env = vf.SingleTurnEnv(
    eval_dataset=dataset['train'].select(range(20)),
    system_prompt=system_prompt,
    parser=parser,
    rubric=rubric
)

import os
from openai import OpenAI
base_url = os.getenv("DEEPSEEK_API_URL")
api_key = os.getenv("DEEPSEEK_API_KEY")
client = OpenAI(base_url=base_url, api_key=api_key)
vf_env.eval_api(client, "deepseek-reasoner", max_concurrent=20, sampling_args={"temperature": 0.6})
