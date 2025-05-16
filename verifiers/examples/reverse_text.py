from datasets import load_dataset
from trl import GRPOConfig

import verifiers as vf


# "text" field = paragraph of wikipedia article
dataset = load_dataset("agentlans/wikipedia-paragraphs")
parser = vf.XMLParser(['think', 'answer'])
system_prompt = f"""Reverse the given text.

Respond in the following format:
{parser.get_format_str()}"""

def lcs_ratio(x: str, y: str) -> float:
    """
    Return the longest common subsequence ratio of x and y.
    """
    from difflib import SequenceMatcher
    return SequenceMatcher(None, x, y).ratio()

def lcs_reward_func(prompts: List[str], completions: List[str]) -> List[float]:
    """
    Return the reward for the completions.
    """
    return [lcs_ratio(prompt, completion) for prompt, parser.parse(co) in zip(prompts, completions)]

vf_env = vf.SingleTurnEnv(
    dataset=dataset,
    system_prompt=sys_prompt,
    parser=parser,
)

reward_funcs = [

]

trainer = vf.GRPOEnvTrainer(
    model=model,
    env=vf_env,
    args=GRPOConfig(...)
    reward_funcs=
)