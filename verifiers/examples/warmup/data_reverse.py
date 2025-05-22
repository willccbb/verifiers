from datasets import Dataset, load_dataset

dataset = load_dataset('agentlans/wikipedia-paragraphs', split='train')
dataset = dataset.map(lambda x: {'question': x['text'], 'answer': x['text'][::-1]})

import verifiers as vf
parser = vf.XMLParser(['think', 'answer'], answer_field="answer")
system_prompt = f"""Respond in the following format:
{parser.get_format_str()}

Reverse the given text character-by-character."""

def lcs_reward_func(completion, answer, **kwargs) -> float:
    """
    LCS ratio of the reversed prompt and the parsed completion.    
    """
    def lcs_ratio(x: str, y: str) -> float:
        """
        Return the longest common subsequence ratio of x and y.
        """
        from difflib import SequenceMatcher
        return SequenceMatcher(None, x, y).ratio()
    response = parser.parse_answer(completion) or ''
    return lcs_ratio(response, answer)

rubric = vf.Rubric(funcs=[
	lcs_reward_func,
	parser.get_format_reward_func(),
], weights=[1.0, 0.2])


vf_env = vf.SingleTurnEnv(
    eval_dataset=dataset,
    system_prompt=system_prompt,
    parser=parser,
    rubric=rubric,
    max_concurrent=10
)
print(len(vf_env.get_eval_dataset())) # type: ignore
# collect V3/R1 rollouts from API
import os
from openai import OpenAI

api = "deepseek"
if api == "openai":
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = "gpt-4.1-mini" 
    client = OpenAI(api_key=api_key)
elif api == "deepseek":
    base_url = "https://api.deepseek.com"
    api_key = os.getenv("DEEPSEEK_API_KEY")
    model_name = "deepseek-chat" # DeepSeek V3-0324
    client = OpenAI(base_url=base_url, api_key=api_key)
else:
    raise ValueError(f"Invalid API: {api}")

# columns = ['prompt', 'completion', 'answer', 'reward']
# use deepseek-chat for multiturn rollouts (V3-0324)
results = vf_env.evaluate(client=client, model=model_name, num_samples=10) 
dataset_dsv3 = vf_env.make_dataset(results)
# filter to top half of rows by rewards
dataset_dsv3 = dataset_dsv3.sort("reward", reverse=True).select(range(len(dataset_dsv3) // 2))
# save to hub
dataset_dsv3.push_to_hub("V3-reverse-wiki-paragraphs-test")