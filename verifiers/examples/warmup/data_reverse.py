from datasets import load_dataset, Dataset
import verifiers as vf

dataset = load_dataset('agentlans/wikipedia-paragraphs', split='train').map(lambda x: {'question': x['text'], 'answer': x['text'][::-1]})
parser = vf.XMLParser(['think', 'answer'], answer_field="answer")
system_prompt = f"""Respond in the following format:
{parser.get_format_str()}

Reverse the given text character-by-character."""

def lcs_reward_func(completion, answer, **kwargs) -> list[float]:
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
    rubric=rubric
)

# collect V3/R1 rollouts from API
import os
from openai import OpenAI
base_url = "https://api.deepseek.com"
api_key = os.getenv("DEEPSEEK_API_KEY")
client = OpenAI(base_url=base_url, api_key=api_key)

# columns = ['prompt', 'completion', 'answer', 'reward']
# use deepseek-chat for multiturn rollouts (V3-0324)
dataset_r1 = vf_env.make_api_dataset(client, model="deepseek-chat", num_samples=10) 
# filter to top half of rows by rewards
dataset_r1 = dataset_r1.sort("reward", reverse=True).select(range(len(dataset_r1) // 2))
# # save to hub
dataset_r1.push_to_hub("V3-reverse-wikipedia-paragraphs-test")