from datasets import load_dataset, Dataset
import verifiers as vf

dataset = load_dataset('agentlans/wikipedia-paragraphs', split='train').map(lambda x: {'question': x['text'], 'answer': x['text'][::-1]})
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

N = len(dataset) # last 2000 rows
vf_env = vf.SingleTurnEnv(
    eval_dataset=dataset.select(range(N - 2000, N)),
    system_prompt=system_prompt,
    parser=parser,
    rubric=rubric
)

# collect R1 rollouts from API
import os
from openai import OpenAI
base_url = os.getenv("DEEPSEEK_API_URL")
api_key = os.getenv("DEEPSEEK_API_KEY")
client = OpenAI(base_url=base_url, api_key=api_key)
results = vf_env.eval_api(client, "deepseek-reasoner", max_concurrent=32, sampling_args={"temperature": 0.6})

print(results['rewards_avg'])

# make dataset from results
# cols: prompt, completions, answer, rewards
def flatten_rewards(rewards: dict) -> list[float]:
    """
    flatten dict of lists into a single list, summing elementwise.
    """
    return [sum(r) for r in zip(*rewards.values())]

dataset = Dataset.from_dict({
    "prompt": results['prompt'],
    "completion": results['completion'],
    "answer": results['answer'],
    "reward": flatten_rewards(results['rewards']),
})

# filter to top half of rows by rewards
dataset = dataset.sort("rewards", reverse=True).select(range(len(dataset) // 2))
print(dataset[0])

# save to hub
dataset.push_to_hub("R1-reverse-wikipedia-paragraphs-v1-1000")