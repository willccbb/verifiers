import os

from openai import OpenAI
from datasets import load_dataset

import verifiers as vf

dataset = load_dataset('agentlans/wikipedia-paragraphs', split='train')
dataset = dataset.map(lambda x: {'question': x['text'], 'answer': x['text']})

parser = vf.XMLParser(['think', 'answer'], answer_field="answer")
system_prompt = f"""Respond in the following format:
{parser.get_format_str()}

Summarize the given text in 3 sentences."""

def sentence_reward_func(completion, **kwargs) -> float:
    """
    Count the number of sentences in the completion.
    """
    response = parser.parse_answer(completion) or ''
    return 1.0 if len(response.split('.')) == 3 else 0.0

def lcs_reward_func(completion, answer, **kwargs) -> float:
    """
    LCS ratio of the prompt and the parsed completion.    
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
    sentence_reward_func,
    lcs_reward_func,
    parser.get_format_reward_func(),
], weights=[1.0, 0.2, 0.2])

vf_env = vf.SingleTurnEnv(
    eval_dataset=dataset,
    system_prompt=system_prompt,
    parser=parser,
    rubric=rubric,
    max_concurrent=10
)

def main(api: str, num_samples: int, max_tokens: int):    
    # collect V3/R1 rollouts from API
    if api == "deepseek":
        base_url = "https://api.deepseek.com"
        api_key = os.getenv("DEEPSEEK_API_KEY")
        model_name = "deepseek-chat" # DeepSeek V3-0324
        client = OpenAI(base_url=base_url, api_key=api_key)
    elif api == "openai":
        # just for testing :) not for distillation :)
        api_key = os.getenv("OPENAI_API_KEY")
        model_name = "gpt-4.1" 
        client = OpenAI(api_key=api_key)
    else:
        raise ValueError(f"Invalid API: {api}")
    sampling_args = {
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    # columns = ['prompt', 'completion', 'answer', 'reward']
    # use deepseek-chat for multiturn rollouts (V3-0324)
    results = vf_env.evaluate(
        client=client, model=model_name, 
        sampling_args=sampling_args, num_samples=num_samples) 
    print(results)
    #dataset_dsv3 = vf_env.make_dataset(results)
    # filter to top half of rows by rewards
    #dataset_dsv3 = dataset_dsv3.sort("reward", reverse=True).select(range(len(dataset_dsv3) // 2))
    # save to hub
    #dataset_dsv3.push_to_hub("V3-wiki-paragraphs-test")
    
if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--api", "-a", type=str, default="openai")
    argparser.add_argument("--num-samples", "-n", type=int, default=10)
    argparser.add_argument("--max-tokens", "-t", type=int, default=4096)
    args = argparser.parse_args()
    main(args.api, args.num_samples, args.max_tokens)