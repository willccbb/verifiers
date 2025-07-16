import os
from openai import OpenAI
import verifiers as vf
from verifiers.envs.textarena_env import TextArenaEnv

NOTHINK_WORDLE_SYSTEM_PROMPT = """You are a competitive game player. \
Make sure you read the game instructions carefully, and always follow the required format.

In each turn, give only your guess inside <guess>...</guess> tags."""

vf_env = TextArenaEnv(
    game="Wordle-v0",
    num_train_examples=2000,
    num_eval_examples=2000,
    system_prompt=NOTHINK_WORDLE_SYSTEM_PROMPT,
    parser=vf.XMLParser(fields=["guess"], answer_field="guess"),
)

parser = vf_env.parser

def partial_credit_reward_func(completion, **kwargs) -> float:
    """Reward function that gives partial credit for the correct guess."""
    final_env_response = parser.get_user_messages(completion)[-1]['content'].strip()
    guess, scoring = final_env_response.split("\n")[:2]
    num_greens = scoring.count("G")
    num_yellows = scoring.count("Y")
    return 0.2 * num_greens + 0.1 * num_yellows

vf_env.rubric.add_reward_func(partial_credit_reward_func)

def main(api: str, num_examples: int, rollouts_per_example: int, max_tokens: int, save_dataset: bool = False):
    # collect V3/R1 rollouts from API
    if api == "deepseek":
        base_url = os.getenv("DEEPINFRA_API_URL")
        api_key = os.getenv("DEEPINFRA_API_KEY")
        model_name = "deepseek-ai/DeepSeek-V3-0324" # DeepSeek V3-0324
        client = OpenAI(base_url=base_url, api_key=api_key)
    elif api == "deepinfra":
        base_url = os.getenv("DEEPINFRA_API_URL")
        api_key = os.getenv("DEEPINFRA_API_KEY")
        model_name = "deepseek-ai/DeepSeek-V3-0324-Turbo"
        client = OpenAI(base_url=base_url, api_key=api_key)
    elif api == "openai":
        # just for testing :) not for distillation :)
        api_key = os.getenv("OPENAI_API_KEY")
        model_name = "gpt-4.1-mini"
        client = OpenAI(api_key=api_key)
    else:
        raise ValueError(f"Invalid API: {api}")
    sampling_args = {
        "max_tokens": max_tokens,
    }
    # columns = ['prompt', 'completion', 'answer', 'reward', ...]
    # use deepseek-chat for multiturn rollouts (V3-0324)
    results = vf_env.evaluate(
        client=client,
        model=model_name, 
        sampling_args=sampling_args,
        num_examples=num_examples,
        rollouts_per_example=rollouts_per_example,
        max_concurrent=20
    )

    print('--- Example ---')
    print('Prompt: ', results['prompt'][0])
    print('Completion: ', results['completion'][0])
    print('Answer: ', results['answer'][0])
    for k, v in results.items():
        if "reward" in k:
            print(k, ': ', v[0]) 
    print("--- Rewards ---")
    for k, v in results.items():
        if 'reward' in k:
            print(k, '-', sum(v) / len(v)) 
    if save_dataset:
        dataset_dsv3 = vf_env.make_dataset(results)
        # filter to top half of rows by rewards
        dataset_dsv3 = dataset_dsv3.sort("reward", reverse=True).select(range(len(dataset_dsv3) // 2))
        # save to hub
        dataset_dsv3.push_to_hub("mini-wordle-nothink-100")

if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--api", "-a", type=str, default="openai")
    argparser.add_argument("--num-examples", "-n", type=int, default=20)
    argparser.add_argument("--rollouts-per-example", "-r", type=int, default=1)
    argparser.add_argument("--max-tokens", "-t", type=int, default=20)
    argparser.add_argument("--save-dataset", "-s", action="store_true", default=False)
    args = argparser.parse_args()
    main(args.api, args.num_examples, args.rollouts_per_example, args.max_tokens, args.save_dataset)