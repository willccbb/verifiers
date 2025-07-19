import argparse
import importlib
import os
import sys
from pathlib import Path

import numpy as np
from openai import OpenAI

import verifiers as vf

current_dir = Path.cwd()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dir-name", "-d", type=str, default="local")
    parser.add_argument("--env", "-e", type=str, default="gsm8k")
    parser.add_argument("--model", "-m", type=str, default="gpt-4.1-mini")
    parser.add_argument("--api-key-var", "-k", type=str, default="OPENAI_API_KEY")
    parser.add_argument(
        "--api-base-url", "-b", type=str, default="https://api.openai.com/v1"
    )
    parser.add_argument("--num-examples", "-n", type=int, default=5)
    parser.add_argument("--rollouts-per-example", "-r", type=int, default=3)
    parser.add_argument("--max-concurrent-requests", "-c", type=int, default=32)
    parser.add_argument("--max-tokens", "-t", type=int, default=1024)
    parser.add_argument("--temperature", "-T", type=float, default=0.7)
    parser.add_argument("--save-dataset", "-s", default=False, action="store_true")
    parser.add_argument("--save-path", "-p", type=str, default="results.jsonl")
    parser.add_argument("--save-to-hf-hub", "-H", default=False, action="store_true")
    parser.add_argument("--hf-hub-dataset-name", "-D", type=str, default="")
    args = parser.parse_args()
    try:
        ENDPOINTS = importlib.import_module(
            f"{args.local_dir_name}.endpoints"
        ).ENDPOINTS
    except ImportError:
        print(
            f"No local endpoint registry found at {args.local_dir_name}/endpoints.py. \
Please run `uv run vf-local` to create a local workspace, \
or specify the model name (-m), API host base URL (-b), and API key variable name (-k)."
        )
        ENDPOINTS = {}

    if args.model in ENDPOINTS:
        args.api_key_var = ENDPOINTS[args.model]["key"]
        args.api_base_url = ENDPOINTS[args.model]["url"]
        args.model = ENDPOINTS[args.model]["model"]

    client = OpenAI(
        api_key=os.getenv(args.api_key_var, "EMPTY"), base_url=args.api_base_url
    )
    vf_env = vf.load_environment(env_id=args.env, env_args={})
    sampling_args = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    results = vf_env.evaluate(
        client=client,
        model=args.model,
        sampling_args=sampling_args,
        num_examples=args.num_examples,
        rollouts_per_example=args.rollouts_per_example,
        max_concurrent_requests=args.max_concurrent_requests,
    )
    rewards = {k: v for k, v in results.items() if "reward" in k}
    print("--- Evaluation ---")
    print(f"Environment: {args.env}")
    print(f"Model: {args.model}")
    print(f"Provider: {args.api_base_url}")
    print(f"Examples: {args.num_examples}")
    print(f"Rollouts per example: {args.rollouts_per_example}")

    print("--- Example ---")
    vf.print_prompt_completions_sample(
        results["prompt"], results["completion"], rewards, step=0
    )
    print("--- All ---")
    print("Rewards:")
    for k, v in results.items():
        if "reward" in k:
            print(f"{k}: avg - {sum(v) / len(v):.3f}, std - {np.std(v):.3f}")
            n = args.num_examples
            r = args.rollouts_per_example
            for i in range(r):
                # rounded to 3 decimal places
                trials = [round(v[(i * r) + j], 3) for j in range(n)]
                out = f"r{i + 1}: {trials}"
                print(out)
    if args.save_dataset:
        dataset = vf_env.make_dataset(results)
        dataset.save_to_disk(args.save_path)
        print(f"Saved dataset to {args.save_path}")
    if args.save_to_hf_hub:
        if args.hf_hub_dataset_name == "":
            dataset_name = f"{args.env}--{args.model}--results"
        else:
            dataset_name = args.hf_hub_dataset_name
        vf_env.make_dataset(results, push_to_hub=True, hub_name=dataset_name)


if __name__ == "__main__":
    main()
