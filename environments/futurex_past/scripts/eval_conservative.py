#!/usr/bin/env python3
"""Run full FutureX-Past eval with a conservative profile.

Defaults:
- model: gpt-4.1-mini (OpenAI)
- n: -1 (full eval split)
- r: 1 rollout per example
- max_concurrent: 4
- max_tokens: 256
- temperature: 0.2

Requires OPENAI_API_KEY in the environment.
"""

import argparse
import os
from pathlib import Path

from openai import OpenAI

import verifiers as vf
from verifiers.utils.report_utils import (
    ReportMeta,
    get_env_version,
    write_html_report,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="../futurex-ai/Futurex-Past/data",
        help="Path to FutureX-Past parquet files",
    )
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--api-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--api-key-var", default="OPENAI_API_KEY")
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    # Load environment
    env_args = {"data_dir": args.data_dir, "num_eval_examples": -1}
    vf_env = vf.load_environment("futurex-past", **env_args)

    client = OpenAI(api_key=os.getenv(args.api_key_var, "EMPTY"), base_url=args.api_base_url)

    results = vf_env.evaluate(
        client=client,
        model=args.model,
        sampling_args={"max_tokens": args.max_tokens, "temperature": args.temperature},
        num_examples=-1,
        rollouts_per_example=1,
        max_concurrent=args.max_concurrent,
    )

    # Write HTML report next to the environment
    env_root = Path(__file__).resolve().parents[1]
    report_dir = env_root / "reports"
    meta = ReportMeta(
        env_id="futurex-past",
        env_version=get_env_version("futurex-past"),
        model=args.model,
        num_examples=len(results.reward),
        rollouts_per_example=1,
        api_base_url=args.api_base_url,
        sampling_args={"max_tokens": args.max_tokens, "temperature": args.temperature},
        env_args=env_args,
    )
    out_path = write_html_report(report_dir, meta, results)
    print(f"Wrote report: {out_path}")


if __name__ == "__main__":
    main()

