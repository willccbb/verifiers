#!/usr/bin/env python3
"""Run FutureX-Past eval in chunks by level and aggregate results.

Defaults:
- model: gpt-4.1-mini (OpenAI)
- levels: 1,2,3,4
- n: -1 per chunk (full chunk)
- r: 1 rollout per example
- max_concurrent: 8
- max_tokens: 512
- temperature: 0.7

Produces:
- Per-level HTML report files under environments/futurex_past/reports/
- One combined HTML report that aggregates all chunks.
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI

import verifiers as vf
from verifiers.types import GenerateOutputs
from verifiers.utils.report_utils import (
    ReportMeta,
    get_env_version,
    write_html_report,
)


def cat_outputs(acc: GenerateOutputs | None, cur: GenerateOutputs) -> GenerateOutputs:
    if acc is None:
        return cur
    # Concatenate lists; union metric keys
    merged_metrics: Dict[str, List[float]] = {k: list(v) for k, v in acc.metrics.items()}
    for k, v in cur.metrics.items():
        merged_metrics.setdefault(k, [])
        merged_metrics[k].extend(v)
    return GenerateOutputs(
        prompt=acc.prompt + cur.prompt,
        completion=acc.completion + cur.completion,
        answer=acc.answer + cur.answer,
        state=acc.state + cur.state,
        info=acc.info + cur.info,
        task=acc.task + cur.task,
        reward=acc.reward + cur.reward,
        metrics=merged_metrics,
    )


def run_chunk(
    data_dir: str,
    level: int,
    model: str,
    api_base_url: str,
    api_key_var: str,
    max_concurrent: int,
    max_tokens: int,
    temperature: float,
) -> tuple[GenerateOutputs, ReportMeta, Path]:
    env_args = {"data_dir": data_dir, "num_eval_examples": -1, "filter_levels": [level]}
    vf_env = vf.load_environment("futurex-past", **env_args)
    client = OpenAI(api_key=os.getenv(api_key_var, "EMPTY"), base_url=api_base_url)
    results = vf_env.evaluate(
        client=client,
        model=model,
        sampling_args={"max_tokens": max_tokens, "temperature": temperature},
        num_examples=-1,
        rollouts_per_example=1,
        max_concurrent=max_concurrent,
    )
    env_root = Path(__file__).resolve().parents[1]
    report_dir = env_root / "reports"
    meta = ReportMeta(
        env_id="futurex-past",
        env_version=get_env_version("futurex-past"),
        model=model,
        num_examples=len(results.reward),
        rollouts_per_example=1,
        api_base_url=api_base_url,
        sampling_args={"max_tokens": max_tokens, "temperature": temperature},
        env_args=env_args,
    )
    out_path = write_html_report(report_dir, meta, results)
    return results, meta, out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../futurex-ai/Futurex-Past/data")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--api-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--api-key-var", default="OPENAI_API_KEY")
    parser.add_argument("--levels", default="1,2,3,4", help="Comma-separated levels to run")
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    levels = [int(x.strip()) for x in args.levels.split(",") if x.strip()]

    # Run each level and write individual reports
    combined: GenerateOutputs | None = None
    per_level_paths: list[Path] = []
    total_n = 0
    for lv in levels:
        print(f"Running level {lv}...")
        res, meta, path = run_chunk(
            data_dir=args.data_dir,
            level=lv,
            model=args.model,
            api_base_url=args.api_base_url,
            api_key_var=args.api_key_var,
            max_concurrent=args.max_concurrent,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        combined = cat_outputs(combined, res) if combined is not None else res
        per_level_paths.append(path)
        total_n += len(res.reward)
        print(f"Level {lv} done: {len(res.reward)} examples → {path.name}")

    # Write aggregate report
    env_root = Path(__file__).resolve().parents[1]
    report_dir = env_root / "reports"
    agg_meta = ReportMeta(
        env_id="futurex-past",
        env_version=get_env_version("futurex-past"),
        model=args.model,
        num_examples=total_n,
        rollouts_per_example=1,
        api_base_url=args.api_base_url,
        sampling_args={"max_tokens": args.max_tokens, "temperature": args.temperature},
        env_args={
            "data_dir": args.data_dir,
            "levels": levels,
            "note": "by-level aggregate",
        },
    )
    agg_path = write_html_report(report_dir, agg_meta, combined)  # type: ignore[arg-type]
    print("Aggregate report:", agg_path)
    print("Per-level reports:")
    for p in per_level_paths:
        print(" -", p)


if __name__ == "__main__":
    main()

