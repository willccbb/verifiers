#!/usr/bin/env python3
"""Minimal smoke test for the FutureX-Past environment without network calls.

Loads a small eval split, fabricates completions using the ground-truth
answers, and scores them locally via the rubric. Also exercises MCQ
label↔text matching if options are present in the example info.
"""

import asyncio
import os
from typing import Any, Dict, List

import importlib.util


def _load_env_module():
    module_path = os.path.join(os.path.dirname(__file__), "..", "futurex_past.py")
    module_path = os.path.abspath(module_path)
    spec = importlib.util.spec_from_file_location("futurex_past", module_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def make_assistant_completion(answer_text: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "assistant",
            "content": f"<think></think><answer>{answer_text}</answer>",
        }
    ]


async def score_example(env, ex_idx: int, ds_item: Dict[str, Any]) -> float:
    prompt = ds_item["prompt"]
    gold = ds_item.get("answer", "") or ""
    info = ds_item.get("info", {}) or {}
    completion = make_assistant_completion(gold)
    score = await env.rubric.score_rollout(
        prompt=prompt, completion=completion, answer=gold, state={"info": info}
    )
    return float(score.reward)


async def score_mcq_bidir(env, ex_idx: int, ds_item: Dict[str, Any]) -> float:
    """If options exist, try label↔text matching both ways."""
    prompt = ds_item["prompt"]
    gold = (ds_item.get("answer", "") or "").strip()
    info = ds_item.get("info", {}) or {}
    opts = info.get("options") or []
    if not opts:
        return -1.0

    # case 1: gold is label, predict text
    reward1 = -1.0
    if gold:
        # map label to index (A->0, B->1, 1->0, etc.)
        gl = gold.upper().strip()
        idx = None
        if gl and gl[0].isalpha():
            idx = ord(gl[0]) - ord("A")
        elif gl.isdigit():
            idx = int(gl) - 1
        if idx is not None and 0 <= idx < len(opts):
            pred_text = str(opts[idx])
            completion = make_assistant_completion(pred_text)
            score = await env.rubric.score_rollout(
                prompt=prompt, completion=completion, answer=gold, state={"info": info}
            )
            reward1 = float(score.reward)

    # case 2: gold is text, predict label 'A'
    reward2 = -1.0
    try:
        pred_label = "A"
        completion = make_assistant_completion(pred_label)
        score = await env.rubric.score_rollout(
            prompt=prompt, completion=completion, answer=gold, state={"info": info}
        )
        reward2 = float(score.reward)
    except Exception:
        pass

    return max(reward1, reward2)


def main():
    data_dir = os.environ.get(
        "FUTUREX_PAST_DATA_DIR", "../futurex-ai/Futurex-Past/data"
    )
    mod = _load_env_module()
    env = mod.load_environment(
        data_dir=data_dir, use_think=True, num_eval_examples=5, num_train_examples=0
    )
    ds = env.get_eval_dataset()
    n = min(5, len(ds))
    print(f"Loaded eval dataset: {len(ds)} items; scoring first {n}...")

    # Score exact matches
    async def run_exact():
        out = []
        for i in range(n):
            out.append(await score_example(env, i, ds[i]))
        return out

    rewards = asyncio.run(run_exact())
    print("Exact-match rewards:", rewards)

    # Try MCQ matching if possible
    async def run_mcq():
        for i in range(n):
            info = ds[i].get("info", {}) or {}
            if info.get("options"):
                return await score_mcq_bidir(env, i, ds[i])
        return -1.0

    mcq_reward = asyncio.run(run_mcq())
    if mcq_reward >= 0:
        print("MCQ bidirectional reward (sample):", mcq_reward)
    else:
        print("No MCQ example found in first batch to test bidirectional matching.")


if __name__ == "__main__":
    main()
