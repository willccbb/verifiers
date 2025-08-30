import os
from typing import Iterable

from datasets import load_dataset

import verifiers as vf


DEFAULT_SYSTEM_PROMPT = (
    "You are a precise QA assistant. Think step-by-step inside <think>...</think>, "
    "then provide the final answer inside <answer>...</answer>. If options are provided, "
    "output only the correct option text or label in <answer>. Do not include explanations inside <answer>."
)


def _normalize(text: str, lowercase: bool, strip_ws: bool) -> str:
    if text is None:
        return ""
    out = text
    if strip_ws:
        out = out.strip()
    if lowercase:
        out = out.lower()
    return out


def _safe_len(x) -> int:
    try:
        return len(x) if x is not None else 0
    except Exception:
        return 0


def _is_mcq(options) -> bool:
    return isinstance(options, Iterable) and not isinstance(options, (str, bytes)) and _safe_len(options) > 0


def load_environment(
    data_dir: str | None = None,
    use_think: bool = True,
    num_train_examples: int = -1,
    num_eval_examples: int = -1,
    filter_levels: list[int] | None = None,
    mcq_only: bool | None = None,
    lowercase_compare: bool = False,
    strip_whitespace_compare: bool = True,
    system_prompt: str | None = None,
):
    """
    FutureX-Past single-turn QA environment (scaffold).

    Args:
        data_dir: Directory containing Parquet shard(s) for FutureX-Past (e.g., '../futurex-ai/Futurex-Past/data').
        use_think: If True, require <think> + <answer> format via XMLParser; else answer-only.
        num_train_examples: Limit training set size (-1 for all).
        num_eval_examples: Limit eval set size (-1 for all). If -1 and train limited, eval is the remainder; if both -1, eval mirrors train.
        filter_levels: If provided, keep only rows where `level` in this list.
        mcq_only: If True, keep only rows with non-empty `options`; if False, keep only rows with empty/absent `options`; if None, keep all.
        lowercase_compare: Normalize both prediction and answer to lowercase for exact match.
        strip_whitespace_compare: Strip surrounding whitespace before comparison.
        system_prompt: Optional override for the system prompt.
    """

    if data_dir is None:
        raise ValueError(
            "data_dir must be provided, e.g., '../futurex-ai/Futurex-Past/data'"
        )

    # Load local parquet(s)
    data_glob = os.path.join(data_dir, "*.parquet")
    dataset = load_dataset("parquet", data_files={"train": data_glob})["train"]  # type: ignore

    # Optional filtering
    if filter_levels:
        allowed = set(filter_levels)
        dataset = dataset.filter(lambda x: int(x.get("level", -1)) in allowed)  # type: ignore
    if mcq_only is True:
        dataset = dataset.filter(lambda x: _is_mcq(x.get("options")))  # type: ignore
    elif mcq_only is False:
        dataset = dataset.filter(lambda x: not _is_mcq(x.get("options")))  # type: ignore

    # Map to common columns used by SingleTurnEnv
    def to_qa(row):
        question = row.get("question") or row.get("prompt") or ""
        answer = row.get("answer") or ""
        info = {}
        opts = row.get("options")
        if _is_mcq(opts):
            info["options"] = list(opts)
        if row.get("level") is not None:
            info["level"] = int(row.get("level"))
        return {"question": question, "answer": answer, "info": info}

    # Preserve original columns so Environment.format_dataset retains 'answer' and 'info'.
    dataset = dataset.map(to_qa)  # type: ignore

    # Split into train/eval deterministically
    ds = dataset.shuffle(seed=42)  # type: ignore
    if num_train_examples is not None and num_train_examples >= 0:
        train_dataset = ds.select(range(num_train_examples)) if num_train_examples >= 0 else ds  # type: ignore
    else:
        train_dataset = ds

    if num_eval_examples is not None and num_eval_examples >= 0:
        if num_train_examples is not None and num_train_examples >= 0:
            start = num_train_examples
        else:
            start = 0
        eval_end = start + num_eval_examples if num_eval_examples >= 0 else None
        eval_dataset = ds.select(range(start, eval_end))  # type: ignore
    else:
        # Mirror train when not specified
        eval_dataset = train_dataset

    # Parser and system prompt
    if use_think:
        parser = vf.XMLParser(fields=["think", "answer"], answer_field="answer")
    else:
        parser = vf.XMLParser(fields=["answer"], answer_field="answer")
    sys_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    # Helper: map common MCQ labels to indices
    def _label_to_index(label: str, num_options: int) -> int | None:
        if not label:
            return None
        s = label.strip().upper()
        # trim wrapping like (A), A., A), etc.
        if len(s) > 2:
            # remove common decorations
            for ch in ["(", ")", ".", ":", ";", "]", "["]:
                s = s.replace(ch, "")
            s = s.replace("OPTION ", "").replace("CHOICE ", "").strip()
        # alpha label
        if len(s) >= 1 and "A" <= s[0] <= "Z":
            idx = ord(s[0]) - ord("A")
            return idx if 0 <= idx < num_options else None
        # numeric label (1-indexed)
        if s.isdigit():
            val = int(s)
            if 1 <= val <= num_options:
                return val - 1
        return None

    # Reward: exact match with optional normalization and MCQ label support
    def exact_match_reward_func(parser, completion, answer, state=None, info=None, **kwargs) -> float:
        pred = parser.parse_answer(completion) or ""
        pred_n = _normalize(pred, lowercase_compare, strip_whitespace_compare)
        ans_n = _normalize(answer, lowercase_compare, strip_whitespace_compare)

        if pred_n == ans_n:
            return 1.0

        # If options are available in info/state, accept label-vs-text matches
        opts = None
        if info and isinstance(info, dict):
            opts = info.get("options")
        if not opts and state and isinstance(state, dict) and isinstance(state.get("info"), dict):
            opts = state["info"].get("options")
        if _is_mcq(opts):
            try:
                # normalize options for comparison
                norm_opts = [
                    _normalize(str(o), lowercase_compare, strip_whitespace_compare)
                    for o in opts
                ]
                # case 1: pred is a label, answer is option text
                li = _label_to_index(pred, len(norm_opts))
                if li is not None and 0 <= li < len(norm_opts):
                    if norm_opts[li] == ans_n:
                        return 1.0
                # case 2: answer is a label, pred is option text
                li = _label_to_index(answer, len(norm_opts))
                if li is not None and 0 <= li < len(norm_opts):
                    if norm_opts[li] == pred_n:
                        return 1.0
            except Exception:
                pass
        return 0.0

    rubric = vf.Rubric(parser=parser)
    rubric.add_reward_func(exact_match_reward_func, weight=1.0)
    rubric.add_reward_func(parser.get_format_reward_func(), weight=0.2)

    vf_env = vf.SingleTurnEnv(
        dataset=train_dataset,  # type: ignore
        eval_dataset=eval_dataset,  # type: ignore
        system_prompt=sys_prompt,
        parser=parser,
        rubric=rubric,
    )
    return vf_env
