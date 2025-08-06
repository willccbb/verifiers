import os
import atexit
import time
import socket
import subprocess
from typing import Any, Dict, List, Optional

import requests
from datasets import Dataset, load_dataset

import verifiers as vf
from verifiers.parsers.parser import Parser


# Track launched background processes to ensure teardown
_LAUNCHED_PROCS: List[subprocess.Popen] = []


def _register_teardown() -> None:
    def _teardown():
        for p in _LAUNCHED_PROCS:
            try:
                if p.poll() is None:
                    p.terminate()
                    try:
                        p.wait(timeout=5)
                    except Exception:
                        p.kill()
            except Exception:
                pass

    atexit.register(_teardown)


def _wait_for_port(host: str, port: int, timeout: int = 30) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            try:
                if sock.connect_ex((host, port)) == 0:
                    return
            except Exception:
                pass
        time.sleep(0.3)
    raise TimeoutError(f"Retriever not reachable at {host}:{port} within {timeout}s")


def _maybe_launch_retriever(
    *,
    retriever_url: str,
    retriever_launch_cmd: Optional[List[str]] = None,
    retriever_ready_timeout: int = 30,
    env: Optional[Dict[str, str]] = None,
) -> None:
    if not retriever_launch_cmd:
        return
    # Best-effort parse host:port from URL
    try:
        from urllib.parse import urlparse

        parsed = urlparse(retriever_url)
        host = parsed.hostname or "127.0.0.1"
        port = int(parsed.port or 8000)
    except Exception:
        host, port = "127.0.0.1", 8000

    # Launch subprocess in background
    proc = subprocess.Popen(
        retriever_launch_cmd,
        env={**os.environ, **(env or {})},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _LAUNCHED_PROCS.append(proc)
    _register_teardown()

    # Wait for readiness by opening TCP port
    _wait_for_port(host, port, timeout=retriever_ready_timeout)


def _make_prefix(question: str) -> str:
    q = question.strip()
    if not q.endswith("?"):
        q += "?"
    return (
        "Answer the given question. "
        "You must conduct reasoning inside <think> and </think> first every time you get new information. "
        "After reasoning, if you find you lack some knowledge, you can call the search tool and it will return the top searched results between <information> and </information>. "
        "You can search as many times as your want. "
        "If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. "
        "For example, <answer> Beijing </answer>. "
        f"Question: {q}\n"
    )


def _format_passages(result: List[Dict[str, Any]]) -> str:
    formatted = []
    for idx, doc_item in enumerate(result):
        content = doc_item.get("document", {}).get("contents", "")
        if not content:
            continue
        parts = content.split("\n")
        title = parts[0].strip('"') if parts else ""
        text = "\n".join(parts[1:]) if len(parts) > 1 else ""
        formatted.append(f"Doc {idx + 1}(Title: {title}) {text}")
    return "\n".join(formatted)


def _search_factory(retriever_url: str, default_topk: int):
    def search(query: str, topk: Optional[int] = None) -> str:
        """Retrieve top passages for a query and return an <information> block.

        Args:
            query: natural language question or subquery
            topk: override for number of passages to return
        Returns:
            A string wrapped in <information>...</information> with formatted passages.
        """
        k = topk or default_topk
        payload = {"queries": [query], "topk": k, "return_scores": True}
        try:
            resp = requests.post(retriever_url, json=payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("result", [])
            first = results[0] if results else []
            formatted = _format_passages(first)
        except Exception:
            formatted = ""
        return f"\n\n<information>{formatted}</information>\n\n"

    return search


# --- Parser specialized for Search-R1 tags ---
class SearchR1Parser(Parser):
    """Parser for Search-R1-style rollouts using <think>, <information>, <answer> tags.

    Works over chat messages (assistant/tool) without requiring a specific chat template.
    """

    def extract_information_blocks(self, completion: List[Dict[str, Any]]) -> List[str]:
        blocks: List[str] = []
        for msg in completion:
            if msg.get("role") not in ("assistant", "tool"):
                continue
            content = str(msg.get("content", ""))
            for m in re.finditer(r"<information>(.*?)</information>", content, re.DOTALL):
                blocks.append(m.group(1).strip())
        return blocks

    def get_last_answer(self, completion: List[Dict[str, Any]]) -> Optional[str]:
        matches: List[str] = []
        for msg in completion:
            if msg.get("role") != "assistant":
                continue
            content = str(msg.get("content", ""))
            for m in re.finditer(r"<answer>(.*?)</answer>", content, re.DOTALL):
                matches.append(m.group(1).strip())
        if not matches:
            return None
        return matches[-1]

    def is_valid_sequence(self, completion: List[Dict[str, Any]]) -> tuple[bool, str]:
        tags = ["think", "search", "information", "answer"]
        for tag in tags:
            opens = sum(len(re.findall(fr"<{tag}>", msg.get("content", ""))) for msg in completion if msg.get("role") in ("assistant", "tool"))
            closes = sum(len(re.findall(fr"</{tag}>", msg.get("content", ""))) for msg in completion if msg.get("role") in ("assistant", "tool"))
            if opens != closes:
                return False, f"Mismatch in {tag} tags: {opens} opening vs {closes} closing tags"

        state = "start"
        for msg in completion:
            if msg.get("role") not in ("assistant", "tool"):
                continue
            content = str(msg.get("content", ""))
            for part in re.split(r"(</?(?:think|search|information|answer)>)", content):
                if not part or not part.strip():
                    continue
                if re.match(r"</?(?:think|search|information|answer)>", part):
                    if part == "<think>" and state in ["start", "information"]:
                        state = "in_think"
                    elif part == "</think>" and state == "in_think":
                        state = "after_think"
                    elif part == "<information>" and state in ["after_think", "information"]:
                        state = "in_information"
                    elif part == "</information>" and state == "in_information":
                        state = "information"
                    elif part == "<answer>" and state in ["after_think", "information"]:
                        state = "in_answer"
                    elif part == "</answer>" and state == "in_answer":
                        state = "end"
                    else:
                        if part in ("<search>", "</search>"):
                            continue
                        return False, f"Unexpected tag {part} in state {state}"
                else:
                    if state in ["in_think", "in_information", "in_answer"]:
                        pass
                    elif state in ["start", "after_think", "information"]:
                        if part.strip():
                            return False, f"Unexpected content between tags (state: {state})"
                    else:
                        return False, f"Unexpected content in state {state}"
        if state != "end":
            return False, f"Incomplete sequence, ended in state {state}"
        return True, "Valid sequence format"


# String-based helpers retained for normalization only
import re
import string


def _normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _em_check(prediction: str, golden_answers: Any) -> int:
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = _normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = _normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def _is_retrieval_correct_via_parser(parser: SearchR1Parser, completion: List[Dict[str, Any]], golden_answers: List[str]) -> bool:
    for block in parser.extract_information_blocks(completion):
        for ga in golden_answers:
            if _normalize_answer(ga) in _normalize_answer(block):
                return True
    return False


def _compute_score_em_parsed(
    parser: SearchR1Parser,
    completion: List[Dict[str, Any]],
    ground_truth: Dict[str, Any],
    structure_format_score: float = 0.0,
    final_format_score: float = 0.0,
    retrieval_score: float = 0.0,
    score: float = 1.0,
) -> float:
    is_valid_format, _ = parser.is_valid_sequence(completion)
    retrieval_correct = False
    if is_valid_format:
        retrieval_correct = _is_retrieval_correct_via_parser(parser, completion, ground_truth["target"])
    answer = parser.get_last_answer(completion)

    if answer is None:
        if is_valid_format:
            if retrieval_correct:
                return structure_format_score + retrieval_score
            return structure_format_score
        return 0.0
    else:
        if _em_check(answer, ground_truth["target"]):
            if is_valid_format:
                return score
            return score - structure_format_score
        else:
            if is_valid_format:
                if retrieval_correct:
                    return structure_format_score + retrieval_score
                return structure_format_score
            return final_format_score


def load_environment(
    *,
    retriever_url: str = "http://127.0.0.1:8000/retrieve",
    retriever_topk: int = 3,
    max_turns: int = 2,
    structure_format_score: float = 0.0,
    final_format_score: float = 0.0,
    retrieval_score: float = 0.0,
    score: float = 1.0,
    data_split: str = "train",
    data_limit: Optional[int] = None,
    # background retriever management
    retriever_launch_cmd: Optional[List[str]] = None,
    retriever_ready_timeout: int = 30,
    retriever_env: Optional[Dict[str, str]] = None,
) -> vf.Environment:
    """Load the Search-R1 environment using ToolEnv and the official dataset + reward.

    If retriever_launch_cmd is provided, a background retriever process is launched and
    torn down automatically at interpreter exit.
    """
    # Optionally provision a local retriever server in the background
    _maybe_launch_retriever(
        retriever_url=retriever_url,
        retriever_launch_cmd=retriever_launch_cmd,
        retriever_ready_timeout=retriever_ready_timeout,
        env=retriever_env,
    )

    dataset = load_dataset("RUC-NLPIR/FlashRAG_datasets", "nq")[data_split]  # type: ignore

    def map_example(ex: Dict[str, Any]) -> Dict[str, Any]:
        prompt = _make_prefix(ex["question"])  # uses exact template
        return {
            "prompt": [{"role": "user", "content": prompt}],
            "info": {
                "ground_truth": {"target": ex["golden_answers"]},
                "data_source": "nq",
            },
        }

    dataset = dataset.map(map_example)
    if data_limit is not None:
        dataset = Dataset.from_dict(dataset[: int(data_limit)])

    search_tool = _search_factory(retriever_url=retriever_url, default_topk=retriever_topk)

    system_prompt = (
        "You are a research assistant with access to the following tools.\n"
        "Use them to answer questions.\n\n"
        "- Think inside <think> ... </think> before each action.\n"
        "- If you need external knowledge, call the search tool. It will return results wrapped between <information> and </information>.\n"
        "- When ready, provide the final answer inside <answer> ... </answer> without extra explanation.\n\n"
        "{tool_descriptions}"
    )

    parser = SearchR1Parser()

    vf_env = vf.ToolEnv(
        dataset=dataset,
        system_prompt=system_prompt,
        tools=[search_tool],
        max_turns=max_turns,
        parser=parser,
    )

    def reward_fn(parser: vf.Parser, completion: List[Dict[str, str]], info: Dict[str, Any], **kwargs) -> float:
        # Operate natively over the chat messages via the parser abstraction
        if not isinstance(parser, SearchR1Parser):  # fallback if overridden externally
            # minimal fallback: extract answer via basic XML pattern from assistant messages
            tmp_parser = SearchR1Parser()
            return _compute_score_em_parsed(
                parser=tmp_parser,
                completion=completion,  # type: ignore[arg-type]
                ground_truth=info["ground_truth"],
                structure_format_score=structure_format_score,
                final_format_score=final_format_score,
                retrieval_score=retrieval_score,
                score=score,
            )
        return _compute_score_em_parsed(
            parser=parser,
            completion=completion,  # type: ignore[arg-type]
            ground_truth=info["ground_truth"],
            structure_format_score=structure_format_score,
            final_format_score=final_format_score,
            retrieval_score=retrieval_score,
            score=score,
        )

    vf_env.rubric = vf.Rubric(parser=vf_env.parser, funcs=[reward_fn])

    return vf_env
