import os
import json
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Any, Optional
from openai import OpenAI
from datasets import Dataset

import verifiers as vf
from verifiers.parsers.parser import Parser


def xor_bytes(data: bytes, key: bytes) -> bytes:
    """XOR the data with the key, repeating the key as needed."""
    key_len = len(key)
    return bytes(data[i] ^ key[i % key_len] for i in range(len(data)))


def download_file(url: str, local_path: str) -> bool:
    """Download a file from URL to local path."""
    try:
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, local_path)
        return True
    except urllib.error.URLError as e:
        print(f"Error downloading {url}: {e}")
        return False


def descramble_file(scrambled_path: str, output_path: str) -> bool:
    """Descramble a .scr file to JSON."""
    scramble_key = b"MisguidedAttention2025"
    try:
        with open(scrambled_path, "rb") as f:
            scrambled_data = f.read()

        descrambled_data = xor_bytes(scrambled_data, scramble_key)

        with open(output_path, "wb") as f:
            f.write(descrambled_data)
        return True
    except Exception as e:
        print(f"Error descrambling file: {e}")
        return False


def load_dataset_raw(
    dataset_name: str = "normal", data_dir: str = "data"
) -> Optional[List[Dict]]:
    """Load a dataset, downloading and descrambling if needed."""
    datasets = {
        "normal": "https://raw.githubusercontent.com/cpldcpu/MisguidedAttention/main/eval/harness/misguided_attention_v4.scr",
        "long": "https://raw.githubusercontent.com/cpldcpu/MisguidedAttention/main/eval/harness/misguided_attention_v4_long.scr",
    }

    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)

    json_file = (
        data_path
        / f"misguided_attention_v4{'_long' if dataset_name == 'long' else ''}.json"
    )
    scr_file = (
        data_path
        / f"misguided_attention_v4{'_long' if dataset_name == 'long' else ''}.scr"
    )

    # Check if JSON already exists
    if json_file.exists():
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
            print(f"✓ Loaded {len(data.get('prompts', []))} examples from {json_file}")
            return data.get("prompts", [])
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            # Remove corrupted file and try to re-download
            json_file.unlink(missing_ok=True)

    # Download scrambled file if needed
    if not scr_file.exists():
        url = datasets[dataset_name]
        if not download_file(url, str(scr_file)):
            return None

    # Descramble to JSON
    print(f"Descrambling {scr_file}...")
    if not descramble_file(str(scr_file), str(json_file)):
        return None

    # Load the JSON data
    try:
        with open(json_file, "r") as f:
            data = json.load(f)
        print(f"✓ Loaded {len(data.get('prompts', []))} examples from {json_file}")
        return data.get("prompts", [])
    except Exception as e:
        print(f"Error loading descrambled JSON: {e}")
        return None


def parse_answer_spec(answer: str) -> tuple[list[str], list[float]]:
    """Parse the canonical JSON answer spec into criteria and weights.

    Expected format (stringified JSON object):
    {
      "criteria": ["...", "..."],
      "weights": [0.5, 0.5]
    }

    Raises ValueError if the format is invalid.
    """
    try:
        data = json.loads(answer)
    except Exception as exc:
        raise ValueError(
            "answer must be a JSON string with keys 'criteria' and 'weights'"
        ) from exc

    if not isinstance(data, dict) or "criteria" not in data or "weights" not in data:
        raise ValueError("answer JSON must contain 'criteria' and 'weights' fields")

    criteria_list = data.get("criteria")
    weights_list_raw = data.get("weights")

    if not isinstance(criteria_list, list) or not isinstance(weights_list_raw, list):
        raise ValueError("'criteria' and 'weights' must be lists")

    criteria_list = [str(c) for c in criteria_list]
    weights_list = [float(w) for w in weights_list_raw]

    if len(criteria_list) != len(weights_list):
        raise ValueError("'criteria' and 'weights' must have the same length")

    return criteria_list, weights_list


def generate_evaluation_prompt(output, prompt, criteria, weights):
    return f"""
You are a critical judge that evaluates the output of another LLM. 
Evaluate the following output based on the given prompt and criteria.
Respond only with true or false for each criterion.

Prompt: {prompt}

Output: {output}

Criteria:
{json.dumps(list(zip(criteria, weights)))}

You MUST return your evaluation in the following JSON format:
{{
  "criteria_results": [
    {{
      "criterion": "string",
      "met": [boolean]
    }},
    ...
  ],
  "feedback": "string"
}}
"""


def calculate_score(evaluation, weights) -> float:
    """Calculate weighted score and clip to [0,1]"""
    if not evaluation or "criteria_results" not in evaluation:
        return 0.0

    total_score = sum(
        weight * (1.0 if result["met"] else 0.0)
        for result, weight in zip(evaluation["criteria_results"], weights)
    )

    return max(0.0, min(1.0, total_score))


class JudgeRubric(vf.Rubric):
    def __init__(
        self,
        parser=None,
        parallelize_scoring: bool = False,
        judge_client: OpenAI | None = None,
        judge_model: str = "gpt-4.1-nano",
        judge_sampling_args: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(
            parser=parser, parallelize_scoring=parallelize_scoring, **kwargs
        )
        self.judge_client = judge_client if judge_client is not None else OpenAI()
        self.judge_model = judge_model
        self.judge_sampling_args = judge_sampling_args or {}

    def judge(
        self,
        prompt,
        completion,
        answer: str,
        state,
        max_depth: int = 5,
        **kwargs,
    ) -> str:
        response = self.parser.parse_answer(completion)
        criteria, weights = parse_answer_spec(answer)
        judge_prompt = generate_evaluation_prompt(response, prompt, criteria, weights)
        cached = state.get("judge_response")
        if isinstance(cached, dict) and judge_prompt in cached:
            return cached[judge_prompt]

        messages = [
            {
                "role": "system",
                "content": "You are an AI assistant that evaluates outputs based on specific criteria. Return only true/false values for each criterion",
            },
            {"role": "user", "content": judge_prompt},
        ]

        depth, success = 0, False
        while depth < max_depth and not success:
            try:
                judge_response = self.judge_client.chat.completions.create(
                    model=self.judge_model,
                    messages=messages,
                    **self.judge_sampling_args,
                )
                judge_response = judge_response.choices[0].message.content
                judge_response = (
                    "{" + judge_response.partition("{")[2].rpartition("}")[0] + "}"
                )
                json_response = json.loads(judge_response)
                success = True
            except json.JSONDecodeError:
                messages.append({"role": "assistant", "content": json_response})
                messages.append(
                    {
                        "role": "user",
                        "content": "The response was not in valid JSON format. Please provide a valid JSON response.",
                    }
                )
                depth += 1

        if not isinstance(cached, dict):
            cached = {}
        cached[judge_prompt] = judge_response
        state["judge_response"] = cached
        return judge_response


def load_environment(
    dataset_name: str = "long",
    judge_model: str = "gpt-4o-mini",
    judge_base_url: str = "https://api.openai.com/v1",
    judge_api_key_var: str = "OPENAI_API_KEY",
    judge_temperature: float = 0.0,
    judge_max_tokens: int = 1000,
    max_examples: int = -1,
    **kwargs,
) -> vf.Environment:
    """
    Loads the MisguidedAttention environment.

    This environment tests a model's ability to follow complex instructions
    while being evaluated against multiple weighted criteria.

    Args:
        dataset_name: "normal" (13 examples) or "long" (52 examples)
        judge_model: Model name for the judge
        judge_base_url: Base URL for the judge API (default: DeepSeek)
        judge_api_key_var: Environment variable name for API key
        max_examples: Maximum number of examples to load (-1 for all)
    """

    # Load the MisguidedAttention dataset
    raw_data = load_dataset_raw(dataset_name)
    if not raw_data:
        raise ValueError(f"Failed to load dataset: {dataset_name}")

    # Limit examples if requested
    if max_examples > 0 and len(raw_data) > max_examples:
        raw_data = raw_data[:max_examples]

    dataset_dict = []
    for example in raw_data:
        dataset_dict.append(
            {
                "question": example["prompt"],
                "answer": json.dumps(
                    {
                        "criteria": [
                            c
                            for c, w in zip(
                                example["criteria"],
                                example.get("weight", [1.0] * len(example["criteria"])),
                            )
                            if float(w) > 0.0
                        ],
                        "weights": [
                            float(w)
                            for w in example.get(
                                "weight", [1.0] * len(example["criteria"])
                            )
                            if float(w) > 0.0
                        ],
                    }
                ),
            }
        )

    # Create HuggingFace Dataset
    dataset = Dataset.from_list(dataset_dict)

    print(
        f"Loaded {len(dataset)} examples from MisguidedAttention {dataset_name} dataset"
    )

    # System prompt for the task
    system_prompt = """Please answer the following question:"""

    parser = Parser()

    # Create the rubric
    rubric = JudgeRubric(
        judge_client=OpenAI(
            base_url=judge_base_url,
            api_key=os.getenv(judge_api_key_var),
        ),
        judge_model=judge_model,
        judge_sampling_args={
            "temperature": judge_temperature,
            "max_tokens": judge_max_tokens,
        },
        parser=parser,
    )

    def misguided_attention_evaluation(
        prompt, completion: str, answer: str, state, **kwargs
    ) -> float:
        """Evaluate response against weighted criteria."""
        _, weights = parse_answer_spec(answer)

        # Call the judge
        judge_response = rubric.judge(
            prompt=prompt,
            completion=completion,
            answer=answer,
            state=state,
        )

        scores = calculate_score(json.loads(judge_response), weights)

        return scores

    rubric.add_reward_func(misguided_attention_evaluation, weight=1.0)

    # Create the environment
    env = vf.SingleTurnEnv(
        eval_dataset=dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        **kwargs,
    )

    return env
