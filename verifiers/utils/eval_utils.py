import json
import uuid
from datetime import datetime
from pathlib import Path

from datasets import Dataset

from verifiers.types import GenerateOutputs
from verifiers.utils.message_utils import sanitize_tool_calls


def prepare_dataset(
    results: GenerateOutputs, num_examples: int, rollouts_per_example: int
) -> Dataset:
    """Prepare dataset from eval results."""
    n, r = num_examples, rollouts_per_example
    ids = [i // r for i in range(n * r)]
    data_dict = {
        "id": ids,
        "prompt": [sanitize_tool_calls(p) for p in results.prompt],
        "completion": [sanitize_tool_calls(c) for c in results.completion],
        "task": results.task,
        "generation_ms": [s["timing"]["generation_ms"] for s in results.state],
        "scoring_ms": [s["timing"]["scoring_ms"] for s in results.state],
        "total_ms": [s["timing"]["total_ms"] for s in results.state],
    }
    if results.info[0] != {}:
        data_dict["info"] = results.info
    if results.answer[0] != "":
        data_dict["answer"] = results.answer
    data_dict["reward"] = results.reward
    for k in results.metrics:
        v = results.metrics[k]
        data_dict[k] = v

    return Dataset.from_dict(data_dict)


def prepare_metadata(
    env: str,
    model: str,
    num_examples: int,
    rollouts_per_example: int,
    sampling_args: dict,
    end_time: float,
    start_time: float,
    results: GenerateOutputs,
) -> dict:
    """Prepare metadata from eval results."""
    metadata = {
        "env": env,
        "model": model,
        "num_examples": num_examples,
        "rollouts_per_example": rollouts_per_example,
        "sampling_args": sampling_args,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "time_ms": (end_time - start_time) * 1000,
        "avg_reward": sum(results.reward) / len(results.reward),
    }
    for k in results.metrics:
        metadata[f"avg_{k}"] = sum(results.metrics[k]) / len(results.metrics[k])

    return metadata


def save_results_to_disk(
    dataset: Dataset, metadata: dict, env: str, model: str, env_dir_path: str
) -> Path:
    """Save eval results dataset to disk."""
    env_model_str = f"{env}--{model.replace('/', '--')}"
    uuid_str = str(uuid.uuid4())[:8]
    module_name = env.replace("-", "_")
    local_env_dir = Path(env_dir_path) / module_name
    if local_env_dir.exists():
        results_path = local_env_dir / "outputs" / "evals" / env_model_str / uuid_str
    else:
        results_path = Path("./outputs") / "evals" / env_model_str / uuid_str
    results_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_json(results_path / "results.jsonl")
    with open(results_path / "metadata.json", "w") as f:
        json.dump(metadata, f)
    return results_path


def save_results_to_hf_hub(
    dataset: Dataset,
    env: str,
    model: str,
    num_examples: int,
    rollouts_per_example: int,
    hf_hub_dataset_name: str,
) -> str:
    """Push eval results dataset to Hugging Face Hub."""
    if hf_hub_dataset_name == "":
        dataset_name = (
            f"{env}_{model.replace('/', '-')}_n{num_examples}_r{rollouts_per_example}"
        )
    else:
        dataset_name = hf_hub_dataset_name
    dataset.push_to_hub(dataset_name, env)
    return dataset_name
