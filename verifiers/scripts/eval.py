import importlib
import importlib.util
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import typer
from datasets import Dataset
from openai import OpenAI

import verifiers as vf
from verifiers.scripts.tui import RunInfo, VerifiersTUI, ViewRunScreen
from verifiers.utils.message_utils import messages_to_printable, sanitize_tool_calls


def eval_environment(
    env: str,
    env_args: dict,
    env_dir_path: str,
    endpoints_path: str,
    model: str,
    api_key_var: str,
    api_base_url: str,
    num_examples: int,
    rollouts_per_example: int,
    max_concurrent: int,
    max_tokens: int | None,
    temperature: float | None,
    sampling_args: dict | None,
    save_dataset: bool,
    save_to_hf_hub: bool,
    hf_hub_dataset_name: str,
) -> Dict[str, Any]:
    try:
        endpoints_path_obj = Path(endpoints_path)
        if endpoints_path_obj.is_dir():
            endpoints_file = endpoints_path_obj / "endpoints.py"
        else:
            endpoints_file = endpoints_path_obj

        if endpoints_file.exists():
            logger.info(f"Loading endpoint registry from {endpoints_file}")
            spec = importlib.util.spec_from_file_location("endpoints", endpoints_file)
            assert spec and spec.loader
            endpoints_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(endpoints_module)
            ENDPOINTS = endpoints_module.ENDPOINTS
            logger.info(f"Successfully loaded {len(ENDPOINTS)} endpoints from registry")
        else:
            raise ImportError(f"endpoints.py not found at {endpoints_file}")
    except (ImportError, AttributeError) as e:
        logger.warning(
            f"No local endpoint registry found at {endpoints_path}. "
            f"Please specify the model name (-m), API host base URL (-b), and API key variable name (-k). "
            f"Error details: {str(e)}"
        )
        logger.info("Using default empty endpoints registry")
        ENDPOINTS = {}

    if model in ENDPOINTS:
        api_key_var = ENDPOINTS[model]["key"]
        api_base_url = ENDPOINTS[model]["url"]
        model = ENDPOINTS[model]["model"]
        logger.info(f"Using endpoint configuration for model '{model}' from registry")
    else:
        logger.info(
            f"Model '{model}' not found in endpoint registry, using command-line arguments"
        )

    # Setup eval client with high limits, effectively preventing any API timeout errors
    client = setup_client(
        api_base_url,
        api_key_var,
        timeout=36000.0,  # 10h
        max_connections=28000,  # Number of available ports
        max_keepalive_connections=28000,  # Number of available ports
        max_retries=10,  # 10 retries (w/ exponential backoffs)
    )
    logger.info(f"Initialized OpenAI client with base_url: {api_base_url}")

    logger.info(f"Loading environment: {env}")
    vf_env = vf.load_environment(env_id=env, **env_args)
    logger.info(f"Successfully loaded environment: {env}")
    # Merge sampling args with precedence to JSON payload over explicit flags
    merged_sampling_args: dict = {}
    if sampling_args is not None:
        merged_sampling_args.update(sampling_args)
    if "max_tokens" not in merged_sampling_args:
        merged_sampling_args["max_tokens"] = max_tokens
    if temperature is not None and "temperature" not in merged_sampling_args:
        merged_sampling_args["temperature"] = temperature

    logger.info(f"Starting evaluation with model: {model}")
    logger.info(
        f"Configuration: num_examples={num_examples}, rollouts_per_example={rollouts_per_example}, max_concurrent={max_concurrent}"
    )

    results = vf_env.evaluate(
        client=client,
        model=model,
        sampling_args=merged_sampling_args,
        num_examples=num_examples,
        rollouts_per_example=rollouts_per_example,
        max_concurrent=max_concurrent,
    )

    # Prepare evaluation results data
    printable_prompts = [messages_to_printable(p) for p in results.prompt]
    printable_completions = [messages_to_printable(c) for c in results.completion]

    # Calculate statistics
    n = num_examples
    r = rollouts_per_example
    if n < 0:
        n = len(results.reward) // r

    # Prepare rewards data
    reward_stats = {
        "avg": sum(results.reward) / len(results.reward),
        "std": np.std(results.reward),
        "trials": [],
    }
    for i in range(r):
        trials = [round(results.reward[(i * n) + j], 3) for j in range(n)]
        reward_stats["trials"].append(trials)

    # Prepare metrics data
    metrics_data = {}
    for k in results.metrics:
        v = results.metrics[k]
        metrics_data[k] = {"avg": sum(v) / len(v), "std": np.std(v), "trials": []}
        for i in range(r):
            trials = [round(v[(i * n) + j], 3) for j in range(n)]
            metrics_data[k]["trials"].append(trials)

    ids = [i // rollouts_per_example for i in range(n * rollouts_per_example)]
    rewards = results.reward
    tasks = results.task
    data_dict = {
        "id": ids,
        "prompt": [sanitize_tool_calls(p) for p in printable_prompts],
        "completion": [sanitize_tool_calls(c) for c in printable_completions],
        "task": tasks,
    }
    if results.info[0] != {}:
        data_dict["info"] = results.info
    if results.answer[0] != "":
        data_dict["answer"] = results.answer
    data_dict["reward"] = rewards
    for k in results.metrics:
        v = results.metrics[k]
        data_dict[k] = v

    dataset = Dataset.from_dict(data_dict)
    metadata = {
        "env": env,
        "model": model,
        "num_examples": n,
        "rollouts_per_example": rollouts_per_example,
        "sampling_args": merged_sampling_args,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "time": datetime.now().strftime("%H:%M:%S"),
        "avg_reward": sum(results.reward) / len(results.reward),
    }
    for k in results.metrics:
        metadata[f"avg_{k}"] = sum(results.metrics[k]) / len(results.metrics[k])

    if save_dataset or save_to_hf_hub:
        uuid_str = str(uuid.uuid4())[:8]
        env_model_str = f"{env}--{model.replace('/', '--')}"
        if save_dataset:
            module_name = env.replace("-", "_")
            local_env_dir = Path(env_dir_path) / module_name
            if local_env_dir.exists():
                results_path = (
                    local_env_dir / "outputs" / "evals" / env_model_str / uuid_str
                )
            else:
                results_path = Path("./outputs") / "evals" / env_model_str / uuid_str
            results_path.parent.mkdir(parents=True, exist_ok=True)
            dataset.to_json(results_path / "results.jsonl")
            with open(results_path / "metadata.json", "w") as f:
                json.dump(metadata, f)

            logger.info(f"Saved dataset to {results_path}")
        if save_to_hf_hub:
            if hf_hub_dataset_name == "":
                dataset_name = (
                    f"{env}_{model.replace('/', '-')}_n{n}_r{rollouts_per_example}"
                )
            else:
                dataset_name = hf_hub_dataset_name
            dataset.push_to_hub(dataset_name)
            logger.info(f"Saved dataset to Hugging Face Hub: {dataset_name}")

    return dataset, metadata


# Create Typer app
app = typer.Typer()


@app.command()
def evaluate(
    env: str = typer.Argument("gsm8k", help="Environment module name"),
    env_args: str = typer.Option(
        "{}",
        "--env-args",
        "-a",
        help='Environment module arguments as JSON object (e.g., \'{"key": "value", "num": 42}\')',
    ),
    env_dir_path: str = typer.Option(
        "./environments",
        "--env-dir-path",
        "-p",
        help="Path to environments directory",
    ),
    endpoints_path: str = typer.Option(
        "./configs/endpoints.py",
        "--endpoints-path",
        "-e",
        help="Path to API endpoints registry",
    ),
    model: str = typer.Option(
        "gpt-4.1-mini",
        "--model",
        "-m",
        help="Name of model to evaluate",
    ),
    api_key_var: str = typer.Option(
        "OPENAI_API_KEY",
        "--api-key-var",
        "-k",
        help="Environment variable name for API key",
    ),
    api_base_url: str = typer.Option(
        "https://api.openai.com/v1",
        "--api-base-url",
        "-b",
        help="Base URL for API",
    ),
    num_examples: int = typer.Option(
        5,
        "--num-examples",
        "-n",
        help="Number of examples to evaluate",
    ),
    rollouts_per_example: int = typer.Option(
        3,
        "--rollouts-per-example",
        "-r",
        help="Number of rollouts per example",
    ),
    max_concurrent_requests: int = typer.Option(
        32,
        "--max-concurrent-requests",
        "-c",
        help="Maximum number of concurrent requests",
    ),
    max_tokens: Optional[int] = typer.Option(
        None,
        "--max-tokens",
        "-t",
        help="Maximum number of tokens to generate (unset to use model default)",
    ),
    temperature: Optional[float] = typer.Option(
        None,
        "--temperature",
        "-T",
        help="Temperature for sampling",
    ),
    sampling_args: Optional[str] = typer.Option(
        None,
        "--sampling-args",
        "-S",
        help=(
            "Sampling arguments as JSON object. Keys here override --max-tokens/--temperature. "
            'Example: \'{"enable_thinking": false, "max_tokens": 256}\''
        ),
    ),
    save_dataset: bool = typer.Option(
        False,
        "--save-dataset",
        "-s",
        help="Save dataset to disk",
    ),
    save_to_hf_hub: bool = typer.Option(
        False,
        "--save-to-hf-hub",
        "-H",
        help="Save dataset to Hugging Face Hub",
    ),
    hf_hub_dataset_name: str = typer.Option(
        "",
        "--hf-hub-dataset-name",
        "-D",
        help="Name of dataset to save to Hugging Face Hub",
    ),
    no_display: bool = typer.Option(
        False,
        "--no-display",
        help="Skip displaying results in Textual interface",
    ),
):
    """
    Evaluate a model using the specified environment and display results.
    """
    # Parse JSON arguments
    try:
        parsed_env_args = json.loads(env_args)
    except json.JSONDecodeError as e:
        typer.echo(f"Error parsing env-args JSON: {e}", err=True)
        raise typer.Exit(1)

    try:
        parsed_sampling_args = json.loads(sampling_args) if sampling_args else None
    except json.JSONDecodeError as e:
        typer.echo(f"Error parsing sampling-args JSON: {e}", err=True)
        raise typer.Exit(1)

    # Run evaluation
    dataset, metadata = eval_environment(
        env=env,
        env_args=parsed_env_args,
        env_dir_path=env_dir_path,
        endpoints_path=endpoints_path,
        model=model,
        api_key_var=api_key_var,
        api_base_url=api_base_url,
        num_examples=num_examples,
        rollouts_per_example=rollouts_per_example,
        max_concurrent_requests=max_concurrent_requests,
        max_tokens=max_tokens,
        temperature=temperature,
        sampling_args=parsed_sampling_args,
        save_dataset=save_dataset,
        save_to_hf_hub=save_to_hf_hub,
        hf_hub_dataset_name=hf_hub_dataset_name,
    )

    # Display results using Textual interface if requested
    if not no_display:
        launch_tui(dataset, metadata, env, model)
    else:
        typer.echo(
            "Evaluation completed. Use --no-display=false to show results in Textual interface."
        )


class DirectViewTUI(VerifiersTUI):
    """TUI that goes directly to scripts/tui.py:ViewRunScreen without showing the main menu."""

    def __init__(self, view_screen):
        super().__init__("./environments", "./outputs", False)
        self.view_screen = view_screen

    def on_mount(self) -> None:
        # Register both custom themes
        self.register_theme(self.BLACK_WARM_THEME)
        self.register_theme(self.WHITE_WARM_THEME)
        # Start with dark theme
        self.theme = "black-warm"
        # Skip the main menu and go directly to ViewRunScreen
        self.push_screen(self.view_screen)


def launch_tui(dataset, metadata, env, model):
    """Launch TUI with the evaluation results."""
    import asyncio

    async def run_tui():
        if dataset and metadata:
            # Convert dataset to list of dictionaries for ViewRunScreen
            records = [row for row in dataset]

            # Create a minimal RunInfo object with just the metadata (no file operations)
            run_info = RunInfo(
                env_id=env,
                model=model,
                run_id=f"run-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}",
                path=Path("./"),
                metadata=metadata,
            )

            # Create ViewRunScreen with the data
            view_screen = ViewRunScreen(run_info, records)

            # Create the direct TUI app that bypasses the main menu
            tui_app = DirectViewTUI(view_screen)

            # Run the TUI
            await tui_app.run_async()

    # Run the async TUI function
    asyncio.run(run_tui())


def main():
    app()


if __name__ == "__main__":
    main()
