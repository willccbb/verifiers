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
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.widgets import Footer, Label, Static

import verifiers as vf
from verifiers.utils.message_utils import sanitize_tool_calls
from verifiers.scripts.tui import format_prompt_or_completion


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
    printable_prompts = [format_prompt_or_completion(p) for p in results.prompt]
    printable_completions = [format_prompt_or_completion(c) for c in results.completion]

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

    # Structure the evaluation data
    eval_data = {
        "environment": env,
        "model": model,
        "provider": api_base_url,
        "num_examples": n,
        "rollouts_per_example": r,
        "max_concurrent_requests": max_concurrent_requests,
        "merged_sampling_args": merged_sampling_args,
        "prompts": printable_prompts,
        "completions": printable_completions,
        "rewards": reward_stats,
        "metrics": metrics_data,
        "results": results,  # Keep original results object for dataset saving
    }

    if save_dataset or save_to_hf_hub:
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

    return eval_data


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
    eval_data = eval_environment(
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

    # Display results using Textual interface
    if not no_display:
        display_app = EvaluationResultsApp(eval_data)
        display_app.run()
    else:
        typer.echo(
            "Evaluation completed. Use --no-display=false to show results in Textual interface."
        )


# Textual app for displaying evaluation results
class EvaluationResultsApp(App):
    """Textual app for displaying evaluation results."""

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("left", "prev_example", "Previous example"),
        ("h", "prev_example", "Previous example"),
        ("right", "next_example", "Next example"),
        ("l", "next_example", "Next example"),
        ("home", "first_example", "First example"),
        ("end", "last_example", "Last example"),
    ]

    CSS = """
    Screen {
        layout: vertical;
        background: #1a1a1a;
    }

    Container {
        layout: vertical;
        height: 100%;
    }

    .header-panel {
        height: auto;
        min-height: 4;
        max-height: 8;
        border: solid #666;
        padding: 1 2;
        margin-bottom: 1;
    }

    .stats-panel {
        height: auto;
        min-height: 6;
        max-height: 10;
        border: solid #666;
        padding: 1 2;
        margin-bottom: 1;
    }

    .rollout-container {
        height: 1fr;
        layout: horizontal;
    }

    .column-panel {
        width: 50%;
        height: 100%;
        layout: vertical;
        border: solid #666;
        padding: 1 2;
    }

    .column-header {
        height: auto;
        margin-bottom: 1;
        text-align: center;
        text-style: bold;
        color: #00ff00;
    }

    VerticalScroll {
        height: 1fr;
        background: #2a2a2a;
        padding: 0 1;
    }

    Static {
        color: #ffffff;
    }

    Label {
        color: #ffffff;
    }

    Footer {
        background: #333;
        color: #ccc;
    }
    """

    def __init__(self, eval_data: Dict[str, Any]):
        super().__init__()
        self.eval_data = eval_data
        self.current_example = 0

    def compose(self) -> ComposeResult:
        with Container():
            # Header with evaluation info
            with Container(classes="header-panel"):
                yield Label("🔬 Evaluation Results", classes="column-header")
                yield Static(self._get_header_text(), id="header-info")

            # Statistics panel
            with Container(classes="stats-panel"):
                yield Label("📊 Statistics", classes="column-header")
                yield Static(self._get_stats_text())

            # Example rollout display
            with Horizontal(classes="rollout-container"):
                with Container(classes="column-panel"):
                    yield Label("📝 Prompt", classes="column-header")
                    yield VerticalScroll(
                        Static(self._get_current_prompt()),
                        id="prompt-content",
                    )

                with Container(classes="column-panel"):
                    yield Label("🤖 Completion", classes="column-header")
                    yield VerticalScroll(
                        Static(self._get_current_completion()),
                        id="completion-content",
                    )

        yield Footer(show_command_palette=False)

    def _get_header_text(self) -> str:
        """Generate header information text."""
        data = self.eval_data
        total_examples = len(data["prompts"]) if data["prompts"] else 0
        current_idx = self.current_example + 1 if total_examples > 0 else 0

        # Prominent example indicator at the top
        example_indicator = f"[bold bright_green]📋 Example {current_idx} of {total_examples}[/bold bright_green]"
        if total_examples == 0:
            example_indicator = "[dim]No examples available[/dim]"

        return f"""{example_indicator}

[b]Environment:[/b] {data["environment"]}
[b]Model:[/b] {data["model"]}
[b]Provider:[/b] {data["provider"]}
[b]Rollouts/example:[/b] {data["rollouts_per_example"]} | [b]Max Concurrent:[/b] {data["max_concurrent_requests"]}"""

    def _get_stats_text(self) -> str:
        """Generate statistics text."""
        data = self.eval_data
        rewards = data["rewards"]

        lines = [
            f"[b]Rewards:[/b] Avg: {rewards['avg']:.3f} | Std: {rewards['std']:.3f}",
        ]

        # Add rollout trials
        for i, trials in enumerate(rewards["trials"], 1):
            lines.append(f"[b]Rollout {i}:[/b] {trials}")

        # Add other metrics
        for metric_name, metric_data in data["metrics"].items():
            lines.append(
                f"[b]{metric_name}:[/b] Avg: {metric_data['avg']:.3f} | Std: {metric_data['std']:.3f}"
            )
            for i, trials in enumerate(metric_data["trials"], 1):
                lines.append(f"[b]{metric_name} R{i}:[/b] {trials}")

        return "\n".join(lines)

    def _get_current_prompt(self) -> str:
        """Get the current example's prompt."""
        if self.current_example < len(self.eval_data["prompts"]):
            prompt = self.eval_data["prompts"][self.current_example]
            return prompt
        return "[dim]No prompt available[/dim]"

    def _get_current_completion(self) -> str:
        """Get the current example's completion."""
        if self.current_example < len(self.eval_data["completions"]):
            completion = self.eval_data["completions"][self.current_example]
            return completion
        return "[dim]No completion available[/dim]"

    def _update_display(self) -> None:
        """Update the display with current example."""
        # Update prompt
        prompt_scroll = self.query_one("#prompt-content", VerticalScroll)
        prompt_scroll.query_one(Static).update(self._get_current_prompt())

        # Update completion
        completion_scroll = self.query_one("#completion-content", VerticalScroll)
        completion_scroll.query_one(Static).update(self._get_current_completion())

        # Update header with current example info
        # Find the header static widget and update it
        try:
            header_widget = self.query_one("#header-info", Static)
            header_widget.update(self._get_header_text())
        except Exception:
            pass  # Header might not be found, continue anyway

        # Reset scroll positions
        try:
            prompt_scroll.scroll_y = 0
            completion_scroll.scroll_y = 0
        except Exception:
            pass  # Scroll widgets might not be found, continue anyway

    def action_prev_example(self) -> None:
        """Navigate to previous example."""
        if self.eval_data["prompts"]:
            self.current_example = (self.current_example - 1) % len(
                self.eval_data["prompts"]
            )
            self._update_display()

    def action_next_example(self) -> None:
        """Navigate to next example."""
        if self.eval_data["prompts"]:
            self.current_example = (self.current_example + 1) % len(
                self.eval_data["prompts"]
            )
            self._update_display()

    def action_first_example(self) -> None:
        """Navigate to first example."""
        if self.eval_data["prompts"]:
            self.current_example = 0
            self._update_display()

    def action_last_example(self) -> None:
        """Navigate to last example."""
        if self.eval_data["prompts"]:
            self.current_example = len(self.eval_data["prompts"]) - 1
            self._update_display()

    def action_quit(self) -> None:
        """Quit the application."""
        self.exit()


def main():
    app()


if __name__ == "__main__":
    main()
