<p align="center">
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/40c36e38-c5bd-4c5a-9cb3-f7b902cd155d#gh-light-mode-only" alt="Prime Intellect" width="312">
  <img src="https://github.com/user-attachments/assets/6414bc9b-126b-41ca-9307-9e982430cde8#gh-dark-mode-only"  alt="Prime Intellect" width="312">
</p>

---

<h3 align="center">
Verifiers: Environments for LLM Reinforcement Learning
</h3>

---

<p align="center">
  <a href="https://github.com/PrimeIntellect-ai/verifiers/actions/workflows/style.yml">
    <img src="https://github.com/PrimeIntellect-ai/verifiers/actions/workflows/style.yml/badge.svg" alt="Style" />
  </a>
  <a href="https://github.com/PrimeIntellect-ai/verifiers/actions/workflows/test.yml">
    <img src="https://github.com/PrimeIntellect-ai/verifiers/actions/workflows/test.yml/badge.svg" alt="Test" />
  </a>
  <a href="https://github.com/PrimeIntellect-ai/verifiers/actions/workflows/publish-environments.yml">
    <img src="https://github.com/PrimeIntellect-ai/verifiers/actions/workflows/publish-environments.yml/badge.svg" alt="Envs" />
  </a>
</p>


## Overview

Verifiers is a library of modular components for creating RL environments and training LLM agents. Environments built with Verifiers can be used directly as LLM evaluations, synthetic data pipelines, or agent harnesses for any OpenAI-compatible model endpoint, in addition to RL training. Verifiers includes an async GRPO implementation built around the `transformers` Trainer, is supported by `prime-rl` for large-scale FSDP training, and can easily be integrated into any RL framework which exposes an OpenAI-compatible inference client.

Full documentation is available [here](https://verifiers.readthedocs.io/en/latest/). 

Verifiers is also the native library used by Prime Intellect's [Environments Hub](https://app.primeintellect.ai/dashboard/environments?ex_sort=most_stars); see [here](https://docs.primeintellect.ai/tutorials-environments/environments) for information about publishing your Environments to the Hub.

## Setup

We recommend using `verifiers` along with [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management in your own project:
```bash
# install uv (first time only)
curl -LsSf https://astral.sh/uv/install.sh | sh
# create a fresh project -- 3.11 + 3.12 supported
uv init && uv venv --python 3.12 
```

For local (CPU) development and evaluation with API models, do:
```bash
uv add verifiers # uv add 'verifiers[dev]' for Jupyter + testing support
```

For training on GPUs with `vf.GRPOTrainer`, do:
```bash
uv add 'verifiers[train]' && uv pip install flash-attn --no-build-isolation
```

To use the latest `main` branch, do:
```bash
uv add verifiers@git+https://github.com/PrimeIntellect-ai/verifiers.git
```

To use with `prime-rl`, see [here](https://github.com/PrimeIntellect-ai/prime-rl).

To install `verifiers` from source for core library development, install with:
```bash
curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/verifiers/main/scripts/install.sh | bash
```

If you want to dev with the trainer, do:
```bash
uv sync --all-extras && uv pip install flash-attn --no-build-isolation
```

In general, we recommend that you build and train Environments *with* `verifiers`, not *in* `verifiers`. If you find yourself needing to clone and modify the core library in order to implement key functionality for your project, we'd love for you to open an issue so that we can try and streamline the development experience. Our aim is for `verifiers` to be a reliable toolkit to build on top of, and to minimize the "fork proliferation" which often pervades the RL infrastructure ecosystem.

## Environments

Environments in Verifiers are installable Python modules which can specify dependencies in a `pyproject.toml`, and which expose a `load_environment` function for instantiation by downstream applications (e.g. trainers). See `environments/` for examples. 

To initialize a blank Environment module template, do:
```bash
uv run vf-init environment-name # -p /path/to/environments (defaults to "./environments")
```

To an install an Environment module into your project, do:
```bash
uv run vf-install environment-name # -p /path/to/environments (defaults to "./environments") 
```

To install an Environment module from this repo's `environments` folder, do:
```bash
uv run vf-install math-python --from-repo # -b branch_or_commit (defaults to "main")
```

Once an Environment module is installed, you can create an instance of the Environment using `load_environment`, passing any necessary args:
```python
import verifiers as vf
vf_env = vf.load_environment("environment-name", **env_args)
```

To run a quick evaluation of your Environment with an API-based model, do:
```bash
uv run vf-eval environment-name -s # run and save eval results locally
# vf-eval -h for config options; defaults to gpt-4.1-mini, 5 prompts, 3 rollouts for each
```

### Multi-Environment Evaluation

`vf-eval` supports evaluating multiple environments in parallel, which is useful for benchmarking models across multiple tasks:

```bash
# Evaluate multiple environments in parallel
vf-eval gsm8k math500 aime2025 -m gpt-4o-mini -n 100 -r 3

# Per-environment configuration
vf-eval gsm8k math500 \
  --per-env-config '{
    "gsm8k": {"num_examples": 100, "rollouts_per_example": 5},
    "math500": {"num_examples": 50, "rollouts_per_example": 3}
  }'

# Save results to Prime Hub for tracking (single or multiple environments)
vf-eval gsm8k \
  -m gpt-4o-mini -n 100 -r 3 \
  --save-to-prime-hub \
  --eval-name "gsm8k-benchmark"

# Or multiple environments
vf-eval gsm8k math500 \
  -m gpt-4o-mini -n 100 -r 3 \
  --save-to-prime-hub \
  --eval-name "math-benchmark"
```

To use Prime Hub integration, install `prime` separately:
```bash
# Install verifiers
uv add verifiers

# Install prime (from local or when published)
pip install -e ../prime-cli  # or: pip install prime-cli (when published)
```

You can also use multi-environment evaluation programmatically:

```python
import asyncio
from openai import AsyncOpenAI
from verifiers.scripts.eval import eval_environments_parallel

client = AsyncOpenAI(api_key="...", base_url="http://localhost:8000/v1")

results = await eval_environments_parallel(
    envs=["gsm8k", "math500"],
    env_args_dict={"gsm8k": {}, "math500": {}},
    client=client,
    model="gpt-4o-mini",
    num_examples=[100, 50],
    rollouts_per_example=[3, 3],
    max_concurrent=[32, 32],
    sampling_args={"temperature": 0.7, "max_tokens": 2048},
)

for env, output in results.items():
    print(f"{env}: avg_reward={sum(output.reward)/len(output.reward):.3f}")
```

If you're using Prime Intellect infrastructure, the [`prime` CLI](https://github.com/PrimeIntellect-ai/prime-cli) provides first-class commands for working with Verifiers environments through the [Environments Hub](https://docs.primeintellect.ai/tutorials-environments/environments). Install it with `uv tool install prime`, authenticate via `prime login`, then use `prime env push` to publish your package and `prime env install owner/name` (optionally pinning a version) to consume it from pods or local machines.

The core elements of Environments are:
- Datasets: a Hugging Face `Dataset` with a `prompt` column for inputs, and optionally `answer (str)` or `info (dict)` columns for evaluation (both can be omitted for environments that evaluate based solely on completion quality)
- Rollout logic: interactions between models and the environment (e.g. `env_response` + `is_completed` for any `MultiTurnEnv`)
- Rubrics: an encapsulation for one or more reward functions
- Parsers: optional; an encapsulation for reusable parsing logic

We support both `/v1/chat/completions`-style and `/v1/completions`-style inference via OpenAI clients, though we generally recommend `/v1/chat/completions`-style inference for the vast majority of applications. Both the included `GRPOTrainer` as well as `prime-rl` support the full set of [SamplingParams](https://docs.vllm.ai/en/stable/api/vllm/sampling_params.html#vllm.sampling_params.SamplingParams) exposed by vLLM (via their OpenAI-compatible [server](https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html) interface), and leveraging this will often be the appropriate way to implement rollout strategies requiring finer-grained control, such as interrupting and resuming generations for interleaved tool use, or enforcing reasoning budgets.

The primary constraint we impose on rollout logic is that token sequences must be *increasing*, i.e. once a token has been added to a model's context in a rollout, it must remain as the rollout progresses. Note that this causes issues with some popular reasoning models such as the Qwen3 and DeepSeek-R1-Distill series; see [Troubleshooting](https://verifiers.readthedocs.io/en/latest/training.html#common-issues) for guidance on adapting these models to support multi-turn rollouts.  

### SingleTurnEnv

For tasks requiring only a single response from a model for each prompt, you can use `SingleTurnEnv` directly by specifying a Dataset and a Rubric. Rubrics are sets of reward functions, which can be either sync or async.

```python
from datasets import load_dataset
import verifiers as vf

dataset = load_dataset("my-account/my-dataset", split="train")

def reward_A(prompt, completion, info) -> float:
	# reward fn, e.g. correctness
	...

def reward_B(parser, completion) -> float:
	# auxiliary reward fn, e.g. format
	...

async def metric(completion) -> float:
	# non-reward metric, e.g. proper noun count
	...

rubric = vf.Rubric(funcs=[reward_A, reward_B, metric], weights=[1.0, 0.5, 0.0])

vf_env = SingleTurnEnv(
	dataset=dataset,
	rubric=rubric
)
results = vf_env.evaluate(client=OpenAI(), model="gpt-4.1-mini", num_examples=100, rollouts_per_example=1)
vf_env.make_dataset(results) # HF dataset format
```

Datasets should be formatted with columns for:
- `'prompt' (List[ChatMessage])` OR `'question' (str)` fields
	- `ChatMessage` = e.g. `{'role': 'user', 'content': '...'}`
	- if `question` is set instead of `prompt`, you can also pass `system_prompt (str)` and/or `few_shot (List[ChatMessage])`
- `answer (str)` AND/OR `info (dict)` (both optional, can be omitted entirely)
- `task (str)`: optional, used by `EnvGroup` and `RubricGroup` for orchestrating composition of Environments and Rubrics

The following named attributes available for use by reward functions in your Rubric:
- `prompt`: sequence of input messages
- `completion`: sequence of messages generated during rollout by model and Environment
- `answer`: primary answer column, optional (defaults to empty string if omitted)
- `state`: can be modified during rollout to accumulate any metadata (`state['responses']` includes full OpenAI response objects by default)
- `info`: auxiliary info needed for reward computation (e.g. test cases), optional (defaults to empty dict if omitted)
- `task`: tag for task type (used by `EnvGroup` and `RubricGroup`)
- `parser`: the parser object declared. Note: `vf.Parser().get_format_reward_func()` is a no-op (always 1.0); use `vf.ThinkParser` or a custom parser if you want a real format adherence reward.

**Note**: Some environments can fully evaluate using only `prompt`, `completion`, and `state` without requiring ground truth `answer` or `info` data. Examples include format compliance checking, completion quality assessment, or length-based rewards.

For tasks involving LLM judges, you may wish to use `vf.JudgeRubric()` for managing requests to auxiliary models.

Note on concurrency: environment APIs accept `max_concurrent` to control parallel rollouts. The `vf-eval` CLI currently exposes `--max-concurrent-requests`; ensure this maps to your environment’s concurrency as expected.

`vf-eval` also supports specifying `sampling_args` as a JSON object, which is sent to the vLLM inference engine:

```bash
uv run vf-eval vf-environment-name --sampling-args '{"reasoning_effort": "low"}'
```

Use `vf-eval -s` to save outputs as dataset-formatted JSON, and view all locally-saved eval results with `vf-tui`.

### ToolEnv

For many applications involving tool use, you can use `ToolEnv` to leverage models' native tool/function-calling capabilities in an agentic loop. Tools must be stateless and idempotent—each call should be fully determined by the provided arguments—because the environment will automatically terminate once the assistant responds without tool calls. Tools can be specified as generic Python functions (with type hints and docstrings), which will then be passed in JSON schema form to each inference request.


```python
import verifiers as vf
vf_env = vf.ToolEnv(
	dataset= ... # HF Dataset with 'prompt'/'question' and optionally 'answer'/'info' columns
	rubric= ... # Rubric object; vf.ToolRubric() can be optionally used for counting tool invocations in each rollout
	tools=[search_tool, read_article_tool, python_tool], # python functions with type hints + docstrings
	max_turns=10
)
```

In cases where your tools require heavy computational resources, we recommend hosting your tools as standalone servers (e.g. MCP servers) and creating lightweight wrapper functions to pass to `ToolEnv`. Parallel tool call support is enabled by default. If you need to inject per-rollout or cross-call state (IDs, credentials, cached resources), promote the environment to `StatefulToolEnv` and populate that state through `setup_state`/`update_tool_args` instead of hiding globals.

#### StatefulToolEnv

`StatefulToolEnv` extends `ToolEnv` for workflows where tool calls must incorporate dynamic state (for example, sandbox handles or per-user secrets). Implement `setup_state` to seed the state dict and override `update_tool_args` to merge state into each tool invocation. Any arguments you strip from the OpenAI schema via `args_to_skip` should be tracked in `skipped_args` so the model never sees sensitive parameters. Avoid storing global state; keep everything in the provided `state` dict.

#### SandboxEnv & PythonEnv

`SandboxEnv` builds on `StatefulToolEnv` to coordinate long-running sandboxes. Queue heavyweight provisioning inside `setup_state` (without awaiting) and gate tool execution on readiness inside `update_tool_args` or the tools themselves. `PythonEnv` is a concrete sandboxed executor that demonstrates the pattern: it spins up a Prime sandbox, injects the sandbox ID into each tool call, and tears down resources when the rollout finishes. Treat both environments as references when building similar stateful tool workflows.

For training, or self-hosted endpoints, you'll want to enable auto tool choice in [vLLM](https://docs.vllm.ai/en/stable/features/tool_calling.html#automatic-function-calling) with the appropriate parser. If your model does not support native tool calling, you may find the `XMLParser` abstraction useful for rolling your own tool call parsing on top of `MultiTurnEnv`; see `environments/xml_tool_env` for an example.

### MultiTurnEnv

Both `SingleTurnEnv` and `ToolEnv` are instances of `MultiTurnEnv`, which exposes an interface for writing custom Environment interaction protocols. Override `is_completed` and `env_response`, and make sure any custom completion logic defers to the base class so turn limits and other shared guards keep working.


```python
from typing import Tuple
import verifiers as vf
from verifiers.types import Messages, State
class YourMultiTurnEnv(vf.MultiTurnEnv):
    def __init__(self,
                 dataset: Dataset,
                 rubric: Rubric,
				 max_turns: int,
                 **kwargs):
	
  async def is_completed(self, messages: Messages, state: State, **kwargs) -> bool:
    # Always call the base check so max_turns and shared guards are respected
    if await super().is_completed(messages, state, **kwargs):
        return True
    # return whether or not a rollout is completed
    return state.get("task_complete", False)

  async def env_response(self, messages: Messages, state: State, **kwargs) -> Tuple[Messages, State]:
    # return new environment message(s) + updated state
```

If your application requires more fine-grained control than is allowed by `MultiTurnEnv`, you may want to inherit from the base `Environment` functionality directly and override the `rollout` method.

### ToolEnv
For many applications involving tool use, you can use `ToolEnv` to leverage models' native tool/function-calling capabilities in an agentic loop. Tools must be stateless and idempotent—each call should be fully determined by the provided arguments—because the environment will automatically terminate once the assistant responds without tool calls.

#### StatefulToolEnv
`StatefulToolEnv` extends `ToolEnv` for workflows where tool calls must incorporate dynamic state ...
#### SandboxEnv & PythonEnv
`SandboxEnv` builds on `StatefulToolEnv` to coordinate long-running sandboxes ... `PythonEnv` is a concrete sandboxed executor that demonstrates the pattern ...

## Training


### GRPOTrainer

The included trainer (`vf.GRPOTrainer`) supports running GRPO-style RL training via Accelerate/DeepSpeed, and uses vLLM for inference. It supports both full-parameter finetuning, and is optimized for efficiently training dense transformer models on 2-16 GPUs.

```bash
# install environment
vf-install vf-wordle (-p /path/to/environments | --from-repo)

# quick eval
vf-eval vf-wordle -m (model_name in configs/endpoints.py) -n NUM_EXAMPLES -r ROLLOUTS_PER_EXAMPLE

# inference (shell 0)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 vf-vllm --model willcb/Qwen3-1.7B-Wordle \
    --data-parallel-size 7 --enforce-eager --disable-log-requests

# training (shell 1)
CUDA_VISIBLE_DEVICES=6,7 accelerate launch --num-processes 2 \
    --config-file configs/zero3.yaml examples/grpo/train_wordle.py --size 1.7B
```

Alternatively, you can train environments with the external `prime-rl` project (FSDP-first orchestration). See the `prime-rl` README for installation and examples. For example:

```toml
# orchestrator config (prime-rl)
[environment]
id = "vf-math-python"  # or your environment ID
```

```bash
# run (prime-rl)
uv run rl \
  --trainer @ configs/your_exp/train.toml \
  --orchestrator @ configs/your_exp/orch.toml \
  --inference @ configs/your_exp/infer.toml
```

### Troubleshooting 
- Ensure your `wandb` and `huggingface-cli` logins are set up (or set `report_to=None` in `training_args`). You should also have something set as your `OPENAI_API_KEY` in your environment (can be a dummy key for vLLM). 
- If using high max concurrency, increase the number of allowed open sockets (e.g. `ulimit -n 4096`)
- On some setups, inter-GPU communication can [hang](https://github.com/huggingface/trl/issues/2923) or crash during vLLM weight syncing. This can usually be alleviated by setting (or unsetting) `NCCL_P2P_DISABLE=1` in your environment (or potentially `NCCL_CUMEM_ENABLE=1`). Try this as your first step if you experience NCCL-related issues.
- If problems persist, please open an [issue](https://github.com/PrimeIntellect-ai/verifiers/issues).

### Resource Requirements
`GRPOTrainer` is optimized for setups with at least 2 GPUs, scaling up to multiple nodes. 2-GPU setups with sufficient memory to enable small-scale experimentation can be [rented](https://app.primeintellect.ai/dashboard/create-cluster?image=ubuntu_22_cuda_12) for <$1/hr.

### PRIME-RL
If you do not require LoRA support, you may want to use the `prime-rl` trainer, which natively supports Environments created using `verifiers`, is more optimized for performance and scalability via FSDP, includes a broader set of configuration options and user experience features, and has more battle-tested defaults. Both trainers support asynchronous rollouts, and use a one-step off-policy delay by default for overlapping training and inference. See the `prime-rl` [docs](https://github.com/PrimeIntellect-ai/prime-rl) for usage instructions.

## Further Documentation

See the full [docs](https://verifiers.readthedocs.io/en/latest/) for more information.

## Contribution Guidelines

Verifiers warmly welcomes community contributions! Please open an issue or PR if you encounter bugs or other pain points during your development, or start a discussion for more open-ended questions.

Please note that the core `verifiers/` library is intended to be a relatively lightweight set of reusable components rather than an exhaustive catalog of RL environments. Consider sharing any environments you create to the [Environments Hub](https://app.primeintellect.ai/dashboard/environments) 🙂

## Citation

Originally created by Will Brown ([@willccbb](https://github.com/willccbb)).

If you use this code in your research, please cite:

```bibtex
@misc{brown_verifiers_2025,
  author       = {William Brown},
  title        = {{Verifiers}: Environments for LLM Reinforcement Learning},
  howpublished = {\url{https://github.com/willccbb/verifiers}},
  note         = {Commit abcdefg • accessed DD Mon YYYY},
  year         = {2025}
}
```
