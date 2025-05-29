# Verifiers: Reinforcement Learning with LLMs in Verifiable Environments

TODO: GitHub header stuff, test coverage, [PyPI](https://pypi.org/project/verifiers/), general release prep

## Roadmap for v0.1 Release (very soon)

New features for this release:
- Async inference support via OpenAI-compatible vLLM server (with weight syncing enabled)
- Async execution for rollouts + rubrics
- Native support for [reasoning-gym](https://github.com/open-thought/reasoning-gym) environments
- Overlapped training + inference (via off-policy steps)
- Rollout-level reward functions by default (with weight=0.0 supported)
- Direct support for API evaluation + synthetic data collection 
- Complete workflow for API eval -> data collection -> SFT -> RL (GRPO)
- Full decoupling of rollout + reward logic from GRPOEnvTrainer
- `transformers` Trainer as the base (replacing TRL's GRPO)
- Direct support for LLM judges via JudgeRubric
Included, but could use more testing:
- Data-parallel vLLM workers
- Multi-node training
Not included, but planned for later releases:
- TextArena environments
- Enigmata environments
- Native MCP tool support
- Multimodal support (image-in, via /v1/chat/completions)
- Tokenizer endpoint exposed for better token-level + turn-level mechanics (edge case handling, token-level rewards)
- More flexible abstractions for dynamic batch construction + rollout reuse
- FSDP (via prime-rl) 

## Overview

`verifiers` is a set of tools and abstractions for training LLMs with reinforcement learning in **verifiable multi-turn environments** via Group-Relative Policy Optimization ([GRPO](https://huggingface.co/docs/trl/main/en/grpo_trainer)). In addition, `verifiers` includes support for synthetic data generation, SFT warmup on filtered traces, and offline evaluation with API clients.

**Core principles**:
RL environments and algorithms should be modular, reusable, and hackable.


- Client (Actor): 
- environment = instructions + tasks + interaction protocol + rubric
- instructions = system prompts
- tasks = datasets + verifiable targets
- (multi-turn) interaction protocol = tool calls, gameplay, multi-agent systems, end-state determination
- rubric = reward mechanisms for verifying performance on instruction + task objectives
- environments = synthetic data engines = RL trainers = eval harnesses

**Key features:**
- First-class support for multi-turn tool use and agentic RL via `GRPOEnvTrainer`, built on top of Transformers.
- Direct compatibility with API clients for synthetic data generation and evaluation, in addition to RL training with vLLM clients.
- Utilities for SFT warmup/"cold start" data (see `examples/warmup` scripts)
- Emphasis on OpenAI/`chat`-compatible messages rather than plaintext.
- `Parser` classes (e.g. `XMLParser`) for standardizing your prompt formats and text extraction.
- `Rubric` classes for managing sets of reward functions.
- `Environment` classes for encapsulating your tasks, parsers, rollout logic, and reward functions, including:
	- `SingleTurnEnv` for "R1-style" reasoning via vLLM's `chat()` method.
	- `SimpleEnv` for single-turn completion-style rollouts using vLLM's `generate()` method.
  - `SimpleEnv` for completion-style resp
	- `ToolEnv` for multi-turn tool use with custom Python functions.
	- `SmolaToolEnv` for multi-turn tool use with Hugging Face [smolagents](https://huggingface.co/docs/smolagents/en/index) tools.
	- `CodeEnv` for interactive Python execution.
	- `MultiTurnEnv` abstract class for implementing custom multi-turn rollout logic on top of vLLM's `chat()` method -- just override `env_response(list[dict]) -> dict` and `is_completed(list[dict]) -> bool` and you're good to go.
	- `Environment` abstract class for implementing whatever rollout logic you can imagine (go nuts!)

Basic usage for a `GRPOEnvTrainer` training script with 4 GPUs (2 inference + 2 training):

```bash
# launch inference server
CUDA_VISIBLE_DEVICES=0,1 uv run -m verifiers.inference.vllm_openai --model 'Qwen/Qwen2.5-7B-Instruct'

# launch training script; copy zero3.yaml or set values globally with `accelerate config`
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num-processes 2 --config-file configs/zero3.yaml train.py
```

The included `train.py` mirrors `verifiers/examples/reverse_text.py` and is configured for 4xH200 GPUs (2 inference + 2 training). See [GRPO Rules of Thumb](#grpo-rules-of-thumb) for further discussion of hyperparameters and best practices; the easiest way to reduce memory requirements is by reducing `per_device_train_batch_size` and increasing `gradient_accumulation_steps` accordingly.
### Citation

If you use this code in your research, please cite:

```bibtex
@article{brown2025verifiers,
  title={Verifiers: Reinforcement Learning with LLMs in Verifiable Environments},
  author={Brown, William},
  year={2025}
}
```

## Getting Started

### Setup 

`verifiers` is available via PyPI (TODO), and we recommend installation with `uv`:

```
uv venv --python 3.11 (or 3.12)
uv add verifiers # optionally, "verifiers[tools]"
uv pip install flash-attn --no-build-isolation
```


To use the latest `main` branch, do:
```bash
git clone https://github.com/willccbb/verifiers.git
cd verifiers
uv sync
uv pip install flash-attn --no-build-isolation
uv pip install -e ".[jupyter,tools,extras]"
```

**Troubleshooting:**
- Ensure your `wandb` and `huggingface-cli` logins are set up (or set `report_to=None` in `training_args`).
- On some setups, inter-GPU communication can [hang](https://github.com/huggingface/trl/issues/2923) during vLLM weight syncing. This can usually be alleviated by setting `NCCL_P2P_DISABLE=1` in your environment.
- Confirm that you are able to run standalone TRL GRPO training scripts (see `examples/trl_grpo.py`)
- If problems persist, please open an [issue](https://github.com/willccbb/verifiers/issues).
### Resource Requirements

`verifiers` currently uses `transformers` Trainer as its primary training backend via `accelerate` (like Hugging Face's [TRL](https://github.com/huggingface/trl/tree/main/trl)), and is optimized for setups with at least 2 GPUs, scaling up to multiple 8xH100 nodes. 2-GPU setups with sufficient memory to enable small-scale experimentation can be [rented](https://app.primeintellect.ai/dashboard/create-cluster?image=ubuntu_22_cuda_12) for <$1/hr.

Depending on your goals, there are other RL frameworks with native support for multi-turn tool use which you may be interested in exploring as well. If you are looking for maximum efficiency on a single GPU, consider OpenPipe's [ART](https://github.com/OpenPipe/ART) framework, which builds on top of [Unsloth](https://github.com/unslothai/unsloth). If you are seeking to maximize absolute performance at large scales, consider Nvidia's [NeMo-RL](https://github.com/NVIDIA/NeMo-RL) or ByteDance's [veRL](https://github.com/volcengine/verl) (which powers many agent RL projects like [RAGEN](https://github.com/RAGEN-AI/RAGEN) and [SkyRL](https://github.com/NovaSky-AI/SkyRL/tree/main)).

We aim to include support for additional trainers backends in the future, and are open to PRs. Our first-order objective is maintaining ease of use for users (and LLMs), and any potential contributions will be considered with this in mind. 

### Levels of Exploration
 
**Level 0:** Inspect and run the included examples for simple training tasks:
- `verifiers/examples/reverse_text.py`  (`SingleTurnEnv`)
- `verifiers/examples/
- `verifiers/examples/math_python.py` (`ToolEnv`)

**Level 1:** Implement your own reasoning task with verifiable rewards using `SingleTurnEnv`:

```python
import verifiers as vf
parser = vf.XMLParser(['think', 'answer']) # <think>...</think>\n<answer>...</answer>
rubric = vf.Rubric(
	your_custom_reward_func, # def func(prompts, completions, answer, **kwargs)
	parser.get_format_reward_func(),
weights=[1.0, 0.2])
vf_env = vf.SingleTurnEnv(
	dataset=..., # hf Dataset with 'question' + 'answer' columns
	system_prompt=f"... Respond in the following format: {parser.get_format_str()}",
	rubric
)
```

**Level 2:** Evaluate API models in your environment and collect synthetic data:

```python
import os
from openai import OpenAI

client = OpenAI(base_url="https://api.deepseek.com", api_key=os.getenv('DEEPSEEK_API_KEY'))

# evaluation
results = vf_env.evaluate(client, model="deepseek-chat", num_samples=100)
print(results['rewards_avg'])

# datasets
# columns = ['prompt', 'completion', 'answer', 'reward']
dataset_dsv3 = vf_env.make_dataset(results)
dataset_dsv3 = dataset_dsv3.sort("reward", reverse=True).select(range(50))
dataset_dsv3.push_to_hub("...")
```

**Level 2.5 (Optional, but recommended for <7B models):** SFT warmup on synthetic data

See `verifiers/examples/warmup/sft_reverse.py` for an example script using TRL's SFTTrainer.

**Level 3:** Train a model in your environment using GRPO:

```python
# train.py

model, tokenizer = vf.get_model_and_tokenizer(model_name)
trainer = vf.GRPOEnvTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=vf.grpo_defaults(run_name="...")
)
trainer.train()
```

**Level 4:** Implement your own multi-turn agent environment using `ToolEnv`, `SmolaToolEnv`, or `CodeEnv`:
```python
import verifiers as vf
vf_env = vf.ToolEnv(
	dataset=..., # hf Dataset with 'question' + 'answer' columns
	system_prompt=...,
	tools=[python, search, ask, calculator] # arbitrary python functions
	max_steps=5
)
```

**Level 5+:** Implement custom interaction protocols using `MultiTurnEnv` or `Environment`

```python

class YourMultiTurnEnv(MultiTurnEnv):
    def __init__(self,
                 dataset: Dataset | None,
                 system_prompt: str | None, 
                 parser: Parser | None,
                 rubric: Rubric | None,
                 **kwargs):
	
  def is_completed(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, str]:
    # return whether or not rollout is completed

  def env_response(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, str]:
    # return environment response for a message-dict sequence

class YourCustomEnv(Environment):

```

Ideas: 
- MCP Environments
- Multi-Player Game Environments
- Computer-use agents via tool-calling to a VLM
- Anything else you can think of!


### GRPO Rules of Thumb
- RL is [notoriously](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) sensitive to implementation details, and this applies to LLM GRPO as well. The default hyperparameter config in `vf.grpo_defaults()` is intended as a starting point which should be relatively stable for a broad variety of medium-difficulty tasks, informed by my own experimentation as well as broader community findings. Particularly for 
- Always start by evaluating the performance of your model and 
	- The `vf_env.eval_api()` and `vf_env.make_api_dataset()` methods provide convenient interfaces 
	- If your model struggles to get non-zero rewards in 10+ trials, the task is likely too hard. Consider adjusting your prompts
	- If a 
	- If your model already gets 80%+ on a task without training, the dataset is likely too easy. Consider prefiltering for questions 
- Tricks which may increase performance/speed, at the cost of risking "collapse":
	- Setting the KL penalty `beta = 0` (removes the reference model)
	- Increasing the learning rate
	- Increasing the number of update steps per batch (`num_iterations`)
- Tricks which may stabilize training, at the cost of speed/performance
	- Increasing group size per prompt (`num_generations`)
	- Increasing prompts per batch (`per_device_train_batch_size`, `gradient_accumulation_steps`)
	- Decreasing `max_grad_norm` (clipping)
	- Using larger models (14B+)
	- Using more num
	- Using LoRA adapters
- Tricks whose benefit remains up-for-debate or context-dependent:
	- High `beta` values (`0.2+`)
	- Dr. GRPO vs GRPO
	- Overlong filtering
	- Masking tool call responses (`mask_env_response` in  `MultiStepEnv`)
- Tricks which are likely a "free lunch":
	- Learning rate warm-up of at least 10-20 steps (`warmup_steps`)
	- Periodically updating reference models (`sync_ref_model`, `ref_model_sync_steps`) if using a reference model, particularly for 500+ step runs
- For successful training, you generally want diversity of reward scores within each group of responses for a prompt (see DAPO [paper](https://arxiv.org/pdf/2503.14476), Sec. 3.2).
- The *best* way to increase diversity is 
- See Hugging Face's [open-r1](https://huggingface.co/spaces/open-r1/README/discussions/20) logbook for lots of discussion, tips, and experimental findings.











