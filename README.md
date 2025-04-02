# Verifiers: Reinforcement Learning with LLMs in Verifiable Environments

This repository contains a set of tools for reinforcement learning with LLMs in verifiable environments.

**WARNING:** This repository in its current state should be viewed as **in-progress research code**, and is not guaranteed to yield stable or optimal training results. Best results will likely be found on reasonable timescales when using 7B+ models, and at least 8 GPUs.

**Note:** If you don't need multi-turn tool calling or multi-agent interactions, you should probably just use TRL (or Unsloth/Axolotl) for GRPO. This is mostly a multi-turn LLM RL repo with some other bells and whistles.


## Setup

PyPI [coming soon](https://pypi.org/project/verifiers/), for now just do:
```
git clone https://github.com/willccbb/verifiers.git
cd verifiers
uv sync
uv pip install flash-attn --no-build-isolation
source .venv/bin/activate
```
Ensure your `wandb` and `huggingface-cli` logins are set up (or set `report_to=None` in `training_args`).

If you encounter version issues, please confirm that you are able to run basic TRL training in your environment before opening an issue.

## Usage (Multi-GPU)

### Training with Multi-Turn GRPO

See `verifiers/examples/math_train.py` for an example with the ToolEnv environment + a Python tool.

To run on a 8-GPU node with 4 inference GPUs and 4 training GPUs:
```sh
# Launch vLLM inference server from verifiers/, with .venv active
CUDA_VISIBLE_DEVICES=0,1,2,3 python verifiers/inference/vllm_serve.py --model "Qwen/Qwen2.5-7B-Instruct" --tensor_parallel_size 4 --max_model_len 8192  --gpu_memory_utilization 0.9 --enable_prefix_caching True
```

```sh
# Run training script from verifiers/, with .venv active
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num-processes 4 --config-file configs/zero3.yaml verifiers/examples/math_train.py
```

Multi-node training setups are supported as well; you can specify the host IP + port of your inference as an argument in the `GRPOConfig` in your training script. See the TRL [docs](https://huggingface.co/docs/trl/main/en/grpo_trainer#trl.GRPOTrainer) for info on multi-node training via SLURM.

### Evaluation

You can also use environment classes to evaluate models with multi-turn tool use offline, i.e. without RL training. See `verifiers/examples/math_eval.py` for an example.

### Custom Environments

To create your own multi-turn environment, inherit from `MultiTurnEnv` and implement:
```python
def is_completed(self, messages: List[Dict[str, str]], **kwargs: Any) -> bool:
    pass

def env_response(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, str]:
    pass
```

## Features
- [X] Environments (`MultiTurnEnv`): `DoubleCheckEnv`, `CodeEnv`, `ToolEnv`
- [X] Multi-turn tool use in `CodeEnv` and `ToolEnv`
- [X] Dataset formatting + XML parsers
- [X] Basic rubrics for math/code correctness + formatting
- [X] Defaults for GRPO, model, tokenizer, etc.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{brown2025verifiers,
  title={Verifiers: Reinforcement Learning with LLMs in Verifiable Environments},
  author={Brown, William},
  year={2025}
}
```
