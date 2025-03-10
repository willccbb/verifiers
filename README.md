# Verifiers: Reinforcement Learning with LLMs in Verifiable Environments

This repository contains a set of tools for reinforcement learning with LLMs in verifiable environments. 

**Note:** This repository in its current state should be viewed as "research code", and is not guaranteed to yield optimal training results. RL is delicate, expect that experimentation will be required. The examples are intended for illustrative purposes of usage patterns rather than stable training recipes. You are encouraged to write your own standalone training scripts, modifying environments/datasets/rewards/configs as needed for your use case.


## Installation

PyPI [coming soon](https://pypi.org/project/verifiers/) once a couple more features are added, just clone it for now and run:
```
git clone https://github.com/willccbb/verifiers.git
cd verifiers
uv sync
uv pip install flash-attn --no-build-isolation
source .venv/bin/activate
accelerate launch --config-file configs/zero3.yaml --num-processes [N-1] verifiers/examples/gsm8k_calculator.py
```


Ensure your `wandb` and `huggingface-cli` logins are set up (or set `report_to=None` in `training_args`).

Tested with Python 3.11 and this [image](https://hub.docker.com/layers/pytorch/pytorch/2.5.1-cuda12.1-cudnn9-devel/images/sha256-e8e63dd7baca894ba11fe1ba48a52a550793c8974f89b533d697784dd20a4dc0). If you encounter version issues, please confirm that you are able to run basic TRL training in your environment before opening an issue. `flash-attn` and `liger-kernel` are used for performance reasons. Recommended usage is via `accelerate` with DeepSpeed ZeRO 3 ([example config](https://github.com/huggingface/trl/blob/main/examples/accelerate_configs/deepspeed_zero3.yaml)) but `torchrun` works in my tests as well. You should really be using `uv` (`curl -LsSf https://astral.sh/uv/install.sh | sh`). I don't have the bandwidth to help debug your version issues if you're using `pip`, sorry.

## Usage

```python
# script.py
import verifiers as vf
from verifiers.tools import calculator
from verifiers.prompts import CALCULATOR_FEW_SHOT

model_name = "Qwen/Qwen2.5-7B-Instruct"
model, tokenizer = vf.get_model_and_tokenizer(model_name)

vf_env = vf.ToolEnv(
    dataset="gsm8k",
    few_shot=CALCULATOR_FEW_SHOT[0],
    tools=[calculator],
    max_steps=3
)
trainer = vf.GRPOEnvTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    reward_funcs=vf_env.get_rubric(),
    args=vf.get_default_grpo_config(run_name="gsm8k-calc", num_gpus=2),
    train_dataset=vf_env.get_dataset(),
)
trainer.train()
```
See `examples` for additional usage examples. 

To create your own multi-step environment, inherit from `MultiStepEnv` and implement:
```python
def get_dataset(self, **kwargs: Any) -> Dataset:
    pass

def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
    pass

def is_completed(self, messages: List[Dict[str, str]], **kwargs: Any) -> bool:
    pass

def env_response(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, str]:
    pass
```

### Launch Commands
Accelerate:
```bash
accelerate launch --config_file /path/to/deepspeed_zero3.yaml --num_processes [N-1] script.py
```
Torchrun:
```bash
torchrun --nproc_per_node=[N-1] script.py
```

## Features
- [X] Environments: `SimpleEnv`, `MathEnv`, `DoubleCheckEnv`, `CodeEnv`, `ToolEnv`
- [X] Multi-step execution in `CodeEnv` and `ToolEnv`
- [X] Dataset formatting + XML parsers
- [X] Basic ubrics for math/code correctness + formatting
- [X] Defaults for GRPO, model, tokenizer, etc.

## Roadmap

There are a number of features we're planning to support in the near future:
- [ ] Integrated evals
- [ ] TextArena games
- [ ] LLM judges
- [ ] Claude-generated rubrics
- [ ] A range of other environments (suggestions welcome!)
- [ ] PPO
- [ ] Potential interoperability with other RL libraries (veRL, OpenRLHF, open-instruct, oat, etc.)

Community contributions are appreciated and encouraged!

## Citation

If you use this code in your research, please cite:

```bibtex
@article{brown2025verifiers,
  title={Verifiers: Reinforcement Learning with LLMs in Verifiable Environments},
  author={Brown, William},
  year={2025}
}
```
