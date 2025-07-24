import argparse
from pathlib import Path

ENDPOINTS_TEMPLATE = """\
ENDPOINTS = {
    "my-model": {
        "model": "my-model",
        "url": "https://some-endpoint.com/v1",
        "key": "SOME_API_KEY",
    },
}
"""

README_TEMPLATE = """\
# Local Workspace for Verifiers

This folder serves as a local workspace for implementing custom RL environments and configuring LLM endpoints for testing via `vf-eval`. Use the `local/endpoints.py` 

## `local/endpoints.py`

```python
# local/endpoints.py
ENDPOINTS = {
    "my-model": {
        "model": "my-model",
        "url": "https://some-endpoint.com/v1",
        "key": "SOME_API_KEY",
    },
    ...
}
```

## `local/registry`

```python
# local/registry/my_environment.py -- exposes `load_environment` function

def load_environment(env_arg1, env_arg2, ...) -> vf.Environment
    # your environment 

    return vf_env
```

```sh
uv run vf-eval -e my-environment -m my-model
```
"""

TRAIN_TEMPLATE = """\
import verifiers as vf

model_id = "my-model"
env_id = "my-environment"
model, tokenizer = vf.get_model_and_tokenizer(model_id)
vf_env = vf.load_environment(env_id)
args = vf.grpo_defaults(run_name=f"{env_id}--{model_id.split('/')[-1]}")
trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=args,
)

trainer.train()
"""

EXAMPLE_ENVIRONMENT_TEMPLATE = """\
import verifiers as vf
from verifiers.utils.data_utils import extract_boxed_answer, load_example_dataset


def load_environment(num_train_examples=-1, num_eval_examples=-1):
    dataset = load_example_dataset("gsm8k", split="train")
    if num_train_examples != -1:
        dataset = dataset.select(range(num_train_examples))
    eval_dataset = load_example_dataset("gsm8k", split="test")
    if num_eval_examples != -1:
        eval_dataset = eval_dataset.select(range(num_eval_examples))

    system_prompt = \"\"\"
    Think step-by-step inside <think>...</think> tags.

    Then, give your final numerical answer inside \\boxed{{...}}.
    \"\"\"
    parser = vf.ThinkParser(extract_fn=extract_boxed_answer)

    def correct_answer_reward_func(completion, answer, **kwargs):
        response = parser.parse_answer(completion) or ""
        return 1.0 if response == answer else 0.0

    rubric = vf.Rubric(
        funcs=[correct_answer_reward_func, parser.get_format_reward_func()],
        weights=[1.0, 0.2],
    )

    vf_env = vf.SingleTurnEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
    )
    return vf_env
"""

MY_ENVIRONMENT_TEMPLATE = """\
import verifiers as vf


def load_environment(**kwargs) -> vf.Environment:
    \"\"\"
    Loads a custom environment.
    \"\"\"
    raise NotImplementedError("Implement your custom environment here.")
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dir-name", "-d", type=str, default="local")
    args = parser.parse_args()

    # get cwd
    current_dir = Path.cwd()

    # make local/registry/ if it doesn't exist
    local_dir = current_dir / args.local_dir_name
    registry_dir = local_dir / "registry"
    registry_dir.mkdir(parents=True, exist_ok=True)

    # create README.md if it doesn't exist
    readme_file = local_dir / "README.md"
    if not readme_file.exists():
        readme_file.write_text(README_TEMPLATE)
    else:
        print(f"README.md already exists at {readme_file}, skipping...")

    # create __init__.py in local/ and local/registry/ if they don't exist
    init_file = local_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text("")
    else:
        print(f"__init__.py already exists at {init_file}, skipping...")

    registry_init_file = registry_dir / "__init__.py"
    if not registry_init_file.exists():
        registry_init_file.write_text("")
    else:
        print(f"__init__.py already exists at {registry_init_file}, skipping...")

    # create endpoints.py if it doesn't exist
    endpoints_file = local_dir / "endpoints.py"
    if not endpoints_file.exists():
        endpoints_file.write_text(ENDPOINTS_TEMPLATE)
    else:
        print(f"endpoints.py already exists at {endpoints_file}, skipping...")

    # create example environment.py if it doesn't exist
    example_environment_file = registry_dir / "example.py"
    if not example_environment_file.exists():
        example_environment_file.write_text(EXAMPLE_ENVIRONMENT_TEMPLATE)
    else:
        print(f"example.py already exists at {example_environment_file}, skipping...")

    # create train.py if it doesn't exist
    train_file = local_dir / "train.py"
    if not train_file.exists():
        train_file.write_text(TRAIN_TEMPLATE)
    else:
        print(f"train.py already exists at {train_file}, skipping...")

    # create my_environment.py if it doesn't exist
    my_environment_file = registry_dir / "my_environment.py"
    if not my_environment_file.exists():
        my_environment_file.write_text(MY_ENVIRONMENT_TEMPLATE)
    else:
        print(f"my_environment.py already exists at {my_environment_file}, skipping...")

    return local_dir


if __name__ == "__main__":
    main()
