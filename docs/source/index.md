# Verifiers

**Note: These docs are mostly written by Claude 4 Opus. Many things are wrong. This is here as a starting point for me to work from, and they will be gradually improved over time.**

Verifiers is a flexible framework for reinforcement learning with large language models. It provides modular components for creating evaluation environments, parsing structured outputs, and training models using automated reward signals.

## What does it do?

Verifiers enables you to:

- **Build custom evaluation environments** for any task
- **Define multi-criteria reward functions** for nuanced evaluation  
- **Train models using GRPO** (Group Relative Policy Optimization)
- **Parse structured outputs** reliably with built-in validation
- **Integrate tools** to extend model capabilities

The framework emphasizes modularity and composability, allowing you to start simple and progressively add complexity as needed.

## Documentation

```{toctree}
:maxdepth: 2
:caption: Getting Started:

overview
examples
```

```{toctree}
:maxdepth: 2
:caption: Core Components:

environments
parsers
rubrics
tools
```

```{toctree}
:maxdepth: 2
:caption: Training & Advanced:

training
advanced
```

```{toctree}
:maxdepth: 2
:caption: Reference:

api_reference
testing
development
```

## Quick Start

### Installation

```bash
# Install using uv (recommended)
uv add verifiers

# Or clone and install locally
git clone https://github.com/your-org/verifiers
cd verifiers
uv sync
```

### Basic Example

Here's a complete example for training a model on math problems:

```python
import verifiers as vf

# Load dataset and create environment
dataset = vf.load_example_dataset("gsm8k", split="train")
eval_dataset = vf.load_example_dataset("gsm8k", split="test")

system_prompt = """
Think step-by-step inside <think>...</think> tags.

Then, give your final numerical answer inside \\boxed{{...}}.
"""

parser = vf.ThinkParser(extract_fn=vf.extract_boxed_answer)

def correct_answer_reward_func(completion, answer, **kwargs):
    response = parser.parse_answer(completion) or ''
    return 1.0 if response == answer else 0.0

rubric = vf.Rubric(funcs=[
    correct_answer_reward_func,
    parser.get_format_reward_func()
], weights=[1.0, 0.2])

vf_env = vf.SingleTurnEnv(
    dataset=dataset,
    eval_dataset=eval_dataset,
    system_prompt=system_prompt,
    parser=parser,
    rubric=rubric,
)

# Load model and set up training
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
model, tokenizer = vf.get_model_and_tokenizer(model_name)

training_args = vf.grpo_defaults(run_name="gsm8k-example")
training_args.per_device_train_batch_size = 4
training_args.num_generations = 8
training_args.max_steps = 100

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=training_args,
)

# Train the model
trainer.train()
```

## Key Concepts

### 1. **Environments** orchestrate the evaluation process
- Handle dataset management and prompt formatting
- Execute model interactions (rollouts)
- Integrate parsers and rubrics for complete evaluation

### 2. **Parsers** extract structured information
- ThinkParser (recommended) for step-by-step reasoning
- Built-in format validation and rewards
- Support for alternative field names

### 3. **Rubrics** define evaluation criteria
- Combine multiple reward functions with weights
- Support task-specific and general evaluations
- Enable sophisticated multi-aspect scoring

### 4. **Tools** extend model capabilities
- Simple Python functions with clear signatures
- Automatic discovery and schema generation
- Safe execution with error handling

## Why Verifiers?

- **Modular Design**: Mix and match components for your use case
- **Production Ready**: Used for training state-of-the-art models
- **Efficient**: Async operations and batch processing built-in
- **Flexible**: Support for any OpenAI-compatible API
- **Extensible**: Easy to add custom environments, parsers, and rubrics

## Learn More

- Start with the [Overview](overview.md) for core concepts
- Walk through [Examples](examples.md) to see real implementations
- Deep dive into [Environments](environments.md), [Parsers](parsers.md), and [Rubrics](rubrics.md)
- Learn about [Training](training.md) models with GRPO
- Explore [Advanced](advanced.md) patterns for complex use cases