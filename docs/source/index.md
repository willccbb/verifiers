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
# Install the package
pip install verifiers

# Or using uv
uv add verifiers
```

### Basic Example

```python
from verifiers.envs import SingleTurnEnv
from verifiers.parsers import XMLParser
from verifiers.rubrics import Rubric
import openai

# 1. Define output format
parser = XMLParser(fields=["think", "answer"])

# 2. Create evaluation criteria
def correct_answer(completion, answer, **kwargs):
    parsed = parser.parse(completion)
    return 1.0 if parsed.answer == answer else 0.0

rubric = Rubric(
    funcs=[correct_answer, parser.get_format_reward_func()],
    weights=[0.8, 0.2]
)

# 3. Set up environment
env = SingleTurnEnv(
    dataset=your_dataset, # with 'question' and 'answer' string columns
    parser=parser,
    rubric=rubric,
    system_prompt="Solve the problem step by step.",
    client=openai.Client()
)

# 4. Generate training data
prompts, completions, rewards = env.generate(
    model="gpt-4",
    n_samples=100
)
```

## Key Concepts

### 1. **Environments** orchestrate the evaluation process
- Handle dataset management and prompt formatting
- Execute model interactions (rollouts)
- Integrate parsers and rubrics for complete evaluation

### 2. **Parsers** extract structured information
- XMLParser (recommended) for reliable field extraction
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