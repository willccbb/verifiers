# Verifiers Documentation

Welcome to the Verifiers documentation! Verifiers is a library of modular components for creating RL environments and training LLM agents. Verifiers includes an async GRPO implementation built around the `transformers` Trainer, is supported by `prime-rl` for large-scale FSDP training, and can easily be integrated into any RL framework which exposes an OpenAI-compatible inference client.

## Quick Start

Get started with Verifiers in minutes:

```python
import verifiers as vf
from openai import OpenAI

# Load an environment module
vf_env = vf.load_environment("math-python", dataset_name="math", num_train_examples=1000)

# Or create a simple environment directly
dataset = vf.load_example_dataset("gsm8k", split="train")
vf_env = vf.SingleTurnEnv(
    dataset=dataset,
    system_prompt="Solve the problem step by step.",
    parser=vf.ThinkParser(),
    rubric=vf.Rubric(funcs=[correct_answer_func])
)

# Evaluate the environment
client = OpenAI()
results = vf_env.evaluate(client=client, model="gpt-4.1-mini", num_examples=10)
```

## Core Concepts

Verifiers is built around modular components that work together:

### 1. Environments
Environments are installable Python modules which expose a `load_environment` function for instantiation. They manage the complete interaction flow between models, datasets, and evaluation.

- **SingleTurnEnv**: One-shot Q&A tasks (most common entry point)
- **MultiTurnEnv**: Interactive conversations and multi-step reasoning
- **ToolEnv**: Tasks requiring external tools
- **Environment Modules**: Installable packages in the `environments/` folder

### 2. Parsers
Extract structured information from model outputs.

- **Parser**: Base parser (returns text as-is)
- **ThinkParser**: Extract content after `</think>` tags
- **XMLParser**: Extract structured XML fields
- **Custom Parsers**: Write your own for specific output formats

### 3. Rubrics
Combine multiple reward functions to evaluate model outputs.

- **Rubric**: Combine multiple reward functions with weights
- **JudgeRubric**: Use LLM judges for evaluation
- **RubricGroup**: Aggregate multiple rubrics
- **ToolRubric**: Count tool invocations

## Environment Management

### Installing Environments

Initialize a new environment module:
```bash
vf-init my-new-environment  # Creates template in ./environments/
```

Install a local environment module:
```bash
vf-install my-new-environment  # From ./environments/
```

Install from the verifiers repository:
```bash
vf-install math-python --from-repo  # From GitHub repo
```

### Quick Evaluation

Test any environment with an API model:
```bash
vf-eval math-python -m gpt-4.1-mini -n 5 -r 3
```

## Documentation Sections

### Getting Started
- **[Overview](overview.md)**: Core concepts and architecture
- **[Environments](environments.md)**: Environment types and usage patterns
- **[Examples](examples.md)**: Practical examples from real usage

### Core Components
- **[Parsers](parsers.md)**: Structured output extraction
- **[Rubrics](rubrics.md)**: Multi-criteria evaluation
- **[Tools](tools.md)**: Tool integration and usage

### Advanced Topics
- **[Advanced](advanced.md)**: Custom patterns and complex workflows
- **[Training](training.md)**: GRPO training and prime-rl integration
- **[Testing](testing.md)**: Testing environments and components
- **[Development](development.md)**: Contributing to the framework
- **[API Reference](api_reference.md)**: Complete API documentation

## Key Design Principles

### 1. Environment-First Design
Most users should start by loading or creating environments:

```python
# Load an existing environment module
vf_env = vf.load_environment("wordle", use_think=True)

# Or create directly for simple cases
vf_env = vf.SingleTurnEnv(dataset=dataset, rubric=rubric)
```

### 2. Multi-Criteria Evaluation
Real-world tasks have multiple success criteria:

```python
rubric = vf.Rubric(funcs=[
    correct_answer_func,      # Is it right?
    reasoning_clarity_func,   # Is it well-explained?
    format_compliance_func    # Does it follow instructions?
], weights=[0.7, 0.2, 0.1])
```

### 3. Modular Components
Build complexity from simple, reusable parts:

```python
# Combine parsers, rubrics, and environments
parser = vf.XMLParser(fields=["reasoning", "answer"])
rubric = vf.Rubric(funcs=[correctness_func, parser.get_format_reward_func()])
vf_env = vf.SingleTurnEnv(dataset=dataset, parser=parser, rubric=rubric)
```

## Environment Types

Different environment types are designed for different tasks:

| Environment | Use Case | Examples |
|-------------|----------|----------|
| **SingleTurnEnv** | One-shot Q&A | Math problems, classification |
| **MultiTurnEnv** | Interactive conversations | Multi-step reasoning, dialogue |
| **ToolEnv** | External tools needed | Code execution, calculations |
| **EnvGroup** | Multiple task types | Mixtures of environments |
| **Custom Modules** | Specific requirements | Installable environment packages |

## Training and Evaluation

Environments support both evaluation and training:

```python
# Evaluate a model
results = vf_env.evaluate(
    client=OpenAI(),
    model="gpt-4.1-mini",
    num_examples=100,
    rollouts_per_example=3
)

# For training, we recommend prime-rl for large-scale FSDP training
# or use the included GRPOTrainer for smaller setups
```

## Examples

See the [Examples](examples.md) section for complete working examples:

- **Math Problem Solving**: Using tools and step-by-step reasoning
- **Interactive Games**: Wordle and other game environments
- **Tool-Augmented Tasks**: Python execution and web search
- **Custom Environment Modules**: Building installable environments
- **Multi-Criteria Evaluation**: Complex rubric patterns

## Best Practices

1. **Start with Environment Modules**: Use `vf.load_environment()` for reusable patterns
2. **Test Before Training**: Use `vf-eval` for quick environment testing
3. **Include Format Rewards**: Always include `parser.get_format_reward_func()` in rubrics
4. **Use Chat Format**: Prefer `message_type="chat"` for most applications
5. **Handle Errors Gracefully**: Ensure parsers and rubrics handle malformed input

## Getting Help

- **Examples**: Check the [Examples](examples.md) section for working code
- **Environment Gallery**: See `environments/` folder for canonical patterns
- **API Reference**: See [API Reference](api_reference.md) for complete documentation
- **Issues**: Report bugs and request features on GitHub
- **Discussions**: Ask questions and share ideas on GitHub Discussions

## Contributing

We welcome contributions! See the [Development](development.md) guide for:
- Setting up the development environment
- Creating new environment modules
- Contributing to the core library
- Code style guidelines

The core `verifiers/` library is intended to be a lightweight set of reusable components. For applications of verifiers (e.g. "an Environment for XYZ task"), consider creating a self-contained module within `environments/` that can serve as a canonical example.
