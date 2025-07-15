# Verifiers Documentation

Welcome to the Verifiers documentation! Verifiers is a flexible framework for reinforcement learning with large language models (LLMs). It provides a modular architecture for creating evaluation environments, parsing structured outputs, and training models using automated reward signals.

## Quick Start

Get started with Verifiers in minutes:

```python
import verifiers as vf

# Load a dataset
dataset = vf.load_example_dataset("gsm8k", split="train")

# Create a simple environment
vf_env = vf.SingleTurnEnv(
    dataset=dataset,
    system_prompt="Solve the problem step by step.",
    parser=vf.ThinkParser(),
    rubric=vf.Rubric(funcs=[correct_answer_func])
)

# Evaluate the environment
results = vf_env.evaluate(
    client=openai_client,
    model="gpt-4",
    num_examples=10
)
```

## Core Concepts

Verifiers is built around three fundamental primitives:

### 1. Environments
Manage the complete interaction flow between models, datasets, and evaluation.

- **SingleTurnEnv**: One-shot Q&A tasks (most common entry point)
- **MultiTurnEnv**: Interactive conversations and multi-step reasoning
- **ToolEnv**: Tasks requiring external tools
- **Custom Environments**: Write your own by extending base classes

### 2. Parsers
Extract structured information from model outputs.

- **XMLParser**: Convenience parser for XML-tagged fields
- **ThinkParser**: Convenience parser for step-by-step reasoning
- **Custom Parsers**: Write your own for specific output formats

### 3. Rubrics
Combine multiple reward functions to evaluate model outputs.

- **Base Rubric**: Combine multiple reward functions with weights
- **RubricGroup**: Aggregate multiple rubrics
- **Custom Rubrics**: Write your own for task-specific evaluation

## Documentation Sections

### Getting Started
- **[Overview](overview.md)**: Core concepts and architecture
- **[Examples](examples.md)**: Practical examples from real usage
- **[Environments](environments.md)**: Environment types and usage

### Core Components
- **[Parsers](parsers.md)**: Structured output extraction
- **[Rubrics](rubrics.md)**: Multi-criteria evaluation
- **[API Reference](api_reference.md)**: Complete API documentation

### Advanced Topics
- **[Tools](tools.md)**: Tool integration and usage
- **[Testing](testing.md)**: Testing environments and components
- **[Development](development.md)**: Contributing to the framework

## Key Design Principles

### 1. Start Simple
Most environments provide sensible defaults:

```python
# Simple - environment chooses defaults
vf_env = vf.SingleTurnEnv(dataset=dataset)

# Custom - specify your own components
vf_env = vf.SingleTurnEnv(
    dataset=dataset,
    parser=vf.ThinkParser(),
    rubric=custom_rubric
)
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

### 3. Incremental Complexity
Start with basic environments and add complexity as needed:

1. Begin with `SingleTurnEnv` for Q&A tasks
2. Use `MultiTurnEnv` for interactive tasks
3. Write custom parsers for specific output formats
4. Write custom rubrics for task-specific evaluation

## Environment Types

Different environment types are designed for different tasks:

| Environment | Use Case | Examples |
|-------------|----------|----------|
| **SingleTurnEnv** | One-shot Q&A | Math problems, classification |
| **MultiTurnEnv** | Interactive conversations | Multi-step reasoning, dialogue |
| **ToolEnv** | Need external tools | Code execution, calculations |
| **TextArenaEnv** | Games/simulations | Wordle, strategy games |
| **Custom** | Specific requirements | Write your own |

## Training and Evaluation

Environments are not just for training - they're also powerful evaluation tools:

```python
# Evaluate a model
results = vf_env.evaluate(
    client=openai_client,
    model="gpt-4",
    num_examples=100
)

# Generate training data
results = vf_env.generate(
    client=openai_client,
    model="gpt-4",
    n_samples=1000
)
```

The framework supports various training approaches:
- **GRPO Trainer**: Reinforcement learning with the environment
- **Verifiers**: Async FSDP environment training
- **Custom Training**: Use environments as reward functions in your own training loops

## Examples

See the [Examples](examples.md) section for complete working examples:

- **GSM8K Math**: Simple Q&A with step-by-step reasoning
- **Tool-Augmented Math**: Complex math with Python execution
- **Interactive Wordle**: Game-based training
- **Multi-Turn Wiki Search**: Interactive search with tools
- **Custom Parsers**: Writing parsers for specific formats
- **Custom Rubrics**: Writing rubrics for specific evaluation

## Best Practices

1. **Start Simple**: Begin with `SingleTurnEnv` and basic parsers
2. **Add Complexity**: Gradually add custom parsers and rubrics
3. **Include Format Rewards**: Always include `parser.get_format_reward_func()` in rubrics
4. **Test Thoroughly**: Test environments before large-scale training
5. **Document Format**: Clearly specify expected output format in system prompts
6. **Handle Errors**: Ensure parsers and rubrics handle malformed input gracefully

## Getting Help

- **Examples**: Check the [Examples](examples.md) section for working code
- **API Reference**: See [API Reference](api_reference.md) for complete documentation
- **Issues**: Report bugs and request features on GitHub
- **Discussions**: Ask questions and share ideas on GitHub Discussions

## Contributing

We welcome contributions! See the [Development](development.md) guide for:
- Setting up the development environment
- Running tests
- Contributing code
- Code style guidelines
