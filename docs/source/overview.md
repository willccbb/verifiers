# Overview

Verifiers is a flexible framework for reinforcement learning with large language models (LLMs). It provides a modular architecture for creating evaluation environments, parsing structured outputs, and training models using automated reward signals.

## Core Concepts

The framework is built around three fundamental primitives that work together:

### 1. Parser: Structured Output Extraction

Parsers extract structured information from model outputs. The base `Parser` class simply returns text as-is, but specialized parsers can extract specific formats.

```python
import verifiers as vf

# Base parser (returns text as-is)
parser = vf.Parser()

# ThinkParser extracts content after </think> tags
parser = vf.ThinkParser()

# XMLParser extracts structured fields
parser = vf.XMLParser(fields=["reasoning", "answer"])
```

**ThinkParser** is useful for step-by-step reasoning:
```python
parser = vf.ThinkParser()

# Model output: "<think>Let me calculate...</think>The answer is 4"
# parser.parse_answer(output) returns: "The answer is 4"
```

**XMLParser** is useful for structured output with multiple fields:
```python
parser = vf.XMLParser(fields=["reasoning", "answer"])

# Model output: "<reasoning>First, I calculate...</reasoning><answer>4</answer>"
parsed = parser.parse(output)
print(parsed.reasoning)  # "First, I calculate..."
print(parsed.answer)     # "4"
```

### 2. Rubric: Multi-Criteria Evaluation

Rubrics combine multiple reward functions to evaluate model outputs. The base `Rubric` class takes a list of functions and weights.

```python
import verifiers as vf

# Basic rubric with one reward function
def correct_answer(completion, answer, **kwargs):
    response = parser.parse_answer(completion) or ''
    return 1.0 if response.strip() == answer.strip() else 0.0

rubric = vf.Rubric(
    funcs=[correct_answer],
    weights=[1.0]
)
```

**Multi-criteria evaluation** combines different aspects:
```python
def correctness_reward(completion, answer, **kwargs):
    response = parser.parse_answer(completion) or ''
    return 1.0 if response.strip() == answer.strip() else 0.0

def format_reward(completion, **kwargs):
    # Check if response follows expected format
    return parser.get_format_reward_func()(completion)

rubric = vf.Rubric(
    funcs=[correctness_reward, format_reward],
    weights=[0.8, 0.2]  # Correctness weighted higher
)
```

### 3. Environment: Task Orchestration

Environments bring everything together, managing the interaction flow between models, parsers, and rubrics:

```python
import verifiers as vf

dataset = vf.load_example_dataset("gsm8k", split="train")

vf_env = vf.SingleTurnEnv(
    dataset=dataset,
    system_prompt="Solve the problem step by step.",
    parser=parser,
    rubric=rubric
)

# Generate training data
model, tokenizer = vf.get_model_and_tokenizer("Qwen/Qwen2.5-1.5B-Instruct")
args = vf.grpo_defaults(run_name="example")
trainer = vf.GRPOTrainer(model=model, processing_class=tokenizer, env=vf_env, args=args)
```

## Why This Architecture?

### Separation of Concerns
- **Parsing** handles format and structure
- **Evaluation** defines what makes a good response  
- **Orchestration** manages the interaction flow

### Composability
Build complex behaviors from simple components:
- Combine multiple reward functions in a rubric
- Use built-in rubric types like `MathRubric`, `ToolRubric`
- Choose from different environment types for different tasks

### Flexibility
- Support for both chat and completion APIs
- Synchronous and asynchronous execution
- Works with any OpenAI-compatible API

## Key Design Principles

### 1. Start Simple
Many environments provide sensible defaults:

```python
# Simple - let environment choose defaults
vf_env = vf.SingleTurnEnv(dataset=dataset)

# Custom - specify your own parser and rubric
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
2. Use `ToolEnv` when models need external tools
3. Try `DoubleCheckEnv` for verification workflows
4. Use `MultiTurnEnv` for interactive tasks

## Quick Start Example

Here's a complete working example:

```python
import verifiers as vf

# 1. Load dataset
dataset = vf.load_example_dataset("gsm8k", split="train")

# 2. Setup parser and rubric
parser = vf.ThinkParser()

def correct_answer(completion, answer, **kwargs):
    response = parser.parse_answer(completion) or ''
    return 1.0 if response.strip() == answer.strip() else 0.0

rubric = vf.Rubric(
    funcs=[correct_answer, parser.get_format_reward_func()],
    weights=[0.8, 0.2]
)

# 3. Create environment
vf_env = vf.SingleTurnEnv(
    dataset=dataset,
    system_prompt="Think step-by-step inside <think>...</think> tags, then give your answer.",
    parser=parser,
    rubric=rubric
)

# 4. Setup and run training
model, tokenizer = vf.get_model_and_tokenizer("Qwen/Qwen2.5-1.5B-Instruct")
args = vf.grpo_defaults(run_name="quick-start")
trainer = vf.GRPOTrainer(model=model, processing_class=tokenizer, env=vf_env, args=args)
trainer.train()
```

## Environment Types

Different environment types are designed for different tasks:

- **SingleTurnEnv**: One-shot Q&A tasks
- **ToolEnv**: Tasks requiring external tools (calculators, code execution)
- **DoubleCheckEnv**: Multi-stage verification workflows
- **TextArenaEnv**: Game-based environments
- **ReasoningGymEnv**: Integration with reasoning gym benchmarks

## Next Steps

- [**Environments**](environments.md): Learn about different environment types
- [**Parsers**](parsers.md): Master structured output parsing
- [**Rubrics**](rubrics.md): Design sophisticated evaluation criteria
- [**Examples**](examples.md): Walk through real-world implementations
- [**Training**](training.md): Train models with GRPO