# Overview

Verifiers is a flexible framework for reinforcement learning with large language models (LLMs). It provides a modular architecture for creating evaluation environments, parsing structured outputs, and training models using automated reward signals.

## Core Concepts

The framework is built around three fundamental primitives that work together:

### 1. Environment: Task Orchestration

Environments manage the complete interaction flow between models, datasets, and evaluation. **Most users should start with `SingleTurnEnv` or `MultiTurnEnv`**:

```python
import verifiers as vf

# SingleTurnEnv: One-shot Q&A tasks (most common)
vf_env = vf.SingleTurnEnv(
    dataset=dataset,
    system_prompt="Solve the problem step by step.",
    parser=parser,
    rubric=rubric
)

# MultiTurnEnv: Interactive conversations
class MyMultiTurnEnv(vf.MultiTurnEnv):
    def is_completed(self, messages, state, **kwargs):
        # Define when conversation should end
        return len(state['responses']) >= 5
    
    def env_response(self, messages, state, **kwargs):
        # Define environment's response logic
        return {"role": "user", "content": "Continue..."}, state
```

**SingleTurnEnv** is perfect for:
- Question-answer tasks
- Classification problems  
- Translation tasks
- Any task with clear input-output structure

**MultiTurnEnv** is ideal for:
- Interactive conversations
- Multi-step reasoning
- Tool-augmented tasks
- Games and simulations

### 2. Parser: Structured Output Extraction

Parsers extract structured information from model outputs. The base `Parser` class simply returns text as-is, but specialized parsers can extract specific formats.

```python
import verifiers as vf

# Base parser (returns text as-is)
parser = vf.Parser()

# ThinkParser extracts content after </think> tags
parser = vf.ThinkParser()

# XMLParser extracts structured fields (recommended for most use cases)
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

**For nontrivial environments, users will want to write their own parsers** to handle specific output formats and requirements.

### 3. Rubric: Multi-Criteria Evaluation

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

**For nontrivial environments, users will want to write their own rubrics** to define task-specific evaluation criteria.

## Why This Architecture?

### Separation of Concerns
- **Environments** handle task orchestration and interaction flow
- **Parsers** handle format and structure extraction
- **Rubrics** define what makes a good response  

### Composability
Build complex behaviors from simple components:
- Combine multiple reward functions in a rubric
- Use built-in parsers like `XMLParser`, `ThinkParser` for common cases
- Choose from different environment types for different tasks
- Write custom parsers and rubrics for specific needs

### Flexibility
- Support for both chat and completion APIs
- Synchronous and asynchronous execution
- Works with any OpenAI-compatible API
- Environments are also evaluations - not just for training

## Key Design Principles

### 1. Start Simple
Most environments provide sensible defaults:

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
2. Use `MultiTurnEnv` for interactive tasks
3. Write custom parsers for specific output formats
4. Write custom rubrics for task-specific evaluation

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

# 4. Evaluate the environment
results = vf_env.evaluate(
    client=openai_client,
    model="gpt-4",
    num_examples=10
)
print(results)
```

## Environment Types

Different environment types are designed for different tasks:

- **SingleTurnEnv**: One-shot Q&A tasks (most common)
- **MultiTurnEnv**: Interactive conversations and multi-step reasoning
- **ToolEnv**: Tasks requiring external tools (calculators, code execution)
- **TextArenaEnv**: Game-based environments
- **Custom Environments**: Write your own by extending `Environment` or `MultiTurnEnv`

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