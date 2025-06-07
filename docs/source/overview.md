# Overview

Verifiers is a flexible framework for reinforcement learning with large language models (LLMs). It provides a modular architecture for creating evaluation environments, parsing structured outputs, and training models using automated reward signals.

## Core Concepts

The framework is built around three fundamental primitives that work together:

### 1. Parser: Structured Output Extraction

Parsers extract structured information from model outputs, and can expose useful functionalities for response composition. While you can use plain text or create your own parsers, **we generally recommend using XMLParser** for its convenience and reliability, particularly when getting started with `verifiers`.

```python
from verifiers.parsers import XMLParser

# Define expected fields
parser = XMLParser(fields=["think", "answer"])

# Parse model output
output = """
<think>
First, I need to calculate 2+2...
</reasoning>
<think>4</answer>
"""
parsed = parser.parse(output)
print(parsed.think)  # "First, I need to calculate 2+2..."
print(parsed.answer)     # "4"
```

### 2. Rubric: Multi-Criteria Evaluation

Rubrics combine multiple reward functions to evaluate model outputs from different perspectives. Each reward function can focus on a specific aspect:

```python
from verifiers.rubrics import Rubric

def correctness_reward(completion, answer, **kwargs):
    """Check if the answer is correct."""
    parsed = parser.parse(completion)
    return 1.0 if parsed.answer == answer else 0.0

def reasoning_reward(completion, **kwargs):
    """Reward clear reasoning steps."""
    parsed = parser.parse(completion)
    return min(len(parsed.reasoning.split('\n')) / 5, 1.0)

rubric = Rubric(
    funcs=[correctness_reward, reasoning_reward],
    weights=[1.0, 0.5],  # Correctness is more important
    parser=parser
)
```

### 3. Environment: Task Orchestration

Environments bring everything together, managing the interaction flow between models, parsers, and rubrics:

```python
from verifiers.envs import SingleTurnEnv

env = SingleTurnEnv(
    dataset=dataset,
    system_prompt="Solve the math problem step by step.",
    parser=parser,
    rubric=rubric,
    client=openai_client
)

# Generate training data
prompts, completions, rewards = env.generate(
    model="gpt-4",
    n_samples=1000
)
```

## Why This Architecture?

### Separation of Concerns
- **Parsing** handles format and structure
- **Evaluation** defines what makes a good response  
- **Orchestration** manages the interaction flow

### Composability
Build complex behaviors from simple components:
- Combine multiple reward functions in a rubric
- Chain multiple rubrics with RubricGroup
- Extend base environments for custom tasks

### Flexibility
- Support for both chat and completion APIs
- Synchronous and asynchronous execution
- Works with any OpenAI-compatible API

## Key Design Principles

### 1. Format First
Always define your output format explicitly. XMLParser makes this easy and reliable:

```python
# Good: Clear structure
parser = XMLParser(["reasoning", "answer"])

# Better: Support alternatives
parser = XMLParser([("reasoning", "thinking"), "answer"])

# Best: Add format validation
rubric = Rubric(
    funcs=[parser.get_format_reward_func(), correctness_func],
    weights=[0.2, 0.8]
)
```

### 2. Multi-Criteria Evaluation
Real-world tasks have multiple success criteria. Design rubrics that capture all important aspects:

```python
rubric = Rubric(funcs=[
    correct_answer_func,      # Is it right?
    reasoning_clarity_func,   # Is it well-explained?
    efficiency_func,         # Is it concise?
    format_compliance_func   # Does it follow instructions?
])
```

### 3. Incremental Complexity
Start simple and add complexity as needed:

1. Begin with SingleTurnEnv for basic Q&A
2. Add custom reward functions for your criteria
3. Move to MultiTurnEnv for interactive tasks
4. Integrate tools when reasoning alone isn't enough

## Quick Start Example

Here's a complete example for a math problem-solving task:

```python
from verifiers.envs import SingleTurnEnv
from verifiers.parsers import XMLParser
from verifiers.rubrics import Rubric
import openai

# 1. Setup parser for structured output
parser = XMLParser(fields=["reasoning", "answer"])

# 2. Define evaluation criteria
def correct_answer(completion, answer, **kwargs):
    parsed = parser.parse(completion)
    return 1.0 if parsed.answer.strip() == answer.strip() else 0.0

def has_reasoning(completion, **kwargs):
    parsed = parser.parse(completion)
    return 1.0 if len(parsed.reasoning) > 20 else 0.0

# 3. Create rubric with multiple criteria
rubric = Rubric(
    funcs=[correct_answer, has_reasoning, parser.get_format_reward_func()],
    weights=[0.7, 0.2, 0.1],
    parser=parser
)

# 4. Setup environment
env = SingleTurnEnv(
    dataset=math_dataset,
    system_prompt="""Solve the given math problem step by step.
    
Format your response as:
<reasoning>
Your step-by-step solution
</reasoning>
<answer>
Your final answer
</answer>""",
    parser=parser,
    rubric=rubric,
    client=openai.Client()
)

# 5. Generate training data
results = env.generate(
    model="gpt-4",
    n_samples=100,
    temperature=0.7
)
```

## Next Steps

- [**Environments**](environments.md): Learn about different environment types
- [**Parsers**](parsers.md): Master structured output parsing
- [**Rubrics**](rubrics.md): Design sophisticated evaluation criteria
- [**Examples**](examples.md): Walk through real-world implementations
- [**Training**](training.md): Train models with GRPO