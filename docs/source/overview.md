# Overview

Verifiers is a library of modular components for creating RL environments and training LLM agents. It provides a modular architecture for creating evaluation environments, parsing structured outputs, and training models using automated reward signals.

## Core Concepts

The framework is built around modular components that work together:

### 1. Environment: Task Orchestration

Environments are installable Python modules which expose a `load_environment` function for instantiation. They manage the complete interaction flow between models, datasets, and evaluation. **Most users should start with environment modules or `SingleTurnEnv`**:

```python
import verifiers as vf
from openai import OpenAI

# Load an environment module (recommended)
vf_env = vf.load_environment("math-python", dataset_name="math", num_train_examples=1000)

# Or create directly for simple cases
dataset = vf.load_example_dataset("gsm8k", split="train")
vf_env = vf.SingleTurnEnv(
    dataset=dataset,
    system_prompt="Solve the problem step by step.",
    parser=vf.ThinkParser(),
    rubric=vf.Rubric(funcs=[correct_answer_func])
)

# Evaluate with any OpenAI-compatible client
client = OpenAI()
results = vf_env.evaluate(client=client, model="gpt-4.1-mini", num_examples=10)
```

**Environment modules** provide complete, reusable patterns:
- Pre-configured datasets, parsers, and rubrics
- Task-specific system prompts and reward functions
- Dependencies specified in `pyproject.toml`
- Examples: `math-python`, `wordle`, `gsm8k`

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

# XMLParser extracts structured fields (recommended for multi-field output)
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
def correct_answer(parser, completion, answer, **kwargs) -> float:
    response = parser.parse_answer(completion) or ''
    return 1.0 if response.strip() == answer.strip() else 0.0

rubric = vf.Rubric(
    funcs=[correct_answer],
    weights=[1.0],
    parser=parser
)
```

**Multi-criteria evaluation** combines different aspects:
```python
def correctness_reward(parser, completion, answer, **kwargs) -> float:
    response = parser.parse_answer(completion) or ''
    return 1.0 if response.strip() == answer.strip() else 0.0

def format_reward(parser, completion, **kwargs) -> float:
    # Check if response follows expected format
    return parser.get_format_reward_func()(completion)

rubric = vf.Rubric(
    funcs=[correctness_reward, format_reward],
    weights=[0.8, 0.2],  # Correctness weighted higher
    parser=parser
)
```

**For nontrivial environments, users will want to write their own rubrics** to define task-specific evaluation criteria.

## Environment Loading Pattern

The primary way to use environments is through installable modules:

```python
import verifiers as vf

# Load environment modules with configuration
vf_env = vf.load_environment("math-python", dataset_name="math", num_train_examples=5000)
vf_env = vf.load_environment("wordle", use_think=True, num_train_examples=2000)
vf_env = vf.load_environment("gsm8k", split="test")

# Quick evaluation
from openai import OpenAI
client = OpenAI()
results = vf_env.evaluate(client=client, model="gpt-4.1-mini", num_examples=10)
```

Environment modules encapsulate:
- Dataset loading and preprocessing
- Task-appropriate parsers and rubrics
- System prompts and sampling configuration
- Dependencies and setup requirements

## Data Types and Formats

### Dataset Format

Datasets should have either `answer` (str) or `info` (dict) columns:

```python
from datasets import Dataset

# Option 1: Simple format with answer column
dataset = Dataset.from_list([
    {"question": "What is 2+2?", "answer": "4"},
    {"question": "What is 3*5?", "answer": "15"},
])

# Option 2: Complex format with info dict
dataset = Dataset.from_list([
    {
        "question": "Solve this math problem: 2+2", 
        "info": {
            "answer": "4",
            "difficulty": "easy",
            "category": "arithmetic"
        }
    }
])

# Built-in datasets
dataset = vf.load_example_dataset("gsm8k", split="train")
dataset = vf.load_example_dataset("math", "train", n=1000)
```

### Message Types: Chat vs Completion

The framework supports two message formats:

```python
# Chat format (recommended for most cases)
message_type = "chat"
# Input: List[Dict[str, str]] with "role" and "content" keys
# Example: [{"role": "user", "content": "What is 2+2?"}]

# Completion format (for legacy models)
message_type = "completion"  
# Input: str (raw text)
# Example: "What is 2+2?"
```

**Recommendation: Use "chat" format in the vast majority of cases** as it's more flexible and supports system prompts.

### State Object

The `state` object contains rollout information and accumulates LLM responses:

```python
# State structure
state: Dict[str, Any] = {
    "prompt": List[Dict[str, str]],      # Original prompt
    "completion": List[Dict[str, str]],  # Model's response
    "answer": str,                        # Ground truth answer
    "task": str,                         # Task identifier
    "info": Dict[str, Any],              # Additional metadata
    "responses": List[Any],              # Full LLM response objects with token_ids, logprobs, etc.
}
```

The `state["responses"]` list accumulates the complete LLM response objects, which can include:
- `token_ids`: List of token IDs
- `logprobs`: Token-level log probabilities  
- `finish_reason`: Why generation stopped
- `usage`: Token usage statistics

### Sampling Arguments

Pass vLLM-specific arguments through the `sampling_args` dict:

```python
# vLLM-specific sampling arguments
sampling_args = {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 2048,
    "extra_body": {
        "skip_special_tokens": False,  # vLLM flag
        "spaces_between_special_tokens": False,  # vLLM flag
        "logprobs": True,  # Get token-level logprobs
        "top_logprobs": 5,  # Top-k logprobs per token
    }
}

# Use in environment
vf_env = vf.SingleTurnEnv(dataset=dataset, sampling_args=sampling_args)

# Or pass during evaluation
results = vf_env.evaluate(client=client, model="gpt-4", sampling_args=sampling_args)
```

This allows access to fine-grained information like token IDs and logprobs in environment and reward functions.

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

### 1. Environment-First Design
Most users should start by loading environment modules:

```python
# Load reusable environment patterns
vf_env = vf.load_environment("math-python")  # Math with Python tools
vf_env = vf.load_environment("wordle")       # Interactive games
vf_env = vf.load_environment("gsm8k")        # Simple Q&A

# Create directly only for simple or custom cases
vf_env = vf.SingleTurnEnv(dataset=dataset, parser=parser, rubric=rubric)
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

### 3. OpenAI-Compatible Integration
Verifiers can easily be integrated into any RL framework which exposes an OpenAI-compatible inference client:

```python
from openai import OpenAI

# Any OpenAI-compatible client works
client = OpenAI(base_url="https://your-endpoint.com/v1", api_key="your-key")
results = vf_env.evaluate(client=client, model="your-model", num_examples=100)
```

## Quick Start Example

Here's a complete working example:

```python
import verifiers as vf
from openai import OpenAI

# 1. Load or create environment
vf_env = vf.load_environment("gsm8k")  # Pre-configured GSM8K environment
# OR create custom:
# dataset = vf.load_example_dataset("gsm8k", split="train")
# parser = vf.ThinkParser()
# rubric = vf.Rubric(funcs=[correct_answer_func, parser.get_format_reward_func()])
# vf_env = vf.SingleTurnEnv(dataset=dataset, parser=parser, rubric=rubric)

# 2. Evaluate the environment
client = OpenAI()
results = vf_env.evaluate(
    client=client,
    model="gpt-4.1-mini",
    num_examples=10,
    rollouts_per_example=3
)
print(f"Results: {results}")

# 3. Generate training data
training_data = vf_env.generate(
    client=client,
    model="gpt-4.1-mini", 
    n_samples=1000
)

# 4. Save results
vf_env.make_dataset(training_data, push_to_hub=True, hub_name="my-training-data")
```

## Environment Types

Different environment types are designed for different tasks:

- **Environment Modules**: Pre-configured, installable packages (recommended)
- **SingleTurnEnv**: One-shot Q&A tasks (most common direct usage)
- **MultiTurnEnv**: Interactive conversations and multi-step reasoning
- **ToolEnv**: Tasks requiring external tools (calculators, code execution)
- **EnvGroup**: Composition of multiple environments
- **Custom Environments**: Write your own by extending `MultiTurnEnv`

## Training and Evaluation

Environments are not just for training - they're also powerful evaluation tools:

```python
# Evaluate a model
results = vf_env.evaluate(
    client=client,
    model="gpt-4.1-mini",
    num_examples=100
)

# Generate training data
results = vf_env.generate(
    client=client,
    model="gpt-4.1-mini",
    n_samples=1000
)
```

The framework supports various training approaches:
- **PRIME-RL (Recommended)**: Large-scale FSDP training with native verifiers support
- **GRPOTrainer**: Included trainer for LoRA and smaller setups
- **Custom Training**: Use environments as reward functions in your own training loops