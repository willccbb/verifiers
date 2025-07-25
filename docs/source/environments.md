# Environments

Environments in Verifiers are installable Python modules which can specify dependencies in a `pyproject.toml`, and which expose a `load_environment` function for instantiation by downstream applications (e.g. trainers). They manage the complete lifecycle of LLM interactions, from dataset processing through rollout generation to reward calculation.

## Environment Loading Pattern

The primary way to use environments is through the `load_environment` function:

```python
import verifiers as vf

# Load an environment module with arguments
vf_env = vf.load_environment("math-python", dataset_name="math", num_train_examples=1000)

# Load with default settings
vf_env = vf.load_environment("wordle", use_think=True)

# Load GSM8K environment
vf_env = vf.load_environment("gsm8k")
```

This pattern allows environments to encapsulate their configuration, dependencies, and setup logic in reusable modules.

## Environment Management

### Initialize a New Environment

Create a template for a new environment module:

```bash
vf-init my-new-environment  # Creates ./environments/my_new_environment/
```

This creates:
- `my_new_environment.py` - Main environment implementation
- `pyproject.toml` - Package configuration and dependencies  
- `README.md` - Documentation

### Install Environment Modules

Install a local environment module:
```bash
vf-install my-new-environment  # From ./environments/
```

Install from the verifiers repository:
```bash
vf-install math-python --from-repo  # From GitHub repo
vf-install wordle --from-repo -b main  # Specify branch
```

### Quick Evaluation

Test any environment with an API model:
```bash
vf-eval math-python -m gpt-4.1-mini -n 5 -r 3
vf-eval wordle --env-args use_think=True -n 10
```

## Environment Hierarchy

```
Environment (base class)
├── MultiTurnEnv      # Interactive conversations (base for most environments)
│   └── SingleTurnEnv # One-shot Q&A tasks (most common entry point)
│   └── ToolEnv       # Tool-augmented reasoning  
│   └── TextArenaEnv  # Game environments
└── EnvGroup          # Composition of multiple environments
```

## Environment Types

### SingleTurnEnv: Question-Answer Tasks

**Most users should start with `SingleTurnEnv`** for one-shot question-answer tasks:

```python
import verifiers as vf
from datasets import Dataset

dataset = vf.load_example_dataset("gsm8k", split="train")

vf_env = vf.SingleTurnEnv(
    dataset=dataset,
    system_prompt="Solve the problem step by step.",
    parser=vf.ThinkParser(),
    rubric=vf.Rubric(funcs=[correct_answer_func])
)

# Evaluate the environment
from openai import OpenAI
results = vf_env.evaluate(
    client=OpenAI(),
    model="gpt-4.1-mini", 
    num_examples=10
)
```

**SingleTurnEnv is perfect for:**
- Math problems and reasoning tasks
- Classification and categorization
- Translation tasks
- Any task with clear input-output structure

### MultiTurnEnv: Interactive Conversations

`MultiTurnEnv` is the base class for interactive environments where the model and environment can have multiple exchanges:

```python
import verifiers as vf
from typing import List, Dict, Any, Tuple

class MyMultiTurnEnv(vf.MultiTurnEnv):
    def is_completed(self, messages: List[Dict[str, str]], state: Dict[str, Any], **kwargs) -> bool:
        """Define when the conversation should end."""
        # End after 5 exchanges
        # state["responses"] contains full LLM response objects with token_ids, logprobs, etc.
        return len(state['responses']) >= 5
    
    def env_response(self, messages: List[Dict[str, str]], state: Dict[str, Any], **kwargs) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """Define how the environment responds to the model."""
        # Simple echo response
        last_message = messages[-1]['content']
        return [{"role": "user", "content": f"You said: {last_message}"}], state

# Use the custom environment
vf_env = MyMultiTurnEnv(
    dataset=dataset,
    parser=parser,
    rubric=rubric
)
```

**MultiTurnEnv is ideal for:**
- Interactive conversations
- Multi-step reasoning tasks
- Tool-augmented workflows
- Games and simulations

### ToolEnv: Tool-Augmented Reasoning

Enable models to use external tools for complex reasoning:

```python
import verifiers as vf
from verifiers.utils.tools import python

dataset = vf.load_example_dataset("math", split="train")

vf_env = vf.ToolEnv(
    dataset=dataset,
    system_prompt="""Think step-by-step inside <think>...</think> tags, then either call a tool inside <tool>...</tool> tags, or give your final answer inside <answer>...</answer> tags.

You have access to tools to help solve problems. Tools can be called with JSON:
<tool>
{"name": "python", "args": {"code": "print(2+2)"}}
</tool>""",
    tools=[python],
    max_turns=3
)
```

**ToolEnv is used when:**
- Task requires computation or external data
- Reasoning alone is insufficient  
- Examples: Complex math, data analysis, code execution

### EnvGroup: Environment Composition

Combine multiple environments for mixed task types:

```python
import verifiers as vf

# Create individual environments
math_env = vf.load_environment("math-python")
gsm8k_env = vf.load_environment("gsm8k") 
wordle_env = vf.load_environment("wordle")

# Combine into a group
vf_env = vf.EnvGroup(
    envs=[math_env, gsm8k_env, wordle_env],
    env_names=["math", "gsm8k", "wordle"]
)

# The group routes to appropriate sub-environment based on 'task' column
```

## Available Environment Modules

The `environments/` folder contains canonical implementations:

### Math Environments
- **`math-python`**: Math problems with Python tool execution
- **`gsm8k`**: Grade school math problems
- **`aime2024`**, **`aime2025`**: Competition math problems

### Question Answering
- **`gpqa`**: Graduate-level science questions
- **`simpleqa`**: Simple question-answering

### Interactive Environments  
- **`wordle`**: Word guessing game
- **`wiki-search`**: Multi-turn Wikipedia search

### Tool-Augmented Environments
- **`smolagents-math-tools`**: Advanced tool integration with SmolaAgents
- **`xml-tool-env`**: Example of XML-based tool calling

### Reasoning Environments
- **`reasoning-gym`**: Various reasoning benchmarks
- **`doublecheck`**: Self-verification workflows
- **`self-reward`**: Self-rewarding training

## Creating Environment Modules

### Basic Structure

Every environment module needs:

1. **`{module_name}.py`** - Main implementation with `load_environment()` function
2. **`pyproject.toml`** - Package configuration and dependencies
3. **`README.md`** - Documentation

### Example Implementation

```python
# math_example.py
import verifiers as vf
from verifiers.utils.data_utils import extract_boxed_answer, load_example_dataset

def load_environment(
    dataset_name: str = "math",
    dataset_split: str = "train", 
    num_train_examples: int = -1,
    **kwargs
):
    dataset = load_example_dataset(dataset_name, dataset_split, n=num_train_examples)
    
    system_prompt = """Think step by step inside <think>...</think>.
Give your final answer inside \\boxed{}."""

    parser = vf.Parser(extract_fn=extract_boxed_answer)

    def correct_answer_reward_func(parser, completion, answer) -> float:
        completion_answer = parser.parse_answer(completion)
        return 1.0 if completion_answer == answer else 0.0

    rubric = vf.Rubric(
        funcs=[correct_answer_reward_func, parser.get_format_reward_func()],
        weights=[1.0, 0.2],
        parser=parser,
    )

    vf_env = vf.SingleTurnEnv(
        dataset=dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        **kwargs,
    )

    return vf_env
```

### Package Configuration

```toml
# pyproject.toml
[project]
name = "math-example"
version = "0.1.0"
dependencies = [
    "verifiers>=0.1.2",
    "math-verify>=0.1.0",  # Any specific dependencies
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build]
include = ["math_example.py"]
```

## Dataset Management

Environments handle datasets uniformly. Datasets should have either `answer` (str) or `info` (dict) columns:

```python
# Built-in datasets
dataset = vf.load_example_dataset("gsm8k", split="train")
dataset = vf.load_example_dataset("math", "train", n=1000)

# Custom datasets - Simple format
dataset = Dataset.from_list([
    {"question": "What is 2+2?", "answer": "4"},
    {"question": "What is 3*5?", "answer": "15"},
])

# Custom datasets - Complex format with info dict
dataset = Dataset.from_list([
    {
        "question": "Solve: 2+2", 
        "info": {
            "answer": "4",
            "difficulty": "easy",
            "category": "arithmetic"
        }
    }
])
```

## Message Types and Chat Templates

The framework supports two message formats:

```python
# Chat format (recommended for most cases)
message_type = "chat"
# Input: List[Dict[str, str]] with "role" and "content" keys
# Supports: system prompts, multi-turn conversations

# Completion format (for legacy models)
message_type = "completion"  
# Input: str (raw text)
# Limited: no system prompts
```

**Recommendation: Use "chat" format in the vast majority of cases** as it's more flexible and supports system prompts.

## Environment Evaluation

Environments are powerful evaluation tools:

```python
from openai import OpenAI

# Evaluate a model on the environment
results = vf_env.evaluate(
    client=OpenAI(),
    model="gpt-4.1-mini",
    num_examples=100,
    rollouts_per_example=3
)

# Check results
print(f"Average reward: {sum(results['rewards']) / len(results['rewards'])}")

# Generate training data  
results = vf_env.generate(
    client=OpenAI(),
    model="gpt-4.1-mini",
    n_samples=1000
)

# Save results to Hugging Face Hub
vf_env.make_dataset(results, push_to_hub=True, hub_name="my-environment-results")
```

## State Object and Responses

The `state` object contains rollout information and accumulates LLM responses:

```python
# State structure
state: Dict[str, Any] = {
    "prompt": List[Dict[str, str]],      # Original prompt
    "completion": List[Dict[str, str]],  # Model's response
    "answer": str,                        # Ground truth answer  
    "task": str,                         # Task identifier
    "info": Dict[str, Any],              # Additional metadata
    "responses": List[Any],              # Full LLM response objects
}
```

The `state["responses"]` list contains complete LLM response objects with:
- `token_ids`: Token-level information
- `logprobs`: Log probabilities  
- `finish_reason`: Generation termination reason
- `usage`: Token usage statistics

## Sampling Arguments

Pass vLLM-specific arguments through the `sampling_args` dict:

```python
# vLLM-specific sampling arguments
sampling_args = {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 2048,
    "extra_body": {
        "skip_special_tokens": False,
        "spaces_between_special_tokens": False, 
        "logprobs": True,        # Get token-level logprobs
        "top_logprobs": 5,       # Top-k logprobs per token
    }
}

# Use in environment
vf_env = vf.SingleTurnEnv(dataset=dataset, sampling_args=sampling_args)

# Or pass during evaluation
results = vf_env.evaluate(client=client, model="gpt-4", sampling_args=sampling_args)
```

## Constraints and Footguns

### Token Sequence Requirements

The primary constraint we impose on rollout logic is that token sequences must be **increasing**, i.e. once a token has been added to a model's context in a rollout, it must remain as the rollout progresses.

**Non-Increasing Chat Templates:** The Qwen3 and DeepSeek-R1 model series both remove `<think>` sections from messages when processing inputs, which violates the increasing context requirement for multi-turn GRPO-style training. We provide versions of many of these models with modified chat templates in the Hugging Face collections.

### Common Issues

1. **Parser Integration**: Always include format rewards from your parser in the rubric
2. **State Management**: MultiTurnEnv maintains conversation state - be careful about state mutations
3. **Completion Criteria**: Define clear completion criteria in `is_completed()` to avoid infinite loops
4. **Error Handling**: Environments should gracefully handle parsing failures and API errors
5. **Dataset Format**: Ensure your dataset has the expected columns or pre-format appropriately
6. **Message Type**: Use "chat" format in the vast majority of cases for better flexibility

## TODO Sections

TODO: Add documentation for:
- Hardware considerations for environment evaluation
- SFT warmup patterns for improving small-model training efficiency  
- Advanced environment composition patterns
- Environment-specific best practices for different domains