# Environments

Environments are the orchestration layer of the verifiers framework. They manage the complete lifecycle of LLM interactions, from dataset processing through rollout generation to reward calculation.

## Environment Hierarchy

```
Environment (base class)
├── MultiTurnEnv      # Interactive conversations (base for most environments)
│   └── SingleTurnEnv # One-shot Q&A tasks (most common entry point)
├── ToolEnv           # Tool-augmented reasoning  
├── SmolaToolEnv      # SmolaAgents integration
├── DoubleCheckEnv    # Multi-stage verification
├── TextArenaEnv      # Game environments
└── ReasoningGymEnv   # Reasoning benchmarks
```

## Getting Started: SingleTurnEnv

**Most users should start with `SingleTurnEnv`** for one-shot question-answer tasks. It's the simplest and most common environment type:

```python
import verifiers as vf
from typing import List, Dict, Any, Union
from datasets import Dataset

# Load a dataset
dataset: Dataset = vf.load_example_dataset("gsm8k", split="train")

# Create a simple environment
vf_env = vf.SingleTurnEnv(
    dataset=dataset,
    system_prompt="Solve the problem step by step.",
    parser=parser,
    rubric=rubric,
    message_type="chat"  # Recommended: use "chat" for most cases
)

# Evaluate the environment
results = vf_env.evaluate(
    client=openai_client,
    model="gpt-4",
    num_examples=10
)
```

**SingleTurnEnv is perfect for:**
- Math problems and reasoning tasks
- Classification and categorization
- Translation tasks
- Any task with clear input-output structure

## MultiTurnEnv: Interactive Conversations

`MultiTurnEnv` is the base class for interactive environments where the model and environment can have multiple exchanges:

```python
import verifiers as vf
from typing import List, Dict, Any, Union

class MyMultiTurnEnv(vf.MultiTurnEnv):
    def is_completed(self, messages: List[Dict[str, str]], state: Dict[str, Any], **kwargs) -> bool:
        """Define when the conversation should end."""
        # End after 5 exchanges
        # state["responses"] contains full LLM response objects with token_ids, logprobs, etc.
        return len(state['responses']) >= 5
    
    def env_response(self, messages: List[Dict[str, str]], state: Dict[str, Any], **kwargs) -> tuple[Dict[str, str], Dict[str, Any]]:
        """Define how the environment responds to the model."""
        # Simple echo response
        last_message = messages[-1]['content']
        return {"role": "user", "content": f"You said: {last_message}"}, state

# Use the custom environment
vf_env = MyMultiTurnEnv(
    dataset=dataset,
    system_prompt="Have a conversation with the user.",
    parser=parser,
    rubric=rubric,
    message_type="chat"  # Recommended format
)
```

**MultiTurnEnv is ideal for:**
- Interactive conversations
- Multi-step reasoning tasks
- Tool-augmented workflows
- Games and simulations

## Core Environment Features

### Dataset Management

Every environment handles datasets uniformly. Datasets should have either `answer` (str) or `info` (dict) columns:

```python
import verifiers as vf
from datasets import Dataset
from typing import List, Dict, Any

# Built-in datasets
dataset: Dataset = vf.load_example_dataset("gsm8k", split="train")
dataset: Dataset = vf.load_example_dataset("math", "train", n=6000)

# Custom datasets - Option 1: Simple format with answer column
dataset = Dataset.from_list([
    {"question": "What is 2+2?", "answer": "4"},
    {"question": "What is 3*5?", "answer": "15"},
])

# Custom datasets - Option 2: Complex format with info dict
dataset = Dataset.from_list([
    {
        "question": "Solve this math problem: 2+2", 
        "info": {
            "answer": "4",
            "difficulty": "easy",
            "category": "arithmetic"
        }
    },
    {
        "question": "What is the capital of France?",
        "info": {
            "answer": "Paris",
            "difficulty": "easy", 
            "category": "geography"
        }
    }
])

# Environments automatically format prompts based on dataset structure
```

### Message Types: Chat vs Completion

The framework supports two message formats:

```python
# Chat format (recommended for most cases)
message_type = "chat"
# Input: List[Dict[str, str]] with "role" and "content" keys
# Example: [{"role": "user", "content": "What is 2+2?"}]
# Supports: system prompts, multi-turn conversations

# Completion format (for legacy models)
message_type = "completion"  
# Input: str (raw text)
# Example: "What is 2+2?"
# Limited: no system prompts
```

**Recommendation: Use "chat" format in the vast majority of cases** as it's more flexible and supports system prompts.

### Automatic Setup

Environments often provide sensible defaults:

```python
# Simple - environment chooses parser/rubric automatically
vf_env = vf.SingleTurnEnv(dataset=dataset)

# Custom - specify your own components
vf_env = vf.SingleTurnEnv(
    dataset=dataset,
    system_prompt="Custom instructions...",
    parser=vf.ThinkParser(),
    rubric=custom_rubric,
    message_type="chat"  # Explicitly set recommended format
)
```

## SingleTurnEnv: Question-Answer Tasks

Perfect for tasks with a single question-answer exchange:

```python
import verifiers as vf
from typing import Union, List, Dict

dataset: Dataset = vf.load_example_dataset("gsm8k", split="train")

system_prompt = """
Think step-by-step inside <think>...</think> tags.
Then give your final answer.
"""

parser = vf.ThinkParser()

def correct_answer_reward_func(completion: Union[str, List[Dict[str, str]]], answer: str, **kwargs) -> float:
    response = parser.parse_answer(completion) or ''
    return 1.0 if response.strip() == answer.strip() else 0.0

rubric = vf.Rubric(funcs=[
    correct_answer_reward_func,
    parser.get_format_reward_func()
], weights=[1.0, 0.2])

vf_env = vf.SingleTurnEnv(
    dataset=dataset,
    system_prompt=system_prompt,
    parser=parser,
    rubric=rubric,
    message_type="chat"  # Recommended format
)
```

**Use SingleTurnEnv when:**
- Task has clear input-output structure
- No external tools needed
- Examples: Math problems, classification, translation

## ToolEnv: Tool-Augmented Reasoning

Enable models to use external tools for complex reasoning:

```python
import verifiers as vf
from verifiers.tools import python
from typing import List, Callable

TOOL_PROMPT = """
Think step-by-step inside <think>...</think> tags, then either call a tool inside <tool>...</tool> tags, or give your final answer inside <answer>...</answer> tags.

You have access to tools to help solve problems. Tools can be called with JSON:
<tool>
{{"name": "python", "args": {{"code": "print(2+2)"}}}}
</tool>
"""

dataset: Dataset = vf.load_example_dataset("math", split="train")

vf_env = vf.ToolEnv(
    dataset=dataset,
    system_prompt=TOOL_PROMPT,
    tools=[python],  # List[Callable]
    max_steps=3,
    message_type="chat"  # Recommended format
)
```

**Use ToolEnv when:**
- Task requires computation or external data
- Reasoning alone is insufficient  
- Examples: Complex math, data analysis, code execution

## SmolaToolEnv: SmolaAgents Integration

Advanced tool integration using SmolaAgents:

```python
import verifiers as vf
from verifiers.envs.smola_tool_env import SmolaToolEnv
from typing import List, Callable

try:    
    from smolagents.default_tools import PythonInterpreterTool
    from verifiers.tools.smolagents import CalculatorTool
except ImportError:
    raise ImportError("Please install smolagents")

dataset: Dataset = vf.load_example_dataset("math", "train", n=6000)

python_tool: PythonInterpreterTool = PythonInterpreterTool(
    authorized_imports=["math", "sympy", "numpy"]
)
calculator_tool: CalculatorTool = CalculatorTool()

vf_env = SmolaToolEnv(
    dataset=dataset,
    system_prompt=MATH_SMOLA_PROMPT_TEMPLATE,

    tools=[python_tool, calculator_tool],  # List[Callable]
    max_steps=5,
    message_type="chat"  # Recommended format
)
```

**Use SmolaToolEnv when:**
- Need advanced tool capabilities
- Want SmolaAgents ecosystem integration
- Examples: Complex scientific computation, multi-step tool workflows

## DoubleCheckEnv: Self-Verification

Multi-stage verification workflow where models check their own work:

```python
import verifiers as vf
from verifiers.envs.doublecheck_env import DoubleCheckEnv

SIMPLE_PROMPT = """
You are a helpful assistant. Think step-by-step inside <think>...</think> tags, then give your final answer inside <answer>...</answer> tags.
"""

dataset: Dataset = vf.load_example_dataset("math", "train", n=1000)

vf_env = DoubleCheckEnv(
    dataset=dataset,
    system_prompt=SIMPLE_PROMPT,

    message_type="chat"  # Recommended format
)
```

**Use DoubleCheckEnv when:**
- Want self-verification workflows
- Need to improve reliability through checking
- Examples: Critical reasoning tasks, factual accuracy

## TextArenaEnv: Game Environments

Training on interactive games and simulations:

```python
import verifiers as vf
from verifiers.envs.textarena_env import TextArenaEnv

vf_env = TextArenaEnv(
    game="Wordle-v0",
    num_train_examples=2000, 
    num_eval_examples=20,
    message_type="chat"  # Recommended format
)
```

**Use TextArenaEnv when:**
- Training on game-based tasks
- Need interactive environment feedback

## Custom Environments

For nontrivial tasks, you'll want to write your own environment by extending `MultiTurnEnv`:

```python
import verifiers as vf
from typing import List, Dict, Any, Union

class MyCustomEnv(vf.MultiTurnEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add custom initialization
    
    def is_completed(self, messages: List[Dict[str, str]], state: Dict[str, Any], **kwargs) -> bool:
        """Define completion criteria."""
        # Your logic here
        # state["responses"] contains full LLM response objects
        return some_condition
    
    def env_response(self, messages: List[Dict[str, str]], state: Dict[str, Any], **kwargs) -> tuple[Dict[str, str], Dict[str, Any]]:
        """Define environment responses."""
        # Your logic here
        return response_message, updated_state

# Use your custom environment
vf_env = MyCustomEnv(
    dataset=dataset,
    parser=parser,
    rubric=rubric,
    message_type="chat"  # Recommended format
)
```

## Environment Evaluation

Environments are powerful evaluation tools, not just for training:

```python
from typing import Dict, Any

# Evaluate a model on the environment
results: Dict[str, Any] = vf_env.evaluate(
    client=openai_client,
    model="gpt-4",
    num_examples=100,
    rollouts_per_example=3
)

# Generate training data
results: Dict[str, Any] = vf_env.generate(
    client=openai_client,
    model="gpt-4",
    n_samples=1000
)

# Process results for training
processed = vf_env.process_env_results(
    prompts=results['prompts'],
    completions=results['completions'],
    states=results['states'],
    rewards=results['rewards'],
    processing_class=tokenizer
)
```

## State Object Details

The `state` object contains rollout information and accumulates LLM responses:

```python
from typing import Dict, Any, List

# State structure
state: Dict[str, Any] = {
    "prompt": List[Dict[str, str]],  # Original prompt
    "completion": List[Dict[str, str]],  # Model's response
    "answer": str,  # Ground truth answer
    "task": str,  # Task identifier
    "info": Dict[str, Any],  # Additional metadata
    "responses": List[Any],  # Full LLM response objects with token_ids, logprobs, etc.
}
```

The `state["responses"]` list accumulates the complete LLM response objects, which can include:
- `token_ids`: List of token IDs
- `logprobs`: Token-level log probabilities  
- `finish_reason`: Why generation stopped
- `usage`: Token usage statistics

This allows access to fine-grained information in environment and reward functions.

## Sampling Arguments

Pass vLLM-specific arguments through the `sampling_args` dict:

```python
from typing import Dict, Any

# vLLM-specific sampling arguments
sampling_args: Dict[str, Any] = {
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
vf_env = vf.SingleTurnEnv(
    dataset=dataset,
    sampling_args=sampling_args
)

# Or pass during evaluation
results = vf_env.evaluate(
    client=openai_client,
    model="gpt-4",
    sampling_args=sampling_args
)
```

This allows access to fine-grained information like token IDs and logprobs in environment and reward functions.

## Key Gotchas

1. **Parser Integration**: Always include format rewards from your parser in the rubric
2. **State Management**: MultiTurnEnv maintains conversation state - be careful about state mutations
3. **Completion Criteria**: Define clear completion criteria in `is_completed()` to avoid infinite loops
4. **Error Handling**: Environments should gracefully handle parsing failures and API errors
5. **Dataset Format**: Ensure your dataset has the expected columns (`question`, `answer`) or pre-format with `prompt` column
6. **Message Type**: Use "chat" format in the vast majority of cases for better flexibility
7. **Type Hints**: Use proper type hints for better code clarity and IDE support