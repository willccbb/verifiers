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

**SingleTurnEnv is perfect for:**
- Math problems and reasoning tasks
- Classification and categorization
- Translation tasks
- Any task with clear input-output structure

## MultiTurnEnv: Interactive Conversations

`MultiTurnEnv` is the base class for interactive environments where the model and environment can have multiple exchanges:

```python
import verifiers as vf

class MyMultiTurnEnv(vf.MultiTurnEnv):
    def is_completed(self, messages, state, **kwargs):
        """Define when the conversation should end."""
        # End after 5 exchanges
        return len(state['responses']) >= 5
    
    def env_response(self, messages, state, **kwargs):
        """Define how the environment responds to the model."""
        # Simple echo response
        last_message = messages[-1]['content']
        return {"role": "user", "content": f"You said: {last_message}"}, state

# Use the custom environment
vf_env = MyMultiTurnEnv(
    dataset=dataset,
    system_prompt="Have a conversation with the user.",
    parser=parser,
    rubric=rubric
)
```

**MultiTurnEnv is ideal for:**
- Interactive conversations
- Multi-step reasoning tasks
- Tool-augmented workflows
- Games and simulations

## Core Environment Features

### Dataset Management

Every environment handles datasets uniformly:

```python
import verifiers as vf

# Built-in datasets
dataset = vf.load_example_dataset("gsm8k", split="train")
dataset = vf.load_example_dataset("math", "train", n=6000)

# Custom datasets should have 'question' and 'answer' columns
# Environments automatically format prompts based on dataset structure
```

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
    rubric=custom_rubric
)
```

## SingleTurnEnv: Question-Answer Tasks

Perfect for tasks with a single question-answer exchange:

```python
import verifiers as vf

dataset = vf.load_example_dataset("gsm8k", split="train")

system_prompt = """
Think step-by-step inside <think>...</think> tags.
Then give your final answer.
"""

parser = vf.ThinkParser()

def correct_answer_reward_func(completion, answer, **kwargs):
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

TOOL_PROMPT = """
Think step-by-step inside <think>...</think> tags, then either call a tool inside <tool>...</tool> tags, or give your final answer inside <answer>...</answer> tags.

You have access to tools to help solve problems. Tools can be called with JSON:
<tool>
{{"name": "python", "args": {{"code": "print(2+2)"}}}}
</tool>
"""

dataset = vf.load_example_dataset("math", split="train")

vf_env = vf.ToolEnv(
    dataset=dataset,
    system_prompt=TOOL_PROMPT,
    tools=[python],
    max_steps=3
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

try:    
    from smolagents.default_tools import PythonInterpreterTool
    from verifiers.tools.smolagents import CalculatorTool
except ImportError:
    raise ImportError("Please install smolagents")

dataset = vf.load_example_dataset("math", "train", n=6000)

python_tool = PythonInterpreterTool(
    authorized_imports=["math", "sympy", "numpy"]
)
calculator_tool = CalculatorTool()

vf_env = SmolaToolEnv(
    dataset=dataset,
    system_prompt=MATH_SMOLA_PROMPT_TEMPLATE,
    few_shot=CALCULATOR_SMOLA_FEW_SHOTS,
    tools=[python_tool, calculator_tool],
    max_steps=5
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

dataset = vf.load_example_dataset("math", "train", n=1000)

vf_env = DoubleCheckEnv(
    dataset=dataset,
    system_prompt=SIMPLE_PROMPT,
    few_shot=[]
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
    num_samples=2000, 
    num_eval_samples=20
)
```

**Use TextArenaEnv when:**
- Training on game-based tasks
- Need interactive environment feedback

## Custom Environments

For nontrivial tasks, you'll want to write your own environment by extending `MultiTurnEnv`:

```python
import verifiers as vf

class MyCustomEnv(vf.MultiTurnEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add custom initialization
    
    def is_completed(self, messages, state, **kwargs):
        """Define completion criteria."""
        # Your logic here
        return some_condition
    
    def env_response(self, messages, state, **kwargs):
        """Define environment responses."""
        # Your logic here
        return response_message, updated_state

# Use your custom environment
vf_env = MyCustomEnv(
    dataset=dataset,
    parser=parser,
    rubric=rubric
)
```

## Environment Evaluation

Environments are powerful evaluation tools, not just for training:

```python
# Evaluate a model on the environment
results = vf_env.evaluate(
    client=openai_client,
    model="gpt-4",
    num_examples=100,
    rollouts_per_example=3
)

# Generate training data
results = vf_env.generate(
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

## Key Gotchas

1. **Parser Integration**: Always include format rewards from your parser in the rubric
2. **State Management**: MultiTurnEnv maintains conversation state - be careful about state mutations
3. **Completion Criteria**: Define clear completion criteria in `is_completed()` to avoid infinite loops
4. **Error Handling**: Environments should gracefully handle parsing failures and API errors
5. **Dataset Format**: Ensure your dataset has the expected columns (`question`, `answer`) or pre-format with `prompt` column