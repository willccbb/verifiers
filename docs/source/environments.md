# Environments

Environments are the orchestration layer of the verifiers framework. They manage the complete lifecycle of LLM interactions, from dataset processing through rollout generation to reward calculation.

## Environment Hierarchy

```
Environment (base class)
├── SingleTurnEnv     # One-shot Q&A tasks
├── ToolEnv           # Tool-augmented reasoning  
├── SmolaToolEnv      # SmolaAgents integration
├── DoubleCheckEnv    # Multi-stage verification
├── TextArenaEnv      # Game environments
├── ReasoningGymEnv   # Reasoning benchmarks
└── MultiTurnEnv      # Interactive conversations
```

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
- Examples: Wordle, word games, strategy games

## ReasoningGymEnv: Reasoning Benchmarks

Integration with reasoning gym benchmarks:

```python
import verifiers as vf
from verifiers.envs.reasoninggym_env import ReasoningGymEnv

vf_env = ReasoningGymEnv(
    gym="arc_1d",
    num_samples=4000,
    max_concurrent=128,
    seed=1,
)
```

**Use ReasoningGymEnv when:**
- Training on established reasoning benchmarks
- Need standardized evaluation metrics
- Examples: ARC, logical reasoning tasks

## MultiTurnEnv: Interactive Conversations

For tasks requiring back-and-forth interaction (less commonly used):

```python
import verifiers as vf

class CustomMultiTurnEnv(vf.MultiTurnEnv):
    def __init__(self, dataset, max_turns=10):
        super().__init__(
            dataset=dataset,
            system_prompt="You are a helpful tutor...",
            max_turns=max_turns
        )
    
    def env_response(self, messages, state):
        """Generate environment's response to user."""
        # Custom logic for generating responses
        return response_message, updated_state
    
    def is_completed(self, messages, state):
        """Check if interaction should end."""
        return state.get("completed", False)
```

**Use MultiTurnEnv when:**
- Task requires dialogue or negotiation
- Environment needs to respond dynamically
- Examples: Tutoring, debugging conversations

## Training Setup

All environments work with the same training pattern:

```python
# Load model and setup training
model, tokenizer = vf.get_model_and_tokenizer("Qwen/Qwen2.5-1.5B-Instruct")
args = vf.grpo_defaults(run_name="my-experiment")

# Common configuration overrides
args.per_device_train_batch_size = 8
args.num_generations = 16
args.max_steps = 500

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=args,
)
trainer.train()
```

## Environment Selection Guide

Choose your environment type based on task requirements:

| Environment | Use Case | Examples |
|-------------|----------|----------|
| **SingleTurnEnv** | Simple Q&A | Math problems, classification |
| **ToolEnv** | Need external tools | Code execution, calculations |
| **SmolaToolEnv** | Advanced tools | Complex scientific computation |
| **DoubleCheckEnv** | Self-verification | Critical reasoning tasks |
| **TextArenaEnv** | Games/simulations | Wordle, strategy games |
| **ReasoningGymEnv** | Benchmarks | ARC, logical reasoning |
| **MultiTurnEnv** | Multi-turn chat | Tutoring, dialogue |

## Advanced Configuration

### Custom System Prompts

```python
system_prompt = """
Think step-by-step inside <think>...</think> tags.
Then give your final answer.

Format requirements:
- Show your reasoning clearly
- Give a definitive answer
- Be concise but complete
"""

vf_env = vf.SingleTurnEnv(
    dataset=dataset,
    system_prompt=system_prompt
)
```

### Custom Parsers and Rubrics

```python
# Custom parser for structured output
parser = vf.XMLParser(fields=["reasoning", "answer"])

# Custom reward function
def custom_reward(completion, answer, **kwargs):
    parsed = parser.parse(completion)
    # Your custom evaluation logic
    return reward_score

rubric = vf.Rubric(
    funcs=[custom_reward, parser.get_format_reward_func()],
    weights=[0.8, 0.2]
)

vf_env = vf.SingleTurnEnv(
    dataset=dataset,
    parser=parser,
    rubric=rubric
)
```

### Multi-GPU Setup

All environments support distributed training:

```bash
# Start vLLM inference server
CUDA_VISIBLE_DEVICES=0,1,2,3 vf-vllm --model 'Qwen/Qwen2.5-7B-Instruct' \
    --tensor-parallel-size 4 --max-model-len 8192

# Run training on separate GPUs  
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config-file configs/zero3.yaml \
    --num-processes 4 your_training_script.py
```

## Best Practices

1. **Start Simple**: Begin with SingleTurnEnv and add complexity as needed
2. **Use Built-ins**: Leverage `vf.load_example_dataset()` and `vf.grpo_defaults()`
3. **Test First**: Verify your environment works before large-scale training
4. **Monitor Training**: Use appropriate eval datasets and logging
5. **Scale Gradually**: Start with small models and datasets

Each environment type is optimized for specific use cases. The framework handles the complexity of distributed training, async generation, and reward computation automatically.
</rewritten_file>