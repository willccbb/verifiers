# Examples

This guide walks through real-world examples from the verifiers codebase, showing how to use different environment types for various tasks.

## Math Problem Solving with SingleTurnEnv

The simplest approach for Q&A tasks using GSM8K dataset:

```python
import verifiers as vf
from verifiers.utils.data_utils import extract_boxed_answer

# Load dataset
dataset = vf.load_example_dataset("gsm8k", split="train")
eval_dataset = vf.load_example_dataset("gsm8k", split="test")

# System prompt for math problems
system_prompt = """
Think step-by-step inside <think>...</think> tags.

Then, give your final numerical answer inside \\boxed{{...}}.
"""

# Parser extracts content after </think> and looks for boxed answers
parser = vf.ThinkParser(extract_fn=extract_boxed_answer)

def correct_answer_reward_func(completion, answer, **kwargs):
    response = parser.parse_answer(completion) or ''
    return 1.0 if response == answer else 0.0

rubric = vf.Rubric(funcs=[
    correct_answer_reward_func,
    parser.get_format_reward_func()
], weights=[1.0, 0.2])

vf_env = vf.SingleTurnEnv(
    dataset=dataset,
    eval_dataset=eval_dataset,
    system_prompt=system_prompt,
    parser=parser,
    rubric=rubric,
)

# Training setup
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
model, tokenizer = vf.get_model_and_tokenizer(model_name)
args = vf.grpo_defaults(run_name="gsm8k-example")
args.per_device_train_batch_size = 12
args.num_generations = 12
args.max_steps = 100

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=args,
    peft_config=vf.lora_defaults()
)
trainer.train()
```

## Tool-Augmented Math with ToolEnv

For complex math problems requiring code execution:

```python
import verifiers as vf
from verifiers.tools import python

# System prompt for tool usage
TOOL_PROMPT = """
Think step-by-step inside <think>...</think> tags in each message, then either call a tool inside <tool>...</tool> tags, or give your final answer inside <answer>...</answer> tags.

You have access to the following tools to help solve problems:

{tool_descriptions}

Tools can be called by writing a JSON command inside <tool> tags with:
- "name": the name of the tool to use
- "args": the arguments for the tool

Example usage:
<tool>
{{"name": "python", "args": {{"code": "import sympy\nx = sympy.symbols('x')\nprint(sympy.solve(x**2 - 4, x))"}}}}
</tool>

The <answer>...</answer> tags should contain only your final answer as a numeric expression.
"""

dataset = vf.load_example_dataset("math", split="train")

vf_env = vf.ToolEnv(
    dataset=dataset,
    system_prompt=TOOL_PROMPT,
    few_shot=[],
    tools=[python],
    max_steps=3
)

# Training with tool environment
model_name = "willcb/Qwen2.5-7B-Math-Python-SFT"
model, tokenizer = vf.get_model_and_tokenizer(model_name)
run_name = "math-grpo_" + model_name.split("/")[-1].lower()

training_args = vf.grpo_defaults(run_name=run_name)
training_args.num_iterations = 2
training_args.per_device_train_batch_size = 8
training_args.num_generations = 8

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=training_args,
)
trainer.train()
```

## SmolaAgents Integration with SmolaToolEnv

Using SmolaAgents for advanced tool integration:

```python
import verifiers as vf
from verifiers.envs.smola_tool_env import SmolaToolEnv
from verifiers.prompts.system_prompts import MATH_SMOLA_PROMPT_TEMPLATE
from verifiers.prompts.few_shots import CALCULATOR_SMOLA_FEW_SHOTS

try:    
    from smolagents.default_tools import PythonInterpreterTool
    from verifiers.tools.smolagents import CalculatorTool
except ImportError:
    raise ImportError("Please install smolagents to use SmolAgents tools.")

dataset = vf.load_example_dataset("math", "train", n=6000)
eval_aime24 = vf.load_example_dataset("aime2024", n=30)
eval_aime25 = vf.load_example_dataset("aime2025", n=30)

# Use SmolaAgents' PythonInterpreterTool with custom calculator
python_tool = PythonInterpreterTool(
    authorized_imports=["math", "sympy", "numpy"]
)
calculator_tool = CalculatorTool()

vf_env = SmolaToolEnv(
    dataset=dataset,
    eval_dataset=eval_dataset,
    system_prompt=MATH_SMOLA_PROMPT_TEMPLATE,
    few_shot=CALCULATOR_SMOLA_FEW_SHOTS,
    tools=[python_tool, calculator_tool],
    max_steps=5
)

model_name = "Qwen/Qwen2.5-7B-Instruct"
model, tokenizer = vf.get_model_and_tokenizer(model_name)
run_name = "math-smola-grpo_" + model_name.split("/")[-1].lower()

args = vf.grpo_defaults(run_name=run_name)
trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=args,
)
trainer.train()
```

## Game Environment with TextArenaEnv

Training on interactive games like Wordle:

```python
import verifiers as vf
from verifiers.envs.textarena_env import TextArenaEnv

# Setup for Wordle training
size = '7B'
model_name = f'willcb/Qwen2.5-{size}-Wordle-SFT'
model, tokenizer = vf.get_model_and_tokenizer(model_name)

vf_env = TextArenaEnv(
    game="Wordle-v0",
    num_samples=2000, 
    num_eval_samples=20
)

run_name = f"wordle-grpo-{size}"
training_args = vf.grpo_defaults(run_name=run_name)
training_args.num_iterations = 1
training_args.per_device_train_batch_size = 8
training_args.num_generations = 16
training_args.gradient_accumulation_steps = 6
training_args.max_prompt_length = 1024
training_args.max_completion_length = 3072
training_args.max_steps = 100
training_args.mask_env_responses = True

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=training_args,
)
trainer.train()
```

## Reasoning Benchmarks with ReasoningGymEnv

Training on reasoning gym benchmarks like ARC:

```python
import verifiers as vf
from verifiers.envs.reasoninggym_env import ReasoningGymEnv

size = '14B'
model_name = f'willcb/Qwen3-{size}-Arc-1D-SFT'
model, tokenizer = vf.get_model_and_tokenizer(model_name)

vf_env = ReasoningGymEnv(
    gym="arc_1d",
    num_samples=4000,
    max_concurrent=128,
    seed=1,
)

run_name = f"arc_1d-grpo-{size}"
training_args = vf.grpo_defaults(run_name=run_name)
training_args.num_iterations = 1
training_args.per_device_train_batch_size = 4
training_args.num_generations = 16
training_args.gradient_accumulation_steps = 8
training_args.max_concurrent = 512
training_args.max_prompt_length = 1024
training_args.max_completion_length = 4096
training_args.max_steps = 500

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=training_args,
)
trainer.train()
```

## Self-Verification with DoubleCheckEnv

Multi-stage verification workflow:

```python
import verifiers as vf
from verifiers.envs.doublecheck_env import DoubleCheckEnv   

SIMPLE_PROMPT = """\
You are a helpful assistant. In each turn, think step-by-step inside <think>...</think> tags, then give your final answer inside <answer>...</answer> tags.
"""

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
dataset = vf.load_example_dataset("math", "train", n=1000)

vf_env = DoubleCheckEnv(
    dataset=dataset,
    system_prompt=SIMPLE_PROMPT,
    few_shot=[]
)

model, tokenizer = vf.get_model_and_tokenizer(model_name)
args = vf.grpo_defaults(run_name="doublecheck-{}".format(model_name.split("/")[-1].lower()))

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=args,
)
trainer.train()
```

## Infrastructure Setup

### Multi-GPU Training

All examples support multi-GPU training with these commands:

```bash
# Start vLLM inference server (4 GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3 vf-vllm --model 'Qwen/Qwen2.5-7B-Instruct' \
    --tensor-parallel-size 4 --max-model-len 8192 --dtype bfloat16 \
    --gpu-memory-utilization 0.9 --enable-prefix-caching \
    --host 0.0.0.0 --port 8000

# Run training on separate GPUs
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config-file configs/zero3.yaml \
    --num-processes 4 your_training_script.py
```

### Common Training Arguments

Most examples use these patterns:

```python
# Basic configuration
args = vf.grpo_defaults(run_name="my-experiment")

# Common overrides
args.per_device_train_batch_size = 8
args.num_generations = 16
args.gradient_accumulation_steps = 4
args.max_steps = 500
args.max_prompt_length = 1024
args.max_completion_length = 2048

# For parameter-efficient training
trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=args,
    peft_config=vf.lora_defaults()  # Add LoRA
)
```

## Key Patterns

### Dataset Loading
```python
# Built-in datasets
dataset = vf.load_example_dataset("gsm8k", split="train")
dataset = vf.load_example_dataset("math", "train", n=6000)

# Custom datasets should have 'question' and 'answer' columns
```

### Model Loading
```python
# Standard pattern
model, tokenizer = vf.get_model_and_tokenizer(model_name)

# All examples use this instead of manual loading
```

### Training Configuration
```python
# Always start with defaults
args = vf.grpo_defaults(run_name="descriptive-name")

# Then customize as needed
args.per_device_train_batch_size = 8
args.num_generations = 16
```

These examples show the actual patterns used in production. Each environment type is designed for specific use cases, and the framework handles the complexity of distributed training, async generation, and reward computation automatically.