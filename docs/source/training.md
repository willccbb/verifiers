# Training and Evaluation

The verifiers framework supports both evaluation and training workflows. Environments are powerful evaluation tools that can also be used for training models using reinforcement learning.

## Environment Evaluation

Environments are not just for training - they're excellent evaluation tools:

```python
import verifiers as vf

# Create environment
dataset = vf.load_example_dataset("gsm8k", split="train")
vf_env = vf.SingleTurnEnv(dataset=dataset)

# Evaluate model performance
results = vf_env.evaluate(
    client=openai_client,
    model="gpt-4",
    num_examples=100,
    rollouts_per_example=3
)

print(f"Average reward: {sum(results['rewards']) / len(results['rewards'])}")
print(f"Correct answers: {sum(1 for r in results['rewards'] if r > 0.8)}")

# Generate training data
results = vf_env.generate(
    client=openai_client,
    model="gpt-4",
    n_samples=1000
)
```

## Training with GRPO

For users who want to train models, the framework integrates with Group Relative Policy Optimization (GRPO). Here's the basic pattern:

```python
import verifiers as vf

# 1. Create environment
dataset = vf.load_example_dataset("gsm8k", split="train")
vf_env = vf.SingleTurnEnv(dataset=dataset)

# 2. Load model
model, tokenizer = vf.get_model_and_tokenizer("Qwen/Qwen2.5-1.5B-Instruct")

# 3. Configure training  
args = vf.grpo_defaults(run_name="my-experiment")

# 4. Train
trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=args,
)
trainer.train()
```

## GRPO: Group Relative Policy Optimization

GRPO is a reinforcement learning algorithm designed specifically for LLMs that:
- Learns from relative comparisons within groups
- Reduces reward hacking through comparative evaluation
- Provides stable training dynamics
- Works with any differentiable model

## Training Configuration

### Using Defaults

All examples start with `vf.grpo_defaults()`:

```python
args = vf.grpo_defaults(run_name="descriptive-name")

# Common overrides
args.per_device_train_batch_size = 8
args.num_generations = 16
args.gradient_accumulation_steps = 4
args.max_steps = 500
args.max_prompt_length = 1024
args.max_completion_length = 2048
```

### Key Parameters

```python
# Batch configuration
args.per_device_train_batch_size = 8    # Samples per GPU
args.num_generations = 16               # Completions per prompt
args.gradient_accumulation_steps = 4    # Steps before update

# Length limits
args.max_prompt_length = 1024           # Truncate long prompts
args.max_completion_length = 2048       # Truncate long responses

# Training schedule
args.max_steps = 500                    # Total training steps
args.num_iterations = 1                 # Updates per batch
args.learning_rate = 1e-6               # Learning rate

# Generation settings
args.temperature = 1.0                  # Sampling temperature
args.top_p = 1.0                        # Nucleus sampling
```

## Model Loading

Use the standard pattern from examples:

```python
# Basic loading
model, tokenizer = vf.get_model_and_tokenizer("Qwen/Qwen2.5-1.5B-Instruct")

# With specific models from examples
model_name = "willcb/Qwen2.5-7B-Math-Python-SFT"  # Pre-trained on math
model, tokenizer = vf.get_model_and_tokenizer(model_name)

# Different sizes
model, tokenizer = vf.get_model_and_tokenizer("Qwen/Qwen2.5-7B-Instruct")
model, tokenizer = vf.get_model_and_tokenizer("willcb/Qwen3-14B-Arc-1D-SFT")
```

## Parameter-Efficient Training

Use LoRA for large models:

```python
trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=args,
    peft_config=vf.lora_defaults()  # Add LoRA
)
```

## Infrastructure Setup

### Single GPU Training

```python
# Simple single GPU training
model, tokenizer = vf.get_model_and_tokenizer("Qwen/Qwen2.5-1.5B-Instruct")
args = vf.grpo_defaults(run_name="single-gpu-experiment")
trainer = vf.GRPOTrainer(model=model, processing_class=tokenizer, env=vf_env, args=args)
trainer.train()
```

### Multi-GPU Training

The examples use this infrastructure pattern:

```bash
# Start vLLM inference server (4 GPUs for generation)
CUDA_VISIBLE_DEVICES=0,1,2,3 vf-vllm --model 'Qwen/Qwen2.5-7B-Instruct' \
    --tensor-parallel-size 4 --max-model-len 8192 --dtype bfloat16 \
    --gpu-memory-utilization 0.9 --enable-prefix-caching \
    --host 0.0.0.0 --port 8000

# Run training on separate GPUs (4 GPUs for training)
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config-file configs/zero3.yaml \
    --num-processes 4 your_training_script.py
```

### ZeRO Configuration

The examples use DeepSpeed ZeRO-3. Create `configs/zero3.yaml`:

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
deepspeed_config:
  gradient_accumulation_steps: 1
  gradient_clipping: 1.0
  offload_optimizer_device: cpu
  offload_param_device: cpu
  zero3_init_flag: true
  zero3_save_16bit_model: true
  zero_stage: 3
```

## Environment-Specific Examples

### Math Training

```python
import verifiers as vf

dataset = vf.load_example_dataset("gsm8k", split="train")
eval_dataset = vf.load_example_dataset("gsm8k", split="test")

system_prompt = """
Think step-by-step inside <think>...</think> tags.
Then, give your final numerical answer inside \\boxed{{...}}.
"""

parser = vf.ThinkParser(extract_fn=vf.extract_boxed_answer)

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

# Train
model, tokenizer = vf.get_model_and_tokenizer("Qwen/Qwen2.5-1.5B-Instruct")
args = vf.grpo_defaults(run_name="gsm8k-training")
trainer = vf.GRPOTrainer(model=model, processing_class=tokenizer, env=vf_env, args=args)
trainer.train()
```

### Tool-Augmented Training

```python
import verifiers as vf
from verifiers.tools import python

# Create tool environment
vf_env = vf.ToolEnv(
    dataset=dataset,
    tools=[python],
    max_steps=3
)

# Train with tools
model, tokenizer = vf.get_model_and_tokenizer("Qwen/Qwen2.5-7B-Instruct")
args = vf.grpo_defaults(run_name="tool-training")
trainer = vf.GRPOTrainer(model=model, processing_class=tokenizer, env=vf_env, args=args)
trainer.train()
```

## Alternative Training Approaches

### Using Environments as Reward Functions

You can use environments as reward functions in your own training loops:

```python
# Generate training data with rewards
results = vf_env.generate(
    client=openai_client,
    model="gpt-4",
    n_samples=1000
)

# Process for training
processed = vf_env.process_env_results(
    prompts=results['prompts'],
    completions=results['completions'],
    states=results['states'],
    rewards=results['rewards'],
    processing_class=tokenizer
)

# Use in your own training loop
for batch in processed:
    # Your custom training logic here
    pass
```

### Verifiers for Async FSDP Training

The framework also supports async FSDP environment training through the verifiers package:

```python
# This is supported by prime-rl for async FSDP environment training
# See the verifiers package for more details
```

## Best Practices

### 1. Start with Evaluation
Before training, thoroughly evaluate your environment:

```python
# Test environment with a few examples
results = vf_env.evaluate(
    client=openai_client,
    model="gpt-4",
    num_examples=10
)
print(f"Average reward: {sum(results['rewards']) / len(results['rewards'])}")
```

### 2. Use Appropriate Model Sizes
Start with smaller models for experimentation:

```python
# Start small
model, tokenizer = vf.get_model_and_tokenizer("Qwen/Qwen2.5-1.5B-Instruct")

# Scale up when ready
model, tokenizer = vf.get_model_and_tokenizer("Qwen/Qwen2.5-7B-Instruct")
```

### 3. Monitor Training
Use evaluation datasets and logging:

```python
args = vf.grpo_defaults(run_name="my-experiment")
args.eval_strategy = "steps"
args.eval_steps = 100
args.save_strategy = "steps"
args.save_steps = 500
```

### 4. Handle Infrastructure
For large-scale training, use proper infrastructure:

```bash
# Use vLLM for efficient generation
vf-vllm --model 'Qwen/Qwen2.5-7B-Instruct' --tensor-parallel-size 4

# Use DeepSpeed for efficient training
accelerate launch --config-file configs/zero3.yaml training_script.py
```

## Key Gotchas

1. **Environment Testing**: Always test your environment before training
2. **Reward Scaling**: Ensure rewards are in reasonable ranges (typically 0-1)
3. **Format Rewards**: Always include format compliance in your rubric
4. **Infrastructure**: Use appropriate infrastructure for your model size
5. **Evaluation**: Regular evaluation helps catch training issues early