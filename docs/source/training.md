# Training

The verifiers framework integrates with Group Relative Policy Optimization (GRPO) to train language models using reward signals from your environments. Training is designed to be simple and scalable.

## Quick Start

Here's the basic pattern used in all examples:

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

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
model, tokenizer = vf.get_model_and_tokenizer(model_name)

args = vf.grpo_defaults(run_name="gsm8k-grpo")
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

### Tool Training

```python
import verifiers as vf
from verifiers.tools import python

TOOL_PROMPT = """
Think step-by-step inside <think>...</think> tags, then either call a tool or give your final answer.

Tools can be called with JSON:
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

model_name = "willcb/Qwen2.5-7B-Math-Python-SFT"
model, tokenizer = vf.get_model_and_tokenizer(model_name)

args = vf.grpo_defaults(run_name="math-tool-grpo")
args.num_iterations = 2
args.per_device_train_batch_size = 8
args.num_generations = 8

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=args,
)
trainer.train()
```

### Game Training

```python
import verifiers as vf
from verifiers.envs.textarena_env import TextArenaEnv

model_name = 'willcb/Qwen2.5-7B-Wordle-SFT'
model, tokenizer = vf.get_model_and_tokenizer(model_name)

vf_env = TextArenaEnv(
    game="Wordle-v0",
    num_samples=2000, 
    num_eval_samples=20
)

args = vf.grpo_defaults(run_name="wordle-grpo")
args.num_iterations = 1
args.per_device_train_batch_size = 8
args.num_generations = 16
args.gradient_accumulation_steps = 6
args.max_prompt_length = 1024
args.max_completion_length = 3072
args.max_steps = 100
args.mask_env_responses = True

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=args,
)
trainer.train()
```

## Advanced Configuration

### Custom Training Arguments

```python
args = vf.grpo_defaults(run_name="my-experiment")

# Scale up for larger experiments
args.per_device_train_batch_size = 4
args.num_generations = 32
args.gradient_accumulation_steps = 8
args.max_concurrent = 512
args.max_steps = 1000

# Memory optimization
args.gradient_checkpointing = True
args.bf16 = True

# Learning schedule
args.learning_rate = 1e-6
args.lr_scheduler_type = "constant_with_warmup"
args.warmup_steps = 10

# Logging and saving
args.logging_steps = 1
args.save_strategy = "steps"
args.save_steps = 100
args.report_to = "wandb"
```

### Environment Masking

For interactive environments, mask environment responses:

```python
args.mask_env_responses = True  # Don't train on environment responses
args.mask_truncated_completions = True  # Ignore truncated outputs
```

### Concurrent Generation

```python
args.max_concurrent = 512  # Parallel rollouts
args.async_generation_timeout = 300.0  # Timeout for generation
```

## Training Scripts

### Complete Training Script

```python
import verifiers as vf

def main():
    # Load dataset and create environment
    dataset = vf.load_example_dataset("gsm8k", split="train")
    vf_env = vf.SingleTurnEnv(dataset=dataset)
    
    # Load model
    model, tokenizer = vf.get_model_and_tokenizer("Qwen/Qwen2.5-1.5B-Instruct")
    
    # Configure training
    args = vf.grpo_defaults(run_name="my-experiment")
    args.max_steps = 100
    
    # Create and run trainer
    trainer = vf.GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        env=vf_env,
        args=args,
    )
    trainer.train()

if __name__ == "__main__":
    main()
```

### Multi-GPU Script

Save as `train.py`:

```python
import verifiers as vf

def main():
    dataset = vf.load_example_dataset("math", "train", n=6000)
    vf_env = vf.ToolEnv(dataset=dataset, tools=[vf.tools.python])
    
    model, tokenizer = vf.get_model_and_tokenizer("Qwen/Qwen2.5-7B-Instruct")
    args = vf.grpo_defaults(run_name="multi-gpu-experiment")
    
    trainer = vf.GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        env=vf_env,
        args=args,
    )
    trainer.train()

if __name__ == "__main__":
    main()
```

Run with:
```bash
accelerate launch --config-file configs/zero3.yaml --num-processes 4 train.py
```

## Monitoring Training

### Weights & Biases Integration

```python
args = vf.grpo_defaults(run_name="my-experiment")
args.report_to = "wandb"
args.log_completions = True  # Log sample completions

# Will automatically log:
# - Training loss
# - Reward statistics
# - Sample completions
# - Model metrics
```

### Manual Evaluation

```python
# During training, evaluate on test set
eval_results = trainer.env.evaluate(
    client=trainer.oai_client,
    model=trainer._get_model_name(),
    sampling_args=trainer._get_sampling_args(),
    num_samples=100
)
print(f"Eval reward: {eval_results['reward']}")
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   args.per_device_train_batch_size = 2  # Reduce batch size
   args.gradient_checkpointing = True     # Enable checkpointing
   args.bf16 = True                       # Use half precision
   ```

2. **Slow Training**
   ```python
   args.max_concurrent = 256              # Increase concurrency
   args.gradient_accumulation_steps = 8   # Larger effective batch
   ```

3. **vLLM Connection Issues**
   ```bash
   # Check vLLM server is running
   curl http://localhost:8000/health
   
   # Check GPU usage
   nvidia-smi
   ```

### Performance Tips

1. **Use appropriate batch sizes**: Start small and scale up
2. **Enable async generation**: Set `max_concurrent` appropriately
3. **Use LoRA for large models**: Add `peft_config=vf.lora_defaults()`
4. **Monitor GPU utilization**: Ensure GPUs are fully used
5. **Use bf16**: Enable mixed precision training

The training framework is designed to scale from single GPU experiments to large multi-GPU production runs. All examples follow the same basic patterns, making it easy to reproduce and extend the training setups.