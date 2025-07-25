# Training and Evaluation

The verifiers framework supports both evaluation and training workflows. Environments are powerful evaluation tools that can also be used for training models using reinforcement learning.

## Environment Evaluation

Environments are not just for training - they're excellent evaluation tools:

```python
import verifiers as vf
from openai import OpenAI

# Create or load environment
vf_env = vf.load_environment("math-python", dataset_name="math", num_train_examples=1000)

# Evaluate model performance
client = OpenAI()
results = vf_env.evaluate(
    client=client,
    model="gpt-4.1-mini",
    num_examples=100,
    rollouts_per_example=3
)

print(f"Average reward: {sum(results['rewards']) / len(results['rewards'])}")
print(f"Correct answers: {sum(1 for r in results['rewards'] if r > 0.8)}")

# Generate training data
results = vf_env.generate(
    client=client,
    model="gpt-4.1-mini",
    n_samples=1000
)

# Save results for future use
vf_env.make_dataset(results, push_to_hub=True, hub_name="my-training-data")
```

## Training Approaches

### PRIME-RL (Recommended)

**Unless you require LoRA support, we now generally recommend** that you use the `prime-rl` trainer, which natively supports Environments created using `verifiers`, is more optimized for performance and scalability via FSDP, includes a broader set of configuration options and user experience features, and has more exhaustively battle-tested defaults.

```python
# Example prime-rl integration (see prime-rl docs for complete setup)
import verifiers as vf

# Create environment
vf_env = vf.load_environment("math-python")

# prime-rl natively supports verifiers environments
# See https://github.com/PrimeIntellect-ai/prime-rl for usage instructions
```

Both `prime-rl` and the included `GRPOTrainer` support asynchronous rollouts, and use a one-step off-policy delay by default for overlapping training and inference.

### GRPOTrainer (For LoRA and Smaller Setups)

The included trainer (`vf.GRPOTrainer`) supports running GRPO-style RL training via Accelerate/DeepSpeed, and uses vLLM for inference. It supports both full-parameter finetuning and LoRA, and is optimized for efficiently training dense transformer models on 2-16 GPUs.

```python
import verifiers as vf

# 1. Create environment
vf_env = vf.load_environment("math-python")

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

## Infrastructure Setup

### vLLM Inference Server

For training, set up a vLLM server for generation:

```bash
# Start vLLM inference server
CUDA_VISIBLE_DEVICES=0,1,2,3 vf-vllm --model Qwen/Qwen2.5-7B-Instruct \
    --data-parallel-size 4 --enforce-eager --disable-log-requests \
    --host 0.0.0.0 --port 8000
```

### Multi-GPU Training Example

Example infrastructure pattern for larger models:

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

### Troubleshooting

- Ensure your `wandb` and `huggingface-cli` logins are set up (or set `report_to=None` in `training_args`)
- You should have something set as your `OPENAI_API_KEY` in your environment (can be a dummy key for vLLM)
- If using high max concurrency, increase the number of allowed open sockets (e.g. `ulimit -n 4096`)
- On some setups, inter-GPU communication can hang or crash during vLLM weight syncing. This can usually be alleviated by setting (or unsetting) `NCCL_P2P_DISABLE=1` in your environment
- If problems persist, please open an issue

### Resource Requirements

`GRPOTrainer` is optimized for setups with at least 2 GPUs, scaling up to multiple nodes. 2-GPU setups with sufficient memory to enable small-scale experimentation can be rented for <$1/hr.

## GRPO Configuration (GRPOTrainer)

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

## Environment-Specific Examples

### Math Training

```python
import verifiers as vf

# Load environment
vf_env = vf.load_environment("math-python", dataset_name="math", num_train_examples=5000)

# Train with GRPOTrainer
model, tokenizer = vf.get_model_and_tokenizer("Qwen/Qwen2.5-1.5B-Instruct")
args = vf.grpo_defaults(run_name="math-training")
trainer = vf.GRPOTrainer(model=model, processing_class=tokenizer, env=vf_env, args=args)
trainer.train()
```

### Tool-Augmented Training

```python
import verifiers as vf

# Create tool environment
vf_env = vf.load_environment("math-python")  # Already includes Python tools

# Train with tools
model, tokenizer = vf.get_model_and_tokenizer("Qwen/Qwen2.5-7B-Instruct")
args = vf.grpo_defaults(run_name="tool-training")
trainer = vf.GRPOTrainer(model=model, processing_class=tokenizer, env=vf_env, args=args)
trainer.train()
```

### Game Environment Training

```python
import verifiers as vf

# Load game environment
vf_env = vf.load_environment("wordle", use_think=True, num_train_examples=2000)

# Train on games
model, tokenizer = vf.get_model_and_tokenizer("Qwen/Qwen2.5-1.5B-Instruct")
args = vf.grpo_defaults(run_name="wordle-training")
trainer = vf.GRPOTrainer(model=model, processing_class=tokenizer, env=vf_env, args=args)
trainer.train()
```

## Alternative Training Approaches

### Using Environments as Reward Functions

You can use environments as reward functions in your own training loops:

```python
# Generate training data with rewards
results = vf_env.generate(
    client=client,
    model="gpt-4.1-mini",
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

### OpenAI-Compatible Client Integration

Verifiers can easily be integrated into any RL framework which exposes an OpenAI-compatible inference client:

```python
from openai import OpenAI

# Any OpenAI-compatible client works
client = OpenAI(base_url="https://your-endpoint.com/v1", api_key="your-key")

# Use with environments
results = vf_env.evaluate(client=client, model="your-model", num_examples=100)
```

## ZeRO Configuration

For multi-GPU training with GRPOTrainer, create `configs/zero3.yaml`:

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

## Best Practices

### 1. Start with Evaluation
Before training, thoroughly evaluate your environment:

```python
# Test environment with a few examples
results = vf_env.evaluate(
    client=client,
    model="gpt-4.1-mini",
    num_examples=10
)
print(f"Average reward: {sum(results['rewards']) / len(results['rewards'])}")
```

### 2. Use Appropriate Infrastructure
- **For large-scale training**: Use prime-rl with FSDP
- **For LoRA or smaller setups**: Use GRPOTrainer
- **For evaluation only**: Any OpenAI-compatible client

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

# Use DeepSpeed for efficient training (with GRPOTrainer)
accelerate launch --config-file configs/zero3.yaml training_script.py
```

## TODO Sections

TODO: Add documentation for:
- SFT warmup patterns for improving small-model training efficiency
- RL + GRPO best practices 
- Hardware considerations for different model sizes
- Integration patterns with different RL frameworks

## Key Gotchas

1. **Environment Testing**: Always test your environment before training
2. **Reward Scaling**: Ensure rewards are in reasonable ranges (typically 0-1)
3. **Format Rewards**: Always include format compliance in your rubric
4. **Infrastructure Choice**: Use prime-rl for large-scale training, GRPOTrainer for LoRA/smaller setups
5. **Evaluation**: Regular evaluation helps catch training issues early