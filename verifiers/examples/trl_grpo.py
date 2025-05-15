# train_grpo.py
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer

"""
CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen2.5-1.5B-Instruct --enable-prefix-caching True --dtype bfloat
16 --max-model-len 2048 --gpu-memory-utilization 0.95

CUDA_VISIBLE_DEVICES=1 accelerate launch --num-processes 1 --config-file configs/zero3.yaml verifiers/examples/trl_grpo.py
"""

dataset: Dataset = load_dataset("trl-lib/tldr", split="train")

# Define the reward function, which rewards completions that are close to 20 characters
def reward_len(completions: list[str], **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]


model_name = "Qwen/Qwen2.5-1.5B-Instruct"
training_args = GRPOConfig(output_dir="Qwen2.5-1.5B-GRPO", logging_steps=10, use_vllm=True)

run_name = "demo-grpo_" + model_name.split("/")[-1].lower()

training_args=GRPOConfig(
    output_dir=f"outputs/{run_name}",
    run_name=run_name,
    learning_rate=1e-6,
    lr_scheduler_type="constant",
    num_train_epochs=1,
    temperature=1.0,
    max_steps=100,
    bf16=True,
    max_grad_norm=0.1,
    num_iterations=1,
    beta=0,
    max_prompt_length=1024,
    max_completion_length=1024,
    per_device_train_batch_size=32,
    num_generations=8,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    save_strategy="steps",
    save_steps=100,
    save_only_model=True,
    use_liger_kernel=True,
    use_liger_loss=True,
    use_vllm=True,
    vllm_server_host="0.0.0.0", # replace with your inference server's host for multi-node setups
    vllm_server_port=8000, 
    vllm_gpu_memory_utilization=0.9,
    logging_steps=1,
    log_on_each_node=False,
    log_completions=True,
    report_to="wandb",
)

trainer = GRPOTrainer(
    model=model_name,
    reward_funcs=[reward_len],
    args=training_args,
    train_dataset=dataset, # type: ignore
)
trainer.train()