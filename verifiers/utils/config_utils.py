from peft import LoraConfig
from typing import List, Optional
from verifiers import GRPOEnvConfig


def get_default_grpo_config(run_name: str) -> GRPOEnvConfig:
    return GRPOEnvConfig(
        output_dir=f"outputs/{run_name}",
        run_name=run_name,
        learning_rate=1e-6,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=10,
        num_train_epochs=1,
        max_steps=500,
        bf16=True,
        max_grad_norm=0.01,
        num_iterations=2,
        max_prompt_length=1024,
        max_completion_length=2048,
        per_device_train_batch_size=12,
        num_generations=6,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        save_strategy="steps",
        save_steps=500,
        save_only_model=True,
        use_vllm=True,
        logging_steps=1,
        log_on_each_node=False,
        log_completions=True,
        report_to="wandb",
    )

def grpo_defaults(run_name: str) -> GRPOConfig:
    return get_default_grpo_config(run_name)

def lora_defaults(r = 8, alpha = 16) -> LoraConfig:
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )