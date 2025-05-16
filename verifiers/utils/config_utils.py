from trl import GRPOConfig
from typing import List, Optional

def get_default_grpo_config(run_name: str) -> GRPOConfig:
    return GRPOConfig(
        output_dir=f"outputs/{run_name}",
        run_name=run_name,
        learning_rate=1e-6,
        lr_scheduler_type="constant",
        num_train_epochs=1,
        bf16=True,
        max_grad_norm=0.01,
        num_iterations=1,
        max_prompt_length=1024,
        max_completion_length=1024,
        per_device_train_batch_size=16,
        num_generations=4,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        save_strategy="steps",
        save_steps=100,
        save_only_model=True,
        use_vllm=True,
        logging_steps=1,
        log_on_each_node=False,
        log_completions=True,
        report_to="wandb",
    )

def defaults(run_name: str) -> GRPOConfig:
    return get_default_grpo_config(run_name)

