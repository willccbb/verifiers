from peft import LoraConfig

from verifiers.trainers.grpo_config import GRPOConfig

def grpo_defaults(run_name: str) -> GRPOConfig:
    return GRPOConfig(
        output_dir=f"outputs/{run_name}",
        run_name=run_name,
        learning_rate=1e-6,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=10,
        num_train_epochs=1,
        max_steps=500,
        bf16=True,
        max_grad_norm=0.001,
        num_iterations=2,
        max_prompt_length=1024,
        max_completion_length=2048,
        per_device_train_batch_size=8,
        num_generations=8,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        save_strategy="steps",
        save_steps=500,
        save_only_model=True,
        logging_steps=1,
        log_on_each_node=False,
        log_completions=True,
        report_to="wandb",
    )

def lora_defaults(r = 8, alpha = 16) -> LoraConfig:
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )