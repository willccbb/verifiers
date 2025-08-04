import argparse
import verifiers as vf
import os

"""
# GRPO Training for Function Calling with BFCL Dataset

This script trains models on the Berkeley Function Calling Leaderboard (BFCL) dataset
with optimizations from the terminal-bench-rl study:
- Higher temperature (1.2) for exploration diversity
- Lower KL penalty (beta=0.001) for more exploration
- 16 rollouts per batch for better policy learning
- Conservative gradient clipping (0.1)
- GRPO-specific settings: dr_grpo loss, PPO clipping, reward scaling

# install
vf-install vf-bfcl-single-turn

# quick eval
vf-eval vf-bfcl-single-turn (-m model_name in endpoints.py)

# inference:
CUDA_VISIBLE_DEVICES=0 vf-vllm --model Qwen/Qwen2.5-0.5B-Instruct \
    --enforce-eager --disable-log-requests \
    --enable-auto-tool-choice --tool-call-parser hermes

# training:
CUDA_VISIBLE_DEVICES=1 accelerate launch --num-processes 1 \
    --config-file configs/zero3.yaml examples/grpo/train_bfcl.py
"""

def main(args):
    # Load the BFCL single turn environment for real-world function calling
    vf_env = vf.load_environment(env_id="vf_bfcl_single_turn", num_eval_examples=100)
    
    # Use Qwen model for function calling
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    run_name = "bfcl-grpo_" + model_name.split("/")[-1].lower()
    
    # Load model and tokenizer
    model, tokenizer = vf.get_model_and_tokenizer(model_name, model_kwargs={"attn_implementation": "sdpa"})
    
    # Set up training arguments
    # Create GRPOConfig directly to avoid bf16 issue on CPU/Mac
    from verifiers.trainers import GRPOConfig
    training_args = GRPOConfig(
        output_dir=f"outputs/{run_name}",
        run_name=run_name,
        learning_rate=5e-7,  # Lower learning rate for more stable training
        lr_scheduler_type="cosine",  # Cosine schedule for smoother decay
        warmup_steps=100,  # Longer warmup for stability
        max_steps=int(os.environ.get("VERIFIERS_MAX_STEPS", "200")),  # More steps for convergence
        bf16=False,  # Disable bf16 for CPU/Mac
        fp16=False,  # Disable fp16 for CPU/Mac
        max_grad_norm=0.1,  # Terminal-bench-rl uses 0.1 for more conservative clipping
        num_iterations=1,
        max_seq_len=4096,
        per_device_train_batch_size=8,  # Reduced for stability
        per_device_eval_batch_size=16,  # Must be divisible by num_generations
        num_generations=16,  # Terminal-bench-rl uses 16 rollouts for better exploration
        gradient_accumulation_steps=4,  # Effective batch size of 32
        gradient_checkpointing=True,
        save_strategy="steps",
        save_steps=50,  # Save more frequently
        save_only_model=True,
        logging_steps=1,
        log_on_each_node=False,
        log_completions=True,
        eval_strategy="steps",
        eval_steps=25,  # Evaluate every 25 steps
        max_tokens=1024,  # Increased for function calling responses
        # Terminal-bench-rl optimizations
        temperature=1.2,  # Higher temperature for diversity (terminal-bench-rl finding)
        beta=0.001,  # Lower KL penalty for more exploration (terminal-bench-rl finding)
        # GRPO-specific settings from best practices
        loss_type="dr_grpo",  # Length-unbiased loss
        epsilon=0.2,  # PPO-style clipping
        scale_rewards=True,
        mask_env_responses=True,
        mask_truncated_completions=True,
    )
    
    # Configure wandb reporting based on API key availability
    training_args.report_to = "wandb" if os.environ.get("WANDB_API_KEY") else []
    
    
    # Check if we should disable weight sync (when using standard vLLM OpenAI API)
    if os.environ.get("VLLM_USE_OPENAI_CLIENT") == "1":
        training_args.disable_weight_sync = True
    
    # Create trainer with LoRA configuration
    trainer = vf.GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        env=vf_env,
        args=training_args,
        peft_config=vf.lora_defaults(),
    )
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with fewer steps")
    parser.add_argument("--steps", type=int, help="Number of training steps")
    args = parser.parse_args()
    
    if args.debug:
        os.environ["VERIFIERS_MAX_STEPS"] = "50"
    elif args.steps:
        os.environ["VERIFIERS_MAX_STEPS"] = str(args.steps)
    
    main(args)
