import argparse
import verifiers as vf
import os

"""
# install
vf-install vf-bfcl-single-turn (-p /path/to/environments)

# quick eval
vf-eval vf-bfcl-single-turn (-m model_name in endpoints.py)

inference:
CUDA_VISIBLE_DEVICES=0 vf-vllm --model Salesforce/xLAM-2-3b-fc-r \
    --enforce-eager --disable-log-requests \
    --enable-auto-tool-choice --tool-call-parser hermes

training:
CUDA_VISIBLE_DEVICES=1 accelerate launch --num-processes 1 \
    --config-file configs/zero3.yaml examples/grpo/train_bfcl_single_turn.py
"""

def main(args):
    # Load the BFCL single turn environment
    vf_env = vf.load_environment(env_id="vf_bfcl_single_turn", num_eval_examples=100)
    
    # Use xLAM model which is specialized for function calling
    model_name = "Salesforce/xLAM-2-3b-fc-r"
    
    run_name = "bfcl-single-turn-grpo_" + model_name.split("/")[-1].lower()
    
    # Load model and tokenizer
    model, tokenizer = vf.get_model_and_tokenizer(model_name, model_kwargs={"attn_implementation": "sdpa"})
    
    # Set up training arguments
    training_args = vf.grpo_defaults(run_name=run_name)
    
    # Configure wandb reporting based on API key availability
    training_args.report_to = "wandb" if os.environ.get("WANDB_API_KEY") else []
    
    # Optimize for function calling with appropriate batch sizes and context length
    training_args.per_device_train_batch_size = 2  # Small batch size for 3B model
    training_args.num_generations = 8  # Multiple generations for better exploration
    training_args.gradient_accumulation_steps = 16  # Effective batch size of 32
    training_args.max_tokens = 1024  # Sufficient for function calling responses
    training_args.max_seq_len = 4096  # xLAM models support longer context
    training_args.eval_strategy = "steps"
    training_args.eval_steps = 20
    training_args.save_strategy = "steps"
    training_args.save_steps = 100
    training_args.max_steps = int(os.environ.get("VERIFIERS_MAX_STEPS", "500"))
    training_args.learning_rate = 5e-6  # Lower learning rate for fine-tuned model
    training_args.warmup_steps = 50
    
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
    parser.add_argument("--size", "-s", type=str, default="3B", help="Model size (0.5B, 1.7B, 3B)")
    parser.add_argument("--steps", type=int, help="Number of training steps")
    args = parser.parse_args()
    
    if args.debug:
        os.environ["VERIFIERS_MAX_STEPS"] = "50"
    elif args.steps:
        os.environ["VERIFIERS_MAX_STEPS"] = str(args.steps)
    
    main(args)