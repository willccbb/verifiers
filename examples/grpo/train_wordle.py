import argparse
import os

import verifiers as vf

"""
# install
vf-install vf-wordle (-p /path/to/environments)

# quick eval
vf-eval vf-wordle -m (model_name in endpoints.py)

1.7b inference:
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 vf-vllm --model Qwen/Qwen2.5-Instruct \
    --data-parallel-size 6 --enforce-eager --disable-log-requests

1.7b training:
CUDA_VISIBLE_DEVICES=6,7 accelerate launch --num-processes 2 \
    --config-file configs/zero3.yaml examples/grpo/train_wordle.py --size 1.7B

4b inference:
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 vf-vllm --model Qwen/Qwen2.5-3B-Instruct \
    --data-parallel-size 6 --enforce-eager --disable-log-requests

4b training:
CUDA_VISIBLE_DEVICES=6,7 accelerate launch --num-processes 2 \
    --config-file configs/zero3.yaml examples/grpo/train_wordle.py --size 4B
"""


def train(model_size: str, max_steps: int = None):
    model_names = {
        "0.5B": "Qwen/Qwen2.5-0.5B-Instruct",
        "1.7B": "Qwen/Qwen2.5-1.5B-Instruct",
        "4B": "Qwen/Qwen2.5-3B-Instruct",
    }
    assert model_size in model_names, f"Invalid model size: {model_size}"
    model_name = model_names[model_size]

    # Load model and tokenizer (disable flash attention since it's not installed)
    model_kwargs = {"attn_implementation": "eager"}
    model, tokenizer = vf.get_model_and_tokenizer(model_name, model_kwargs=model_kwargs)

    # Load environment
    env = vf.load_environment("vf-wordle")

    # Training
    training_args = vf.GRPOConfig(
        output_dir=f"outputs/wordle-{model_size}-grpo",
        run_name=f"wordle-{model_size}-grpo",
        learning_rate=5e-7,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=10,
        max_steps=max_steps if max_steps is not None else 500,
        bf16=False,  # Disable bf16 on Mac
        fp16=False,  # Also disable fp16
        max_grad_norm=0.1,
        beta=0.0,
        num_iterations=1,
        max_seq_len=2048,
        per_device_train_batch_size=4,  # Reduced to save memory
        num_generations=4,  # Reduced to save memory
        gradient_accumulation_steps=8,  # Increased to maintain effective batch size
        gradient_checkpointing=True,
        save_strategy="steps",
        save_steps=500,
        save_only_model=True,
        logging_steps=1,
        log_on_each_node=False,
        log_completions=True,
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else [],
    )
    
    # Check if we should disable weight sync (when using standard vLLM OpenAI API)
    if os.environ.get("VLLM_USE_OPENAI_CLIENT") == "1":
        training_args.disable_weight_sync = True

    trainer = vf.GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        env=env,
        args=training_args,
        # lora_config=vf.lora_defaults()
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", "-s", type=str, default="1.7B")
    parser.add_argument("--steps", type=int, default=None, help="Maximum number of training steps")
    args = parser.parse_args()
    train(args.size, args.steps)
