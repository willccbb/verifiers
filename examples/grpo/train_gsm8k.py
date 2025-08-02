import argparse
import verifiers as vf
import os

"""
# install
vf-install vf-gsm8k (-p /path/to/environments)

# quick eval
vf-eval vf-gsm8k (-m model_name in endpoints.py)

inference:
CUDA_VISIBLE_DEVICES=0 vf-vllm --model Qwen/Qwen2.5-0.5B-Instruct --enforce-eager --disable-log-requests

training:
CUDA_VISIBLE_DEVICES=1 accelerate launch --num-processes 1 \
    --config-file configs/zero3.yaml examples/grpo/train_gsm8k.py
"""

def main(args):
    size = args.size
    # Use Qwen2.5 models based on size
    if size == "1.7B":
        model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    elif size == "4B":
        model_name = "Qwen/Qwen2.5-3B-Instruct"
    else:
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    vf_env = vf.load_environment(env_id="vf_gsm8k", num_eval_examples=100)

    run_name = "gsm8k-grpo_" + model_name.split("/")[-1].lower()

    model, tokenizer = vf.get_model_and_tokenizer(model_name, model_kwargs={"attn_implementation": "sdpa"})
    training_args = vf.grpo_defaults(run_name=run_name)

    # Optimized for Modal single GPU setup with memory constraints
    training_args.per_device_train_batch_size = 1  # Minimal batch size to save memory
    training_args.num_generations = 4  # Minimal generations to save memory  
    training_args.gradient_accumulation_steps = 16  # Increase grad accumulation to maintain effective batch size
    training_args.max_tokens = 300   # Reduce max tokens per generation to fit in 1024 context
    training_args.max_seq_len = 1024  # Reduce max sequence length
    training_args.eval_strategy = "steps"
    training_args.eval_steps = 10
    training_args.save_strategy = "steps"
    training_args.save_steps = 100
    training_args.max_steps = int(os.environ.get("VERIFIERS_MAX_STEPS", "200"))
    training_args.eval_strategy = "steps"
    training_args.eval_steps = 10
    
    # Check if we should disable weight sync (when using standard vLLM OpenAI API)
    if os.environ.get("VLLM_USE_OPENAI_CLIENT") == "1":
        training_args.disable_weight_sync = True

    trainer = vf.GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        env=vf_env,
        args=training_args,
        peft_config=vf.lora_defaults(),
    )
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", "-s", type=str, default="0.5B")
    args = parser.parse_args()
    main(args)
