import verifiers as vf

"""
inference:
CUDA_VISIBLE_DEVICES=0 vf-vllm --model willcb/Qwen2.5-0.5B-Reverse-SFT --enforce-eager

training:
CUDA_VISIBLE_DEVICES=1 accelerate launch --num-processes 1 --config-file configs/zero3.yaml verifiers/examples/reverse_text.py
"""

model_name = "willcb/Qwen2.5-0.5B-Reverse-SFT"
model, tokenizer = vf.get_model_and_tokenizer(model_name)

vf_env = vf.load_environment(env_id="reverse_text")

args = vf.grpo_defaults(run_name="reverse_text_warmup")
args.per_device_train_batch_size = 12
args.num_generations = 12
args.gradient_accumulation_steps = 8
args.max_steps = 100
args.eval_strategy = "steps"
args.eval_steps = 2

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    # peft_config=vf.lora_defaults(),
    args=args,
)
trainer.train()
