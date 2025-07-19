import verifiers as vf

"""
inference:
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 vf-vllm --model willcb/Qwen3-1.7B-Wordle --data-parallel-size 7 --enforce-eager

training:
CUDA_VISIBLE_DEVICES=7 accelerate launch --config-file configs/zero3.yaml --num-processes 1 examples/grpo/wordle_nothink.py
"""

model_name = "willcb/Qwen3-1.7B-Wordle"
model, tokenizer = vf.get_model_and_tokenizer(model_name)

vf_env = vf.load_environment(env_id="wordle", use_think=False)

run_name = "wordle-nothink-grpo-1.7b"
training_args = vf.grpo_defaults(run_name=run_name)
training_args.per_device_train_batch_size = 16
training_args.num_generations = 16
training_args.gradient_accumulation_steps = 16
training_args.max_seq_len = 1024
training_args.max_tokens = 16
training_args.max_steps = 500
training_args.mask_env_responses = True

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=training_args,
)
trainer.train()
