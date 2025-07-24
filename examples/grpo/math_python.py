import verifiers as vf

"""
Multi-GPU training (single node, 4 training + 4 inference)

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 vf-vllm --model 'willcb/Qwen3-4B' --data-parallel-size 6

CUDA_VISIBLE_DEVICES=6,7 accelerate launch --config-file configs/zero3.yaml verifiers/examples/math_train.py
"""

vf_env = vf.load_environment(env_id="math_python")

model_name = "willcb/Qwen3-4B"
model, tokenizer = vf.get_model_and_tokenizer(model_name)
run_name = "math-python_" + model_name.split("/")[-1].lower()

training_args = vf.grpo_defaults(run_name=run_name)
training_args.num_iterations = 2
training_args.per_device_train_batch_size = 8
training_args.num_generations = 8
training_args.gradient_accumulation_steps = 2

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=training_args,
)
trainer.train()
