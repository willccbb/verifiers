import verifiers as vf
from verifiers.envs.reasoninggym_env import ReasoningGymEnv

"""
inference:
CUDA_VISIBLE_DEVICES=0,1,2,3 vf-vllm --model willcb/Qwen3-14B-Arc-1D-SFT --tensor-parallel-size 4 --max-batch-size 128

training:
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config-file configs/zero3.yaml --num-processes 4 verifiers/examples/arc_1d.py
"""

size = '14B'
model_name = f'willcb/Qwen3-{size}-Arc-1D-SFT'
model, tokenizer = vf.get_model_and_tokenizer(model_name)

vf_env = ReasoningGymEnv(
    gym="arc_1d",
    num_samples=4000,
    max_concurrent=128,
    seed=1,
)

run_name = f"arc_1d-grpo-{size}"
training_args=vf.grpo_defaults(run_name=run_name)
training_args.num_iterations=1
training_args.per_device_train_batch_size=4
training_args.num_generations=16
training_args.gradient_accumulation_steps=8
training_args.max_concurrent = 512
training_args.max_prompt_length=1024
training_args.max_completion_length=4096
training_args.max_steps=500

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=training_args,
)
trainer.train()