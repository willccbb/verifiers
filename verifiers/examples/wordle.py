import verifiers as vf
from verifiers.envs.textarena_env import TextArenaEnv

"""
first time:
import nltk
nltk.download('words', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

inference:
CUDA_VISIBLE_DEVICES=0,1,2,3 vf-vllm --model willcb/Qwen2.5-7B-Wordle-SFT --tensor-parallel-size 4

training:
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config-file configs/zero3.yaml --num-processes 4 verifiers/examples/wordle.py
"""

size = '7B'
model_name = f'willcb/Qwen2.5-{size}-Wordle-SFT'
model, tokenizer = vf.get_model_and_tokenizer(model_name)

vf_env = TextArenaEnv(
    game="Wordle-v0",
    num_samples=2000, 
    num_eval_samples=20,
    max_concurrent=32,
)

run_name = f"wordle-grpo-{size}"
training_args=vf.grpo_defaults(run_name=run_name)
training_args.num_iterations=1
training_args.per_device_train_batch_size=6
training_args.num_generations=12
training_args.gradient_accumulation_steps=4
training_args.max_prompt_length=1024
training_args.max_completion_length=4096
training_args.max_steps=100
training_args.mask_env_responses=True

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=training_args,
)
trainer.train()

