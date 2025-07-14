import verifiers as vf
from verifiers.envs.textarena_env import TextArenaEnv

"""
inference:
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 vf-vllm --model willcb/Qwen3-1.7B-Wordle --tensor-parallel-size 1 --data-parallel-size 6 --enforce-eager

training:
CUDA_VISIBLE_DEVICES=6,7 accelerate launch --config-file configs/zero3.yaml --num-processes 2 verifiers/examples/wordle_nothink.py
"""

NOTHINK_WORDLE_SYSTEM_PROMPT = """You are a competitive game player. \
Make sure you read the game instructions carefully, and always follow the required format.

In each turn, give only your guess inside <guess>...</guess> tags. /no_think"""


size = '1.7B'
model_name = f'willcb/Qwen3-{size}-Wordle'
model, tokenizer = vf.get_model_and_tokenizer(model_name)

vf_env = TextArenaEnv(
    game="Wordle-v0",
    num_samples=2000, 
    num_eval_samples=20,
    system_prompt=NOTHINK_WORDLE_SYSTEM_PROMPT,
    parser=vf.XMLParser(fields=["guess"], answer_field="guess"),
)

run_name = f"wordle-nothink-grpo-{size}"
training_args=vf.grpo_defaults(run_name=run_name)
training_args.per_device_train_batch_size=16
training_args.num_generations=16
training_args.gradient_accumulation_steps=4
training_args.max_seq_len=1024
training_args.max_tokens=16
training_args.max_steps=500
training_args.eval_strategy="steps"
training_args.eval_steps=10
training_args.mask_env_responses=True

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=training_args,
)
trainer.train()