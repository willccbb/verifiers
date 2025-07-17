import verifiers as vf
from verifiers.envs.textarena_env import TextArenaEnv

"""
inference:
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 vf-vllm --model willcb/Qwen3-4B-Wordle --data-parallel-size 7 --enforce-eager

training:
CUDA_VISIBLE_DEVICES=7 accelerate launch --config-file configs/zero3.yaml --num-processes 1 verifiers/examples/wordle.py
"""

model_name = "willcb/Qwen3-4B-Wordle"
model, tokenizer = vf.get_model_and_tokenizer(model_name)

vf_env = TextArenaEnv(
    game="Wordle-v0",
    num_train_examples=2000,
    num_eval_examples=20,
)


# partial credit reward function
def partial_credit_reward_func(parser, completion, **kwargs) -> float:
    """Reward function that gives partial credit for the correct guess."""
    final_env_response = parser.get_user_messages(completion)[-1]["content"].strip()
    guess, scoring = final_env_response.split("\n")[:2]
    num_greens = scoring.count("G")
    num_yellows = scoring.count("Y")
    return 0.2 * num_greens + 0.1 * num_yellows


vf_env.rubric.add_reward_func(partial_credit_reward_func)

run_name = "wordle-grpo-4b"
training_args = vf.grpo_defaults(run_name=run_name)
training_args.per_device_train_batch_size = 8
training_args.num_generations = 8
training_args.gradient_accumulation_steps = 16
training_args.max_prompt_length = 1024
training_args.max_completion_length = 3072
training_args.max_steps = 200
training_args.eval_strategy = "steps"
training_args.eval_steps = 20
training_args.mask_env_responses = True
training_args.beta = 0.0

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=training_args,
)
trainer.train()
