import verifiers as vf
from verifiers.envs.textarena_env import TextArenaEnv

"""
inference:
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 vf-vllm --model willcb/Qwen3-1.7B-Wordle --tensor-parallel-size 1 --data-parallel-size 6 --enforce-eager

training:
CUDA_VISIBLE_DEVICES=6,7 accelerate launch --config-file configs/zero3.yaml --num-processes 2 verifiers/examples/wordle_nothink.py
"""

model_name = "willcb/Qwen3-1.7B-Wordle"
model, tokenizer = vf.get_model_and_tokenizer(model_name)


NOTHINK_WORDLE_SYSTEM_PROMPT = """You are a competitive game player. \
Make sure you read the game instructions carefully, and always follow the required format.

In each turn, give only your guess inside <guess>...</guess> tags."""

vf_env = TextArenaEnv(
    game="Wordle-v0",
    num_train_examples=2000,
    num_eval_examples=20,
    system_prompt=NOTHINK_WORDLE_SYSTEM_PROMPT,
    parser=vf.XMLParser(fields=["guess"], answer_field="guess"),
)

# partial credit reward function
parser = vf_env.parser


def partial_credit_reward_func(completion, **kwargs) -> float:
    """Reward function that gives partial credit for the correct guess."""
    final_env_response = parser.get_user_messages(completion)[-1]["content"].strip()
    guess, scoring = final_env_response.split("\n")[:2]
    num_greens = scoring.count("G")
    num_yellows = scoring.count("Y")
    return 0.2 * num_greens + 0.1 * num_yellows


vf_env.rubric.add_reward_func(partial_credit_reward_func)

run_name = "wordle-nothink-grpo-1.7b"
training_args = vf.grpo_defaults(run_name=run_name)
training_args.per_device_train_batch_size = 16
training_args.num_generations = 16
training_args.gradient_accumulation_steps = 4
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
