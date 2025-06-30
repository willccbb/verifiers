import verifiers as vf
from verifiers.utils.data_utils import load_example_dataset, extract_boxed_answer

"""
inference:
CUDA_VISIBLE_DEVICES=0 vf-vllm --model willcb/Qwen3-0.6B

training:
CUDA_VISIBLE_DEVICES=1 accelerate launch --num-processes 1 --config-file configs/zero3.yaml verifiers/examples/gsm8k.py
"""

dataset = load_example_dataset("gsm8k", split="train")
eval_dataset = load_example_dataset("gsm8k", split="test")


system_prompt = """
Think step-by-step inside <think>...</think> tags.

Then, give your final numerical answer inside \\boxed{{...}}.
"""
parser = vf.ThinkParser(extract_fn=extract_boxed_answer)

def correct_answer_reward_func(completion, answer, **kwargs):
    response = parser.parse_answer(completion) or ''
    return 1.0 if response == answer else 0.0

rubric = vf.Rubric(funcs=[
    correct_answer_reward_func,
    parser.get_format_reward_func()
], weights=[1.0, 0.2])

vf_env = vf.SingleTurnEnv(
    dataset=dataset,
    eval_dataset=eval_dataset,
    system_prompt=system_prompt,
    parser=parser,
    rubric=rubric,
)


model_name = "willcb/Qwen3-0.6B"
run_name = "gsm8k-grpo_" + model_name.split("/")[-1].lower()

model, tokenizer = vf.get_model_and_tokenizer(model_name)
training_args=vf.grpo_defaults(run_name=run_name)

training_args.per_device_train_batch_size=18
training_args.num_generations=18
training_args.gradient_accumulation_steps=8
training_args.num_iterations=1
training_args.max_prompt_length=512
training_args.max_completion_length=2048
training_args.max_steps=100
training_args.save_strategy="steps"
training_args.save_steps=10
training_args.save_total_limit=10

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=training_args,
    peft_config=vf.lora_defaults()
)
trainer.train() 



