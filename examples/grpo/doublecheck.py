import verifiers as vf

SIMPLE_PROMPT = """\
You are a helpful assistant. In each turn, think step-by-step inside <think>...</think> tags, then give your final answer inside <answer>...</answer> tags.
"""

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
vf_env = vf.load_environment(env_id="doublecheck")
model, tokenizer = vf.get_model_and_tokenizer(model_name)
args = vf.grpo_defaults(run_name=f"doublecheck-{model_name.split('/')[-1].lower()}")
trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=args,
)
trainer.train()
