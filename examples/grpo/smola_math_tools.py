import verifiers as vf

vf_env = vf.load_environment("smolagents_math_tools", use_few_shot=True)

model_name = "Qwen/Qwen2.5-7B-Instruct"
model, tokenizer = vf.get_model_and_tokenizer(model_name)
run_name = "math-smola-grpo_" + model_name.split("/")[-1].lower()

args = vf.grpo_defaults(run_name=run_name)
trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=args,
)
trainer.train()
