import verifiers as vf

model_name = "Qwen/Qwen2.5-7B-Instruct"
vf_env = vf.load_environment(env_id="self-reward", model_name=model_name)
model, tokenizer = vf.get_model_and_tokenizer(model_name)
trainer = vf.GRPOTrainer(
    env=vf_env,
    model=model,
    processing_class=tokenizer,
    args=vf.grpo_defaults(run_name="self-reward"),
)
trainer.train()
