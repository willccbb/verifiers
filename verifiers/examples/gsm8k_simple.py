import verifiers as vf
import torch 

model_name = "Qwen/Qwen2.5-Math-1.5B"
model_kwargs = dict(
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    use_cache=False,
    device_map="cuda", # must use in order to avoid `model.to('cuda')` error from Flash Atention
)
model, tokenizer = vf.get_model_and_tokenizer(model_name, model_kwargs=model_kwargs)

vf_env = vf.MathEnv(dataset="gsm8k")
dataset = vf_env.get_dataset()
rubric = vf_env.get_rubric()
training_args = vf.get_default_grpo_config(run_name="gsm8k_qwen2.5-math-1.5b", num_gpus=2)
trainer = vf.GRPOEnvTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=rubric, 
    env=vf_env,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
