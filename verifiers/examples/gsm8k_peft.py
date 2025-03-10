import verifiers as vf
from verifiers.prompts import CODE_PROMPT

model_name = "Qwen/Qwen2.5-7B-Instruct"
model, tokenizer = vf.get_model_and_tokenizer(model_name)

vf_env = vf.CodeEnv(dataset="gsm8k", few_shot=[], system_prompt=CODE_PROMPT)
dataset = vf_env.get_dataset()
#eval_dataset = vf_env.get_eval_dataset(n=20)
rubric = vf_env.get_rubric()

# notable defaults: lr = 1e-6, max_grad_norm = 0.01, constant lr 10 warmup steps, 1024 tokens in+out
run_name = "gsm8k-code-peft_" + model_name.split("/")[-1].lower()
training_args = vf.get_default_grpo_config(run_name=run_name, num_gpus=8)
# rollouts per prompt
training_args.num_generations = 21
# minibatch size per GPU ( bs 12 * 7 gpus / 21 rollouts -> 4 prompts per batch)
training_args.per_device_train_batch_size = 12
# batches to accumulate (4 prompts * 2 -> 8 prompts per global batch)
training_args.gradient_accumulation_steps = 4
# steps per global batch (1 on-policy, 1 off-policy)
training_args.num_iterations = 2
training_args.max_steps = 100
training_args.beta = 0.01
# evals
# training_args.eval_strategy = "steps"
# training_args.per_device_eval_batch_size = 8
# training_args.eval_steps = 100
# lora
from peft import LoraConfig # type: ignore
peft_config = LoraConfig(
    r=16,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj"]
)

trainer = vf.GRPOEnvTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=rubric, 
    env=vf_env,
    args=training_args,
    train_dataset=dataset,
    #eval_dataset=eval_dataset,
    peft_config=peft_config
)
trainer.train()