import os
import verifiers as vf

if os.getenv("BRAVE_API_KEY"):
    print("Using Brave as a search engine. BRAVE_API_KEY must be set. See https://brave.com/search/api/")
    from verifiers.tools import search_brave as search 
else:
    print(
        "WARNING: Using DuckDuckGo as a search engine. \
        This may be rate limited (which can cause training to fail). \
        Consider setting a paid BRAVE_API_KEY (https://brave.com/search/api/) to use Brave instead."
    )
    from verifiers.tools import search_ddg as search 

from verifiers.prompts import SEARCH_FEW_SHOT

model_name = "Qwen/Qwen2.5-7B-Instruct"
model, tokenizer = vf.get_model_and_tokenizer(model_name)

vf_env = vf.ToolEnv(
    dataset="openbookqa",
    few_shot=SEARCH_FEW_SHOT[0],
    tools=[search],
    max_steps=3
)
train_dataset = vf_env.get_dataset()
rubric = vf_env.get_rubric()

# notable defaults: lr = 1e-6, max_grad_norm = 0.01, constant lr 10 warmup steps, 1024 tokens in+out
run_name = "obqa-search_" + model_name.split("/")[-1].lower()
training_args = vf.get_default_grpo_config(
    run_name=run_name,
    num_gpus=8 # 7 train + 1 inference
)
training_args.learning_rate = 1e-6

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
trainer = vf.GRPOEnvTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=rubric,
    env=vf_env,
    args=training_args,
    train_dataset=train_dataset
)
trainer.train() 

