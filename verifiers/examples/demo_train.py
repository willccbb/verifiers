from trl import GRPOConfig

import verifiers as vf
from verifiers.tools import python
from verifiers.utils import preprocess_dataset

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

"""
2-GPU training (single node, 1 training + 1 inference)

CUDA_VISIBLE_DEVICES=0 python verifiers/inference/vllm_serve.py --model 'Qwen/Qwen2.5-1.5B-Instruct' --max_model_len 4096 --dtype bfloat16 --gpu_memory_utilization 0.95 --enable_prefix_caching True
CUDA_VISIBLE_DEVICES=1 accelerate launch --num-processes 1 --config-file configs/zero3.yaml verifiers/examples/demo_train.py
---
4-GPU training (single node, 2 training + 2 inference)

CUDA_VISIBLE_DEVICES=0,1 python verifiers/inference/vllm_serve.py --model 'Qwen/Qwen2.5-1.5B-Instruct' --max_model_len 4096 --dtype bfloat16 --gpu_memory_utilization 0.95 --enable_prefix_caching True
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num-processes 2 --config-file configs/zero3.yaml verifiers/examples/demo_train.py
"""

SIMPLE_PROMPT = """
You are a helpful assistant. In each turn, think step-by-step inside <think>...</think> tags, then give your final answer inside <answer>...</answer> tags.
"""

dataset = preprocess_dataset("math", "train", n=1000)

#vf_env = vf.SingleTurnEnv(
vf_env = vf.DoubleCheckEnv(
    dataset=dataset,
    system_prompt=SIMPLE_PROMPT,
    few_shot=[]
)
print(vf_env.system_prompt)

model, tokenizer = vf.get_model_and_tokenizer(model_name)
run_name = "demo-grpo_" + model_name.split("/")[-1].lower()

training_args=GRPOConfig(
    output_dir=f"outputs/{run_name}",
    run_name=run_name,
    learning_rate=1e-6,
    lr_scheduler_type="constant",
    num_train_epochs=1,
    temperature=1.0,
    max_steps=100,
    bf16=True,
    max_grad_norm=0.1,
    num_iterations=1,
    beta=0,
    max_prompt_length=512,
    max_completion_length=1536,
    per_device_train_batch_size=16,
    num_generations=4,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    save_strategy="steps",
    save_steps=100,
    save_only_model=True,
    use_vllm=True,
    logging_steps=1,
    log_on_each_node=False,
    log_completions=True,
    report_to="wandb",
    reward_weights=vf_env.get_reward_weights()
)

trainer = vf.GRPOEnvTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=vf_env.get_reward_funcs(),
    env=vf_env,
    args=training_args,
    train_dataset=vf_env.get_dataset()
)
trainer.train()