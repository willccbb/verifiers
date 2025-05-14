from peft import LoraConfig
from trl import GRPOConfig

import verifiers as vf
from verifiers.tools import python
from verifiers.utils import preprocess_dataset

model_name = "Qwen/Qwen3-1.7B"

"""
2-GPU training (single node, 1 training + 1 inference)

CUDA_VISIBLE_DEVICES=0 python verifiers/inference/vllm_serve.py --model 'Qwen/Qwen3-1.7B' --max_model_len 2048 --dtype bfloat16 --gpu_memory_utilization 0.95 --enable_prefix_caching True

CUDA_VISIBLE_DEVICES=1 accelerate launch --num-processes 1 --config-file configs/zero3.yaml verifiers/examples/demo_train.py
"""

TOOL_PROMPT = """
Think step-by-step inside <think>...</think> tags, then either call a tool inside <tool>...</tool> tags, or give your final answer inside <answer>...</answer> tags.

You have access to the following tools to help solve problems:

{tool_descriptions}

Tools can be called by writing a JSON command inside <tool> tags with:
- "name": the name of the tool to use
- "args": the arguments for the tool

Example usage:
<tool>
{{"name": "python", "args": {{"code": "import sympy\nx = sympy.symbols('x')\nprint(sympy.solve(x**2 - 4, x))"}}}}
</tool>

You will then see the tool's output inside <result> tags. You may call tools multiple times if needed.

The <answer>...</answer> tags should contain only your final answer.
"""

dataset = preprocess_dataset("math", "train", n=1000)

vf_env = vf.ToolEnv(
    dataset=dataset,
    system_prompt=TOOL_PROMPT,
    few_shot=[],
    tools=[python],
    llm_fields=["think", ("tool", "answer")],
    env_fields=["result"],
    max_steps=3
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
    max_prompt_length=1024,
    max_completion_length=1024,
    per_device_train_batch_size=16,
    num_generations=4,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    save_strategy="steps",
    save_steps=100,
    save_only_model=True,
    use_vllm=True,
    vllm_server_host="0.0.0.0", # replace with your inference server's host for multi-node setups
    vllm_server_port=8000,
    vllm_gpu_memory_utilization=0.9,
    logging_steps=1,
    log_on_each_node=False,
    log_completions=True,
    report_to="wandb",
    reward_weights=vf_env.get_reward_weights()
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    task_type="CAUSAL_LM",
)

trainer = vf.GRPOEnvTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=vf_env.get_reward_funcs(),
    env=vf_env,
    args=training_args,
    train_dataset=vf_env.get_dataset(),
    eval_dataset=vf_env.get_eval_dataset(),
    peft_config=peft_config
)
trainer.train()