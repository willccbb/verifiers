from datasets import concatenate_datasets
from trl import GRPOConfig

import verifiers as vf
from verifiers.tools import search, python, ask
from verifiers.utils import preprocess_dataset

"""
Multi-GPU training (single node, 4 training + 4 inference)

CUDA_VISIBLE_DEVICES=0,1,2,3 python verifiers/inference/vllm_serve.py --model 'Qwen/Qwen2.5-7B-Instruct' \
    --tensor_parallel_size 4 --max_model_len 8192 --dtype bfloat16 \
    --gpu_memory_utilization 0.9 --enable_prefix_caching True \
    --host 0.0.0.0 --port 8000

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config-file configs/zero3.yaml verifiers/examples/math_train.py
"""

TOOL_PROMPT = """
Think step-by-step inside <reasoning>...</reasoning> tags, then either call a tool inside <tool>...</tool> tags, or give your final answer inside <answer>...</answer> tags.

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

Example for multiple choice questions:
<answer>
A
</answer>

Example for math problems:
<answer>
\\frac{{1}}{{2}}
</answer>
"""

dataset = preprocess_dataset("math", "train", n=6000)

eval_aime24 = preprocess_dataset("aime2024", n=30)
eval_aime25 = preprocess_dataset("aime2025", n=30)
eval_dataset = concatenate_datasets([eval_aime24, eval_aime25]).shuffle(seed=0)

vf_env = vf.ToolEnv(
    dataset=dataset,
    eval_dataset=eval_dataset,
    system_prompt=TOOL_PROMPT,
    few_shot=[],
    tools=[python],
    max_steps=5
)
print(vf_env.system_prompt)

model_name = "Qwen/Qwen2.5-7B-Instruct"
model, tokenizer = vf.get_model_and_tokenizer(model_name)
run_name = "math-grpo_" + model_name.split("/")[-1].lower()

training_args=GRPOConfig(
    output_dir=f"outputs/{run_name}",
    run_name=run_name,
    learning_rate=1e-6,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=10,
    num_train_epochs=1,
    temperature=1.0,
    max_steps=1000,
    bf16=True,
    max_grad_norm=0.1,
    num_iterations=2,
    beta=0.002,
    max_prompt_length=1024,
    max_completion_length=2048,
    per_device_train_batch_size=12,
    per_device_eval_batch_size=12,
    num_generations=8,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    eval_strategy="steps",
    eval_steps=100,
    eval_accumulation_steps=1,
    eval_on_start=False,
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
trainer = vf.GRPOEnvTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=vf_env.get_reward_funcs(),
    env=vf_env,
    args=training_args,
    train_dataset=vf_env.get_dataset(),
    eval_dataset=vf_env.get_eval_dataset()
)
trainer.train() 