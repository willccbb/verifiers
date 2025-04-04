from datasets import concatenate_datasets
from trl import GRPOConfig

import verifiers as vf
from verifiers.utils import preprocess_dataset
from verifiers.prompts.smola_templates import MATH_SMOLA_PROMPT_TEMPLATE
from verifiers.prompts.smola_few_shots import CALCULATOR_SMOLA_FEW_SHOTS
from verifiers.envs.smola_tool_env import SmolaToolEnv
from smolagents.default_tools import PythonInterpreterTool
from verifiers.tools.smolagents import CalculatorTool

"""
Multi-GPU training (single node, 4 training + 4 inference) using SmolaAgents tools

CUDA_VISIBLE_DEVICES=0,1,2,3 python verifiers/inference/vllm_serve.py --model 'Qwen/Qwen2.5-7B-Instruct' \
    --tensor_parallel_size 4 --max_model_len 8192 --dtype bfloat16 \
    --gpu_memory_utilization 0.9 --enable_prefix_caching True \
    --host 0.0.0.0 --port 8000

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config-file configs/zero3.yaml verifiers/examples/smola_math_train.py
"""

dataset = preprocess_dataset("math", "train", n=6000)

eval_aime24 = preprocess_dataset("aime2024", n=30)
eval_aime25 = preprocess_dataset("aime2025", n=30)
eval_dataset = concatenate_datasets([eval_aime24, eval_aime25]).shuffle(seed=0)

# Use SmolaAgents' PythonInterpreterTool as a replacement for the python tool
python_tool = PythonInterpreterTool(
    authorized_imports=["math", "sympy", "numpy"]
)
# Add our custom calculator tool
calculator_tool = CalculatorTool()

vf_env = SmolaToolEnv(
    dataset=dataset,
    eval_dataset=eval_dataset,
    system_prompt=MATH_SMOLA_PROMPT_TEMPLATE,
    few_shot=CALCULATOR_SMOLA_FEW_SHOTS,
    tools=[python_tool, calculator_tool],
    max_steps=5
)
print(vf_env.system_prompt)

model_name = "Qwen/Qwen2.5-7B-Instruct"
model, tokenizer = vf.get_model_and_tokenizer(model_name)
run_name = "math-smola-grpo_" + model_name.split("/")[-1].lower()

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