from datasets import concatenate_datasets

import verifiers as vf
from verifiers.utils import load_example_dataset
from verifiers.prompts.system_prompts import MATH_SMOLA_PROMPT_TEMPLATE
from verifiers.prompts.few_shots import CALCULATOR_SMOLA_FEW_SHOTS
from verifiers.envs.smola_tool_env import SmolaToolEnv

try:    
    from smolagents.default_tools import PythonInterpreterTool # type: ignore
    from verifiers.tools.smolagents import CalculatorTool
except ImportError:
    raise ImportError("Please install smolagents to use SmolAgents tools.")

"""
Multi-GPU training (single node, 4 training + 4 inference) using SmolaAgents tools

CUDA_VISIBLE_DEVICES=0,1,2,3 python verifiers/inference/vllm_server.py \
    --model 'Qwen/Qwen2.5-7B-Instruct' \
    --tensor-parallel-size 4 \
    --max-model-len 8192 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --enable-prefix-caching \
    --host 0.0.0.0 \
    --port 8000

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config-file configs/zero3.yaml --num-processes 4 verifiers/examples/smola_math_tools.py
"""

dataset = load_example_dataset("math", "train", n=6000)

eval_aime24 = load_example_dataset("aime2024", n=30)
eval_aime25 = load_example_dataset("aime2025", n=30)
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

args = vf.grpo_defaults(run_name=run_name)
trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=args,
)
trainer.train()