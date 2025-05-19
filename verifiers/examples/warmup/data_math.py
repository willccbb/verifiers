from openai import OpenAI

import verifiers as vf
from verifiers.tools import python
from verifiers.utils import preprocess_dataset

"""
Evaluating multi-turn reasoning before/after training.

CUDA_VISIBLE_DEVICES=0,1 vllm serve 'Qwen/Qwen2.5-7B-Instruct' --tensor_parallel_size 2 --max_model_len 8192 --dtype bfloat16 \
    --gpu_memory_utilization 0.9 --enable_prefix_caching \
    --host 0.0.0.0 --port 8001

uv run verifiers/examples/math_eval.py
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

You will then see the tool's output inside <result> tags. \
You may call tools multiple times if needed. \
Tool state does not persist between calls. \
Always use tools to solve problems whenever possible.

The <answer>...</answer> tags should contain only your final answer.

Example:
<answer>
\\frac{{1}}{{2}}
</answer>
"""

dataset = preprocess_dataset("openrs")
vf_env = vf.ToolEnv(
    eval_dataset=dataset,
    system_prompt=TOOL_PROMPT,
    llm_fields=["think", ("tool", "answer")],
    env_fields=["result"],
    few_shot=[],
    tools=[python],
    max_steps=3
)

# collect V3 rollouts from API
import os
from openai import OpenAI
base_url = "https://api.deepseek.com"
api_key = os.getenv("DEEPSEEK_API_KEY")
client = OpenAI(base_url=base_url, api_key=api_key)

# columns = ['prompt', 'completion', 'answer', 'reward']
# use deepseek-chat for multiturn rollouts (V3-0324)
dataset_dsv3 = vf_env.make_api_dataset(client, model="deepseek-chat", num_samples=10) 
# filter to top half of rows by rewards
dataset_dsv3 = dataset_dsv3.sort("reward", reverse=True).select(range(len(dataset_dsv3) // 2))
# # save to hub
dataset_dsv3.push_to_hub("V3-openrs-python-test")