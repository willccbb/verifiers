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

dataset = preprocess_dataset("math500")
vf_env = vf.ToolEnv(
    eval_dataset=dataset,
    system_prompt=TOOL_PROMPT,
    few_shot=[],
    tools=[python],
    max_steps=1
)
print(vf_env.system_prompt)

model_name = "Qwen/Qwen2.5-7B-Instruct"
base_url = "http://0.0.0.0:8001/v1"
client = OpenAI(base_url=base_url, api_key="EMPTY")
vf_env.eval_api(client, model_name, max_concurrent=20, sampling_args={"temperature": 0.6})
