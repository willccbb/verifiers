import verifiers as vf
from verifiers.tools import python
from verifiers.utils import load_example_dataset

"""
Multi-GPU training (single node, 4 training + 4 inference)

CUDA_VISIBLE_DEVICES=0,1,2,3 python verifiers/inference/vllm_serve.py --model 'Qwen/Qwen2.5-7B-Instruct' \
    --tensor_parallel_size 4 --max_model_len 8192 --dtype bfloat16 \
    --gpu_memory_utilization 0.9 --enable_prefix_caching True \
    --host 0.0.0.0 --port 8000

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config-file configs/zero3.yaml verifiers/examples/math_train.py
"""

TOOL_PROMPT = """
Think step-by-step inside <think>...</think> tags in each message, then either call a tool inside <tool>...</tool> tags, or give your final answer inside <answer>...</answer> tags.

You have access to the following tools to help solve problems:

{tool_descriptions}

Tools can be called by writing a JSON command inside <tool> tags with:
- "name": the name of the tool to use
- "args": the arguments for the tool

Example usage:
<tool>
{{"name": "python", "args": {{"code": "import sympy\nx = sympy.symbols('x')\nprint(sympy.solve(x**2 - 4, x))"}}}}
</tool>

After concluding your message with a tool call,
you will then see the tool's output inside <result> tags as a new message. \
You may call tools multiple times if needed. \
Tool state does not persist between calls. \
Always use tools to solve problems whenever possible, rather than using your own knowledge.

The <answer>...</answer> tags should contain only your final answer as a numeric expression.

Example:
<think>
Let's submit the answer.
</think>
<answer>
\\frac{{1}}{{2}}
</answer>
"""

dataset = load_example_dataset("math", split="train")

vf_env = vf.ToolEnv(
    dataset=dataset,
    system_prompt=TOOL_PROMPT,
    few_shot=[],
    tools=[python],
    max_steps=3
)
print(vf_env.system_prompt)

model_name = "willcb/Qwen2.5-7B-Math-Python-SFT"
model, tokenizer = vf.get_model_and_tokenizer(model_name)
run_name = "math-grpo_" + model_name.split("/")[-1].lower()

training_args=vf.grpo_defaults(run_name=run_name)
training_args.num_iterations=2
training_args.per_device_train_batch_size=8
training_args.num_generations=8
training_args.gradient_accumulation_steps=2

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=training_args,
)
trainer.train() 