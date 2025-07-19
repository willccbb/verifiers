import verifiers as vf
from verifiers.tools import python
from verifiers.utils.data_utils import load_example_dataset

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


def load_environment(
    dataset_name: str = "math",
    dataset_split: str = "train",
    num_train_examples: int = -1,
    **kwargs,
):
    dataset = load_example_dataset(dataset_name, dataset_split, n=num_train_examples)

    vf_env = vf.ToolEnv(
        dataset=dataset,
        system_prompt=TOOL_PROMPT,
        few_shot=[],
        tools=[python],
        max_steps=3,
        **kwargs,
    )

    return vf_env
