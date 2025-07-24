from datasets import concatenate_datasets
from smolagents.default_tools import PythonInterpreterTool  # type: ignore
from smolagents.tools import Tool  # type: ignore

from verifiers.envs.smolagents_tool_env import SmolagentsToolEnv
from verifiers.utils.data_utils import load_example_dataset

DEFAULT_SMOLA_PROMPT_TEMPLATE = """You are an intelligent assistant designed to solve problems that require careful reasoning.

When tackling a task, you should:
1. Break the problem down into steps
2. Reason carefully about how to solve it
3. Use available tools to help you solve the problem
4. Provide a clear final answer

Available tools:
{tool_descriptions}

Format your response using these XML tags:
<reasoning>
Think step-by-step about how to solve the task.
</reasoning>

<tool>
{{
  "name": "tool_name",
  "args": {{
    "arg1": "value1",
    "arg2": "value2"
  }}
}}
</tool>

<answer>
Your final answer or response to the user's request.
</answer>

First use the <reasoning> tag to think through the problem. When you need to use a tool, use the <tool> tag with the appropriate JSON format. When you're ready to provide the final answer, use the <answer> tag.
"""

MATH_SMOLA_PROMPT_TEMPLATE = """You are an intelligent math assistant designed to solve math problems that require careful reasoning.

When solving a math problem, you should:
1. Break the problem down into steps
2. Reason carefully through each step
3. Use the calculator tool to help with calculations
4. Provide a clear final answer in simplified form

Available tools:
{tool_descriptions}

Format your response using these XML tags:
<reasoning>
Think step-by-step about how to solve the math problem, explaining the approach clearly.
</reasoning>

<tool>
{{
  "name": "calculator", 
  "args": {{
    "expression": "math expression to calculate"
  }}
}}
</tool>

<answer>
Your final answer to the math problem, in simplified form.
</answer>

First use the <reasoning> tag to think through the problem. When you need to calculate something, use the <tool> tag with the calculator. When you're ready to provide the final answer, use the <answer> tag.
"""


"""
Few shot examples for SmolaAgents tool-based assistants.
"""

CALCULATOR_SMOLA_FEW_SHOTS = [
    {
        "role": "user",
        "content": "Janet has 3 times as many marbles as Tom. If Tom has 12 marbles, how many marbles do they have together?",
    },
    {
        "role": "assistant",
        "content": """<reasoning>
Let's solve this step by step:
1. First, calculate how many marbles Janet has (3 times Tom's)
</reasoning>

<tool>
{
  "name": "calculator", 
  "args": {
    "expression": "3 * 12"
  }
}
</tool>

<result>
36
</result>

<reasoning>
2. Now we know:
   - Tom has 12 marbles
   - Janet has 36 marbles
   Let's add them together
</reasoning>

<tool>
{
  "name": "calculator", 
  "args": {
    "expression": "36 + 12"
  }
}
</tool>

<result>
48
</result>

<reasoning>
Together they have 48 marbles:
- Tom: 12 marbles
- Janet: 36 marbles (3 times Tom's)
- Total: 48 marbles
</reasoning>

<answer>
48 marbles
</answer>""",
    },
    {
        "role": "user",
        "content": "Samantha is baking cookies. Each batch requires 2.5 cups of flour. If she has 10 cups of flour, how many complete batches can she make?",
    },
    {
        "role": "assistant",
        "content": """<reasoning>
To find how many complete batches Samantha can make, I need to divide the total amount of flour by the amount needed per batch.
</reasoning>

<tool>
{
  "name": "calculator", 
  "args": {
    "expression": "10 / 2.5"
  }
}
</tool>

<result>
4.0
</result>

<reasoning>
Samantha has 10 cups of flour and each batch requires 2.5 cups of flour.
10 ÷ 2.5 = 4
So Samantha can make 4 complete batches of cookies with her 10 cups of flour.
</reasoning>

<answer>
4 batches
</answer>""",
    },
]

SEARCH_SMOLA_FEW_SHOTS = [
    {
        "role": "user",
        "content": "Question: Which of the following is an example of mechanical weathering?\\n\\nChoices:\\nA. A rock turns red when iron minerals are exposed to oxygen.\\nB. A rock breaks apart after water freezes and expands in a crack.\\nC. A rock dissolves as acidic rainwater flows over its surface.\\nD. A rock changes composition when exposed to extreme heat.",
    },
    {
        "role": "assistant",
        "content": """<reasoning>
I need to understand what mechanical weathering is and which of these options matches that phenomenon. Let me search for information about mechanical weathering.
</reasoning>

<tool>
{
  "name": "web_search",
  "args": {
    "query": "what is mechanical weathering examples"
  }
}
</tool>

<result>
• Mechanical Weathering - Definition, Processes and Types
  Mechanical weathering, also known as physical weathering, is the process by which rocks are broken down into smaller pieces by external forces. Examples include frost wedging (when water freezes and expands in cracks), thermal expansion/contraction, and pressure release.

• Mechanical and Chemical Weathering - Earth Science
  Mechanical weathering breaks rocks into smaller pieces without changing their composition. Ice wedging occurs when water gets into cracks, freezes, and expands, breaking the rock apart. Other forms include plant root growth, animal activity, and abrasion.

• Mechanical Weathering: Definition, Types, Examples - Geology Science
  Types of mechanical weathering include frost wedging, salt crystallization, thermal expansion, plant activity, and animal activity. The most common mechanical weathering process in cold climates is frost wedging, where water enters rock cracks, freezes, and expands.
</result>

<reasoning>
Based on the search results, mechanical weathering is the process by which rocks are broken down into smaller pieces by external forces, without changing their chemical composition. Some examples include:

1. Frost wedging - when water gets into cracks, freezes, and expands, breaking the rock apart
2. Thermal expansion/contraction
3. Plant root growth
4. Animal activity
5. Abrasion

Looking at the answer choices:
A. A rock turns red when iron minerals are exposed to oxygen - This is chemical weathering, as it involves a chemical change (oxidation).
B. A rock breaks apart after water freezes and expands in a crack - This is frost wedging, a form of mechanical weathering.
C. A rock dissolves as acidic rainwater flows over its surface - This is chemical weathering (dissolution).
D. A rock changes composition when exposed to extreme heat - This is a chemical/thermal change, not mechanical weathering.

The answer is B, as it's the only option that describes mechanical weathering.
</reasoning>

<answer>
B
</answer>""",
    },
]

TOOL_SMOLA_FEW_SHOTS = [
    {"role": "user", "content": "What is the current working directory?"},
    {
        "role": "assistant",
        "content": """<reasoning>
Let's use the pwd command to find out the current working directory.
</reasoning>

<tool>
{
  "name": "python_interpreter",
  "args": {
    "code": "import os; print(os.getcwd())"
  }
}
</tool>

<result>
Stdout:
/Users/user/project
Output: None
</result>

<reasoning>
The current working directory is /Users/user/project.
</reasoning>

<answer>
/Users/user/project
</answer>""",
    },
    {
        "role": "user",
        "content": "How many Python files are in the current directory and its subdirectories?",
    },
    {
        "role": "assistant",
        "content": """<reasoning>
Let's use Python to count the number of Python files in the current directory and its subdirectories.
</reasoning>

<tool>
{
  "name": "python_interpreter",
  "args": {
    "code": "import os; count = 0; \\nfor root, dirs, files in os.walk('.'): \\n    count += sum(1 for file in files if file.endswith('.py')); \\nprint(count)"
  }
}
</tool>

<result>
Stdout:
42
Output: None
</result>

<reasoning>
There are 42 Python files in the current directory and its subdirectories.
</reasoning>

<answer>
42
</answer>""",
    },
]


class CalculatorTool(Tool):
    """A calculator tool for evaluating mathematical expressions."""

    name = "calculator"
    description = "Evaluates a single line of Python math expression. No imports or variables allowed."
    inputs = {
        "expression": {
            "type": "string",
            "description": "A mathematical expression using only numbers and basic operators (+,-,*,/,**,())",
        }
    }
    output_type = "string"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.allowed = set("0123456789+-*/.() ")
        self.is_initialized = True

    def forward(self, expression: str) -> str:  # type: ignore
        """Evaluates a single line of Python math expression. No imports or variables allowed.

        Args:
            expression: A mathematical expression using only numbers and basic operators (+,-,*,/,**,())

        Returns:
            The result of the calculation or an error message

        Examples:
            "2 + 2" -> "4"
            "3 * (17 + 4)" -> "63"
            "100 / 5" -> "20.0"
        """
        if not all(c in self.allowed for c in expression):
            return "Error: Invalid characters in expression"

        try:
            # Safely evaluate the expression with no access to builtins
            result = eval(expression, {"__builtins__": {}}, {})
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"


def load_environment(use_few_shot: bool = False, **kwargs):
    dataset = load_example_dataset("math", "train", n=6000)

    eval_aime24 = load_example_dataset("aime2024", n=30)
    eval_aime25 = load_example_dataset("aime2025", n=30)
    eval_dataset = concatenate_datasets([eval_aime24, eval_aime25]).shuffle(seed=0)

    # Use SmolaAgents' PythonInterpreterTool as a replacement for the python tool
    python_tool = PythonInterpreterTool(authorized_imports=["math", "sympy", "numpy"])
    # Add our custom calculator tool
    calculator_tool = CalculatorTool()

    vf_env = SmolagentsToolEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        system_prompt=MATH_SMOLA_PROMPT_TEMPLATE,
        few_shot=CALCULATOR_SMOLA_FEW_SHOTS if use_few_shot else [],
        tools=[python_tool, calculator_tool],
        max_steps=5,
        **kwargs,
    )

    return vf_env
