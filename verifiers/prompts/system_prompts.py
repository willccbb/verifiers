SIMPLE_PROMPT = """
Respond in the following format, using careful step-by-step reasoning.

<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

CODE_PROMPT = """\
Given a math problem, use step-by-step reasoning and code execution to solve the problem. 

For each step:
1. Think through your reasoning inside <reasoning> tags
2. Write Python scripts inside <code> tags to work out calculations
   - Functions and variables do not persist across <code> calls and should be redefined each time
   - Scripts should be written in Python 3.10+ syntax, and should run in under 10 seconds
   - Any desired outputs should be printed using print() statements
   - You may import numpy, scipy, and sympy libraries for your calculations
3. You will see the output from print() statements in your code in <output> tags
4. Continue until you can give the final answer inside <answer> tags
"""

DEFAULT_TOOL_PROMPT_TEMPLATE = """\
You have access to the following tools to help solve problems:

{tool_descriptions}

For each step:
1. Think through your reasoning inside <reasoning> tags
2. If needed, use a tool by writing a JSON command inside <tool> tags with:
   - "name": the tool to use
   - "args": the arguments for the tool
3. You will see the tool's output inside <result> tags
4. Continue until you can give the final answer inside <answer> tags

Tools expect specific JSON input formats. Follow the examples carefully.
Do not make up tools or arguments that aren't listed.
"""

"""
Templates for SmolaAgents-style tool prompts.
"""

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