"""
Few shot examples for SmolaAgents tool-based assistants.
"""

CALCULATOR_SMOLA_FEW_SHOTS = [
    {
        'role': 'user',
        'content': 'Janet has 3 times as many marbles as Tom. If Tom has 12 marbles, how many marbles do they have together?'
    },
    {
        'role': 'assistant',
        'content': """<reasoning>
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
</answer>"""
    },
    {
        'role': 'user',
        'content': 'Samantha is baking cookies. Each batch requires 2.5 cups of flour. If she has 10 cups of flour, how many complete batches can she make?'
    },
    {
        'role': 'assistant',
        'content': """<reasoning>
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
</answer>"""
    }
]

SEARCH_SMOLA_FEW_SHOTS = [
    {
        'role': 'user', 
        'content': 'Question: Which of the following is an example of mechanical weathering?\\n\\nChoices:\\nA. A rock turns red when iron minerals are exposed to oxygen.\\nB. A rock breaks apart after water freezes and expands in a crack.\\nC. A rock dissolves as acidic rainwater flows over its surface.\\nD. A rock changes composition when exposed to extreme heat.'
    },
    {
        'role': 'assistant', 
        'content': """<reasoning>
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
</answer>"""
    }
]

TOOL_SMOLA_FEW_SHOTS = [
    {
        'role': 'user',
        'content': 'What is the current working directory?'
    },
    {
        'role': 'assistant',
        'content': """<reasoning>
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
</answer>"""
    },
    {
        'role': 'user',
        'content': 'How many Python files are in the current directory and its subdirectories?'
    },
    {
        'role': 'assistant',
        'content': """<reasoning>
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
</answer>"""
    }
]