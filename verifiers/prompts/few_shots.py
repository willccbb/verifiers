"""
Few-shot examples for verifiers.

NOTE: We recommend leveraging SFT warmup rather than few-shot examples.
"""

from verifiers.parsers import XMLParser

math_parser = XMLParser(fields=["reasoning", "answer"])
MATH_FEW_SHOT = [
    [
        {'role': 'user', 'content': 'What is the largest single-digit prime number?'},
        {'role': 'assistant', 'content': math_parser.format(
            reasoning='The largest single-digit prime number is 7.',
            answer='7'
        )}
    ]
]

DOUBLECHECK_FEW_SHOT = [
    [
        {'role': 'user', 'content': 'What is the largest single-digit prime number?'},
        {'role': 'assistant', 'content': math_parser.format(
            reasoning='The largest single-digit prime number is 7.',
            answer='7'
        )},
        {'role': 'user', 'content': 'Are you sure?'},
        {'role': 'assistant', 'content': math_parser.format(
            reasoning='The only larger single-digit numbers are 8 and 9, which are not prime. So yes, the answer is 7.',
            answer='7'
        )}
    ]
]

code_parser = XMLParser(fields=["reasoning", ("code", "answer")])
output_parser = XMLParser(fields=["output"])
CODE_FEW_SHOT = [
    [
        {
            'role': 'user',
            'content': 'What is sum of the first 100 positive even integers?'
        },
        {
            'role': 'assistant',
            'content': code_parser.format(
                reasoning='Let\'s compute the sum of the first 100 positive even integers.',
                code='print(sum(range(2, 102, 2)))'
            )
        },
        {
            'role': 'user', 
            'content': output_parser.format(output='2550')
        },
        {
            'role': 'assistant',
            'content': code_parser.format(reasoning='The answer is 2550.', answer='2550')
        },
        {
            'role': 'user',
            'content': 'What is the sum of the first 100 natural numbers, minus the largest prime number less than 100?'
        },
        {
            'role': 'assistant',
            'content': code_parser.format(
                reasoning='The sum of the first n natural numbers is given by the formula n(n+1)/2.',
                code='print(100*101/2)'
            )
        },
        {
            'role': 'user',
            'content': output_parser.format(output='5050')
        },
        {
            'role': 'assistant',
            'content': code_parser.format(
                reasoning='The sum of the first 100 natural numbers is 5050. Now we need to subtract the largest prime number less than 100.',
                code='print(5050 - 97)'
            )
        },
        {
            'role': 'user',
            'content': output_parser.format(output='4953')
        },
        {
            'role': 'assistant',
            'content': code_parser.format(
                reasoning='The largest prime number less than 100 is 97. Subtracting this from 5050 gives 4953.',
                answer='4953'
            )
        }
    ]
]

tool_parser = XMLParser(fields=["reasoning", ("tool", "answer")])
result_parser = XMLParser(fields=["result"])

TOOL_FEW_SHOT = [
    [
        {
            'role': 'user',
            'content': 'What is the current working directory?'
        },
        {
            'role': 'assistant',
            'content': tool_parser.format(
                reasoning='Let\'s use the pwd command to find out the current working directory.',
                tool='pwd'
            )
        },
        {
            'role': 'user',
            'content': result_parser.format(result='/Users/user/project')
        },
        {
            'role': 'assistant',
            'content': tool_parser.format(
                reasoning='The current working directory is /Users/user/project.',
                answer='/Users/user/project'
            )
        },
        {
            'role': 'user',
            'content': 'How many Python files are in the current directory and its subdirectories?'
        },
        {
            'role': 'assistant',
            'content': tool_parser.format(
                reasoning='Let\'s use the find command to count Python files.',
                tool='find . -name "*.py" | wc -l'
            )
        },
        {
            'role': 'user',
            'content': result_parser.format(result='42')
        },
        {
            'role': 'assistant',
            'content': tool_parser.format(
                reasoning='There are 42 Python files in the current directory and its subdirectories.',
                answer='42'
            )
        }
    ]
]

COMMONSENSE_FEW_SHOT = [
    [
        {
            'role': 'user',
            'content': 'Which would be louder: a mouse or an elephant?'
        },
        {
            'role': 'assistant',
            'content': tool_parser.format(
                reasoning='Let\'s compare the volume levels of a mouse and an elephant.',
                tool='compare mouse elephant volume'
            )
        },
        {
            'role': 'user',
            'content': result_parser.format(result='''{
  "difference": -4,
  "mouse_volume": 1,
  "elephant_volume": 5
}''')
        },
        {
            'role': 'assistant',
            'content': tool_parser.format(
                reasoning='Based on the comparison, an elephant has a volume level of 5 while a mouse has a volume level of 1 (on a scale of 1-5). The difference of -4 indicates the elephant is much louder.',
                answer='An elephant would be louder than a mouse.'
            )
        },
        {
            'role': 'user',
            'content': 'What properties does a car have?'
        },
        {
            'role': 'assistant',
            'content': tool_parser.format(
                reasoning='Let\'s look up the properties of a car.',
                tool='get_related car'
            )
        },
        {
            'role': 'user',
            'content': result_parser.format(result='''{
  "properties": ["metallic", "fast", "loud", "heavy"],
  "category": "vehicle"
}''')
        },
        {
            'role': 'assistant',
            'content': tool_parser.format(
                reasoning='A car has several key properties: it is metallic, fast, loud, and heavy. It belongs to the category of vehicles.',
                answer='A car is a vehicle that is metallic, fast, loud, and heavy.'
            )
        }
    ]
]

SEARCH_FEW_SHOT = [
    [
        {'role': 'user', 'content': 'Question: Which of the following is an example of mechanical weathering?\n\nChoices:\nA. A rock turns red when iron minerals are exposed to oxygen.\nB. A rock breaks apart after water freezes and expands in a crack.\nC. A rock dissolves as acidic rainwater flows over its surface.\nD. A rock changes composition when exposed to extreme heat.'},
        {'role': 'assistant', 'content': tool_parser.format(
            reasoning="I need to understand what mechanical weathering is and which of these options matches that phenomenon. Let me search for information about mechanical weathering.",
            tool='{"name": "search", "args": {"query": "what is mechanical weathering examples", "num_results": 3}}'
        )},
        {'role': 'user', 'content': result_parser.format(
            result="• Mechanical Weathering - Definition, Processes and Types\n  Mechanical weathering, also known as physical weathering, is the process by which rocks are broken down into smaller pieces by external forces. Examples include frost wedging (when water freezes and expands in cracks), thermal expansion/contraction, and pressure release.\n\n• Mechanical and Chemical Weathering - Earth Science\n  Mechanical weathering breaks rocks into smaller pieces without changing their composition. Ice wedging occurs when water gets into cracks, freezes, and expands, breaking the rock apart. Other forms include plant root growth, animal activity, and abrasion.\n\n• Mechanical Weathering: Definition, Types, Examples - Geology Science\n  Types of mechanical weathering include frost wedging, salt crystallization, thermal expansion, plant activity, and animal activity. The most common mechanical weathering process in cold climates is frost wedging, where water enters rock cracks, freezes, and expands."
        )},
        {'role': 'assistant', 'content': tool_parser.format(
            reasoning="Based on the search results, mechanical weathering is the process by which rocks are broken down into smaller pieces by external forces, without changing their chemical composition. Some examples include:\n\n1. Frost wedging - when water gets into cracks, freezes, and expands, breaking the rock apart\n2. Thermal expansion/contraction\n3. Plant root growth\n4. Animal activity\n5. Abrasion\n\nLooking at the answer choices:\nA. A rock turns red when iron minerals are exposed to oxygen - This is chemical weathering, as it involves a chemical change (oxidation).\nB. A rock breaks apart after water freezes and expands in a crack - This is frost wedging, a form of mechanical weathering.\nC. A rock dissolves as acidic rainwater flows over its surface - This is chemical weathering (dissolution).\nD. A rock changes composition when exposed to extreme heat - This is a chemical/thermal change, not mechanical weathering.\n\nThe answer is B, as it's the only option that describes mechanical weathering.",
            answer="B"
        )}
    ]
]

CALCULATOR_FEW_SHOT = [
    [
        {
            'role': 'user',
            'content': 'Janet has 3 times as many marbles as Tom. If Tom has 12 marbles, how many marbles do they have together?'
        },
        {
            'role': 'assistant',
            'content': tool_parser.format(
                reasoning="Let's solve this step by step:\n1. First, calculate how many marbles Janet has (3 times Tom's)",
                tool='{"name": "calculator", "args": {"expression": "3 * 12"}}'
            )
        },
        {
            'role': 'user',
            'content': result_parser.format(result="36")
        },
        {
            'role': 'assistant',
            'content': tool_parser.format(
                reasoning="2. Now we know:\n   - Tom has 12 marbles\n   - Janet has 36 marbles\n   Let's add them together",
                tool='{"name": "calculator", "args": {"expression": "36 + 12"}}'
            )
        },
        {
            'role': 'user',
            'content': result_parser.format(result="48")
        },
        {
            'role': 'assistant',
            'content': tool_parser.format(
                reasoning="Together they have 48 marbles:\n- Tom: 12 marbles\n- Janet: 36 marbles (3 times Tom's)\n- Total: 48 marbles",
                answer="48 marbles"
            )
        }
    ],
    [
        {
            'role': 'user',
            'content': 'Samantha is baking cookies. Each batch requires 2.5 cups of flour. If she has 10 cups of flour, how many complete batches can she make?'
        },
        {
            'role': 'assistant',
            'content': tool_parser.format(
                reasoning="To find how many complete batches Samantha can make, I need to divide the total amount of flour by the amount needed per batch.",
                tool='{"name": "calculator", "args": {"expression": "10 / 2.5"}}'
            )
        },
        {
            'role': 'user',
            'content': result_parser.format(result="4.0")
        },
        {
            'role': 'assistant',
            'content': tool_parser.format(
                reasoning="Samantha has 10 cups of flour and each batch requires 2.5 cups of flour.\n10 ÷ 2.5 = 4\nSo Samantha can make 4 complete batches of cookies with her 10 cups of flour.",
                answer="4 batches"
            )
        }
    ]
]

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