import subprocess
from typing import List, Dict
import re

import verifiers as vf
from datasets import Dataset


def python_sandbox(code: str) -> str:
    """Execute Python code in a subprocess with resource limits"""
    try:
        # Wrap code to handle input/output properly
        wrapped_code = f"""
import sys
{code}
"""
        
        result = subprocess.run(
            ["python3", "-c", wrapped_code],
            capture_output=True,
            text=True,
            timeout=10,
        )
        
        if result.returncode != 0:
            return f"Error: {result.stderr}"
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out after 10 seconds"
    except Exception as e:
        return f"Error: {str(e)}"


def load_environment(
    num_examples: int = 5,
    **kwargs,
):
    # Create simple coding problems for testing
    examples = [
        {
            'prompt': [{'role': 'user', 'content': 
                'Write a function called `add_two` that takes two integers and returns their sum.\n\n'
                'Example:\n'
                'Input: 1, 2\n'
                'Output: 3'}],
            'info': {
                'test_cases': [
                    {'input': '1, 2', 'expected': '3'},
                    {'input': '5, 7', 'expected': '12'},
                    {'input': '0, 0', 'expected': '0'},
                    {'input': '-1, 1', 'expected': '0'},
                ],
                'solution': 'def add_two(a, b):\n    return a + b',
            }
        },
        {
            'prompt': [{'role': 'user', 'content': 
                'Write a function called `is_even` that takes an integer and returns True if it is even, False otherwise.\n\n'
                'Example:\n'
                'Input: 4\n'
                'Output: True'}],
            'info': {
                'test_cases': [
                    {'input': '4', 'expected': 'True'},
                    {'input': '3', 'expected': 'False'},
                    {'input': '0', 'expected': 'True'},
                    {'input': '-2', 'expected': 'True'},
                ],
                'solution': 'def is_even(n):\n    return n % 2 == 0',
            }
        },
        {
            'prompt': [{'role': 'user', 'content': 
                'Write a function called `reverse_string` that takes a string and returns it reversed.\n\n'
                'Example:\n'
                'Input: "hello"\n'
                'Output: "olleh"'}],
            'info': {
                'test_cases': [
                    {'input': '"hello"', 'expected': 'olleh'},
                    {'input': '"world"', 'expected': 'dlrow'},
                    {'input': '"a"', 'expected': 'a'},
                    {'input': '""', 'expected': ''},
                ],
                'solution': 'def reverse_string(s):\n    return s[::-1]',
            }
        },
        {
            'prompt': [{'role': 'user', 'content': 
                'Write a function called `find_max` that takes a list of integers and returns the maximum value.\n\n'
                'Example:\n'
                'Input: [3, 1, 4, 1, 5]\n'
                'Output: 5'}],
            'info': {
                'test_cases': [
                    {'input': '[3, 1, 4, 1, 5]', 'expected': '5'},
                    {'input': '[10, 20, 30]', 'expected': '30'},
                    {'input': '[-5, -1, -10]', 'expected': '-1'},
                    {'input': '[42]', 'expected': '42'},
                ],
                'solution': 'def find_max(lst):\n    return max(lst)',
            }
        },
        {
            'prompt': [{'role': 'user', 'content': 
                'Write a function called `count_vowels` that takes a string and returns the number of vowels (a, e, i, o, u) in it.\n\n'
                'Example:\n'
                'Input: "hello"\n'
                'Output: 2'}],
            'info': {
                'test_cases': [
                    {'input': '"hello"', 'expected': '2'},
                    {'input': '"world"', 'expected': '1'},
                    {'input': '"aeiou"', 'expected': '5'},
                    {'input': '"xyz"', 'expected': '0'},
                ],
                'solution': 'def count_vowels(s):\n    return sum(1 for c in s.lower() if c in "aeiou")',
            }
        },
    ]
    
    dataset = Dataset.from_list(examples[:num_examples])
    
    def extract_code(completion: List[Dict]) -> str:
        """Extract code from model completion"""
        if not completion:
            return ""
        
        last_message = completion[-1]['content']
        
        # Extract code between ```python and ```
        code_match = re.search(r'```python\n(.*?)```', last_message, re.DOTALL)
        if code_match:
            return code_match.group(1)
        
        # Extract code between ``` and ```
        code_match = re.search(r'```\n(.*?)```', last_message, re.DOTALL)
        if code_match:
            return code_match.group(1)
        
        # Look for function definition
        if 'def ' in last_message:
            # Extract from first def to the end
            def_index = last_message.find('def ')
            return last_message[def_index:]
        
        return last_message
    
    parser = vf.Parser(extract_fn=extract_code)
    
    def correctness_score(parser, completion, info) -> float:
        """Evaluate code correctness using test cases"""
        code = parser.parse_answer(completion)
        
        if not code:
            return 0.0
        
        # Extract function name from the problem
        func_match = re.search(r'function called `(\w+)`', completion[0]['content'])
        if not func_match:
            return 0.0
        
        func_name = func_match.group(1)
        
        # Run tests
        passed = 0
        total = len(info['test_cases'])
        
        for test in info['test_cases']:
            test_code = f"""
{code}

# Run test
result = {func_name}({test['input']})
print(result)
"""
            output = python_sandbox(test_code)
            
            if output.strip() == test['expected'].strip():
                passed += 1
        
        return passed / total if total > 0 else 0.0
    
    def num_lines(parser, completion, info) -> float:
        """Count lines of code (for analysis)"""
        code = parser.parse_answer(completion)
        return float(len(code.strip().split('\n')) if code else 0)
    
    rubric = vf.Rubric(
        funcs=[correctness_score, num_lines],
        weights=[1.0, 0.0],
        parser=parser,
    )
    
    vf_env = vf.SingleTurnEnv(
        dataset=dataset,
        rubric=rubric,
        **kwargs,
    )
    
    return vf_env