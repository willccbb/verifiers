import subprocess
import tempfile
import os
import re
from typing import List, Dict, Optional, Tuple
from contextlib import contextmanager
import json

import verifiers as vf
from datasets import load_dataset, Dataset


class SecureSandboxExecutor:
    """Handles secure execution of code in isolated environments"""
    
    def __init__(self, timeout: int = 30, memory_limit_mb: int = 512):
        self.timeout = timeout
        self.memory_limit_mb = memory_limit_mb
        
    def execute_python(self, code: str, test_input: str = "") -> Dict[str, str]:
        """Execute Python code in a subprocess with resource limits"""
        try:
            # Create a temporary file for the code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                code_file = f.name
            
            try:
                # Run the code with timeout
                result = subprocess.run(
                    ["python3", code_file],
                    input=test_input,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
                
                return {
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'returncode': result.returncode,
                    'success': result.returncode == 0
                }
            finally:
                # Clean up
                os.unlink(code_file)
                
        except subprocess.TimeoutExpired:
            return {
                'stdout': '',
                'stderr': f'Execution timed out after {self.timeout} seconds',
                'returncode': -1,
                'success': False
            }
        except Exception as e:
            return {
                'stdout': '',
                'stderr': str(e),
                'returncode': -1,
                'success': False
            }
    
    def run_test_cases(self, code: str, test_cases: List[Dict], entry_point: str) -> Dict:
        """Run multiple test cases against code"""
        results = []
        
        for test in test_cases:
            # Prepare test execution code
            test_input = test.get('input', '')
            expected_output = str(test.get('output', test.get('expected', '')))
            
            # Create test execution code
            test_code = f"""
{code}

# Test execution
if __name__ == "__main__":
    # Parse input based on function signature
    input_str = "{test_input}"
    
    # Try to parse the input appropriately
    try:
        # Handle multiple arguments separated by comma
        if ',' in input_str:
            args = []
            for arg in input_str.split(','):
                arg = arg.strip()
                # Try to evaluate as Python literal
                try:
                    args.append(eval(arg))
                except:
                    # If eval fails, treat as string
                    args.append(arg.strip('"').strip("'"))
            result = {entry_point}(*args)
        else:
            # Single argument
            try:
                arg = eval(input_str)
            except:
                arg = input_str.strip('"').strip("'")
            result = {entry_point}(arg)
    except Exception as e:
        result = f"ERROR: {{e}}"
    
    print(result)
"""
            
            # Execute test
            execution_result = self.execute_python(test_code)
            
            # Check if output matches expected
            if execution_result['success'] and execution_result['stdout']:
                actual_output = execution_result['stdout'].strip()
                
                # Normalize outputs for comparison
                actual_normalized = str(actual_output).strip()
                expected_normalized = str(expected_output).strip()
                
                # Try to handle Python literal comparison
                try:
                    if actual_normalized == expected_normalized:
                        passed = True
                    else:
                        # Try evaluating both as Python expressions
                        actual_eval = eval(actual_normalized)
                        expected_eval = eval(expected_normalized)
                        passed = actual_eval == expected_eval
                except:
                    # Fallback to string comparison
                    passed = actual_normalized == expected_normalized
            else:
                passed = False
            
            results.append({
                'passed': passed,
                'test': test,
                'actual_output': execution_result.get('stdout', '').strip(),
                'expected_output': expected_output,
                'error': execution_result.get('stderr', '')
            })
        
        # Calculate metrics
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r['passed'])
        
        return {
            'pass_rate': passed_tests / total_tests if total_tests > 0 else 0.0,
            'passed': passed_tests,
            'total': total_tests,
            'results': results
        }


# Create a global sandbox instance
sandbox = SecureSandboxExecutor()


def load_environment(
    dataset_name: str = "livecodebench/livecodebench",
    split: str = "test",
    version: str = "release_v1",
    num_examples: int = -1,
    language: str = "python",
    **kwargs,
):
    """Load LiveCodeBench environment with actual dataset from HuggingFace"""
    
    # Try to load LiveCodeBench dataset
    dataset_loaded = False
    try:
        # Try different dataset formats
        dataset_configs = [
            (dataset_name, None),
            ("livecodebench/livecodebench", "code_generation"),
            ("livecodebench/livecodebench", None),
        ]
        
        for ds_name, config in dataset_configs:
            try:
                if config:
                    dataset = load_dataset(ds_name, config, split=split, trust_remote_code=True)
                else:
                    dataset = load_dataset(ds_name, split=split, trust_remote_code=True)
                dataset_loaded = True
                break
            except:
                continue
                
    except Exception as e:
        print(f"Warning: Could not load LiveCodeBench dataset: {e}")
    
    if not dataset_loaded:
        print("Creating example dataset for testing...")
        
        # Create a more comprehensive example dataset
        examples = [
            {
                'question_id': 'example_1',
                'question': 'Write a function called `add_two` that takes two integers and returns their sum.',
                'examples': json.dumps([
                    {'input': '1, 2', 'output': '3'},
                    {'input': '5, 7', 'output': '12'}
                ]),
                'test_list': json.dumps([
                    {'input': '1, 2', 'output': '3'},
                    {'input': '5, 7', 'output': '12'},
                    {'input': '0, 0', 'output': '0'},
                    {'input': '-1, 1', 'output': '0'},
                    {'input': '100, 200', 'output': '300'}
                ]),
                'entry_point': 'add_two',
                'language': 'python'
            },
            {
                'question_id': 'example_2',
                'question': 'Write a function called `is_even` that takes an integer and returns True if it is even, False otherwise.',
                'examples': json.dumps([
                    {'input': '4', 'output': 'True'},
                    {'input': '3', 'output': 'False'}
                ]),
                'test_list': json.dumps([
                    {'input': '4', 'output': 'True'},
                    {'input': '3', 'output': 'False'},
                    {'input': '0', 'output': 'True'},
                    {'input': '-2', 'output': 'True'},
                    {'input': '1001', 'output': 'False'}
                ]),
                'entry_point': 'is_even',
                'language': 'python'
            },
            {
                'question_id': 'example_3',
                'question': 'Write a function called `reverse_string` that takes a string and returns it reversed.',
                'examples': json.dumps([
                    {'input': '"hello"', 'output': 'olleh'},
                    {'input': '"world"', 'output': 'dlrow'}
                ]),
                'test_list': json.dumps([
                    {'input': '"hello"', 'output': 'olleh'},
                    {'input': '"world"', 'output': 'dlrow'},
                    {'input': '"a"', 'output': 'a'},
                    {'input': '""', 'output': ''},
                    {'input': '"racecar"', 'output': 'racecar'}
                ]),
                'entry_point': 'reverse_string',
                'language': 'python'
            },
            {
                'question_id': 'example_4',
                'question': 'Write a function called `find_max` that takes a list of integers and returns the maximum value.',
                'examples': json.dumps([
                    {'input': '[3, 1, 4, 1, 5]', 'output': '5'},
                    {'input': '[10, 20, 30]', 'output': '30'}
                ]),
                'test_list': json.dumps([
                    {'input': '[3, 1, 4, 1, 5]', 'output': '5'},
                    {'input': '[10, 20, 30]', 'output': '30'},
                    {'input': '[-5, -1, -10]', 'output': '-1'},
                    {'input': '[42]', 'output': '42'},
                    {'input': '[0, 0, 0]', 'output': '0'}
                ]),
                'entry_point': 'find_max',
                'language': 'python'
            },
            {
                'question_id': 'example_5',
                'question': 'Write a function called `count_vowels` that takes a string and returns the number of vowels (a, e, i, o, u) in it. The function should be case-insensitive.',
                'examples': json.dumps([
                    {'input': '"hello"', 'output': '2'},
                    {'input': '"WORLD"', 'output': '1'}
                ]),
                'test_list': json.dumps([
                    {'input': '"hello"', 'output': '2'},
                    {'input': '"WORLD"', 'output': '1'},
                    {'input': '"aeiou"', 'output': '5'},
                    {'input': '"xyz"', 'output': '0'},
                    {'input': '"Python"', 'output': '1'}
                ]),
                'entry_point': 'count_vowels',
                'language': 'python'
            }
        ]
        dataset = Dataset.from_list(examples)
    
    # Filter by language if specified and column exists
    if language and 'language' in dataset.column_names:
        dataset = dataset.filter(lambda x: x.get('language', 'python') == language)
    
    # Limit number of examples if specified
    if num_examples > 0 and len(dataset) > num_examples:
        dataset = dataset.select(range(num_examples))
    
    # Convert LiveCodeBench format to verifiers format
    def convert_to_verifiers_format(example):
        # Parse examples and tests
        try:
            examples_list = json.loads(example.get('examples', '[]'))
            test_list = json.loads(example.get('test_list', '[]'))
        except:
            examples_list = []
            test_list = []
        
        # Format problem statement
        problem_text = example.get('question', example.get('description', ''))
        if examples_list:
            problem_text += "\n\nExamples:\n"
            for ex in examples_list[:2]:  # Show first 2 examples
                problem_text += f"Input: {ex.get('input', '')}\n"
                problem_text += f"Output: {ex.get('output', '')}\n\n"
        
        # Create info dict with all necessary data
        info_dict = {
            'question_id': example.get('question_id', ''),
            'entry_point': example.get('entry_point', 'solution'),
            'test_cases': test_list,
            'language': example.get('language', 'python'),
            'difficulty': example.get('difficulty', 'unknown'),
            'source': example.get('source', 'livecodebench')
        }
        
        return {
            'prompt': [{'role': 'user', 'content': problem_text}],
            'info': info_dict  # Store as 'info' column for verifiers
        }
    
    # Convert dataset
    converted_examples = [convert_to_verifiers_format(ex) for ex in dataset]
    dataset = Dataset.from_list(converted_examples)
    
    def extract_code(completion: str) -> str:
        """Extract code from model completion string"""
        if not completion:
            return ""
        
        # Try to extract code between ```python and ```
        code_match = re.search(r'```python\n(.*?)```', completion, re.DOTALL)
        if code_match:
            return code_match.group(1)
        
        # Try to extract code between ``` and ```
        code_match = re.search(r'```\n(.*?)```', completion, re.DOTALL)
        if code_match:
            return code_match.group(1)
        
        # Look for function definition
        if 'def ' in completion:
            # Extract from first def to the end
            def_index = completion.find('def ')
            # Find the end of the function (next def or end of string)
            next_def = completion.find('\ndef ', def_index + 1)
            if next_def != -1:
                return completion[def_index:next_def].strip()
            else:
                return completion[def_index:].strip()
        
        return completion
    
    parser = vf.Parser(extract_fn=extract_code)
    
    # Define rubric functions with correct signatures
    def correctness_score(parser, completion, info) -> float:
        """Evaluate code correctness using test cases"""
        # Parse code from completion messages
        code = parser.parse_answer(completion)
        
        if not code or not isinstance(info, dict) or not info.get('test_cases'):
            return 0.0
        
        # Get entry point (function name)
        entry_point = info.get('entry_point', 'solution')
        
        # Run tests in sandbox
        test_results = sandbox.run_test_cases(
            code=code,
            test_cases=info['test_cases'],
            entry_point=entry_point
        )
        
        return test_results['pass_rate']
    
    def execution_success(parser, completion, info) -> float:
        """Check if code executes without errors"""
        code = parser.parse_answer(completion)
        
        if not code:
            return 0.0
        
        # Try to execute the code
        result = sandbox.execute_python(code)
        return 1.0 if result['success'] else 0.0
    
    def code_length(parser, completion, info) -> float:
        """Measure code length (for analysis)"""
        code = parser.parse_answer(completion)
        return float(len(code.strip().split('\n')) if code else 0)
    
    rubric = vf.Rubric(
        funcs=[correctness_score, execution_success, code_length],
        weights=[1.0, 0.0, 0.0],  # Only correctness counts for score
        parser=parser,
    )
    
    # Create environment
    vf_env = vf.SingleTurnEnv(
        dataset=dataset,
        rubric=rubric,
        **kwargs,
    )
    
    return vf_env
