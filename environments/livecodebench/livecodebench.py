"""
LiveCodeBench Real Dataset Implementation for Verifiers

This implementation uses the ACTUAL LiveCodeBench dataset from HuggingFace
with proper Docker-based sandboxing for secure code execution.
"""

import os
import re
import json
import time
import subprocess
import tempfile
import resource
import urllib.request
import urllib.error
import base64
import zlib
import gzip
import pickle
from typing import Dict, List, Any, Optional, Tuple

import verifiers as vf
from datasets import Dataset, load_dataset
import docker


class SecureSandboxExecutor:
    """Secure Docker-based sandbox for code execution"""
    
    def __init__(self):
        """Initialize Docker client"""
        try:
            # Try to connect to Docker
            self.docker_client = docker.from_env()
            self.docker_available = True
            print("Docker sandbox initialized successfully")
            
            # Test if we can actually use Docker
            try:
                self.docker_client.ping()
            except docker.errors.APIError:
                # Try with sudo
                print("Regular Docker access failed, trying with sudo...")
                # For environments where Docker requires sudo, we'll need to handle this differently
                # In production, the user running the evaluation should be in the docker group
                raise RuntimeError(
                    "Docker requires elevated permissions. Please ensure the user is in the docker group:\n"
                    "  sudo usermod -aG docker $USER\n"
                    "  Then log out and back in."
                )
            
            # Pull Python image if not available
            try:
                self.docker_client.images.get("python:3.11-slim")
            except docker.errors.ImageNotFound:
                print("Pulling Python Docker image...")
                self.docker_client.images.pull("python:3.11-slim")
                print("Python Docker image ready")
                
        except Exception as e:
            raise RuntimeError(f"Docker is required but not available: {e}\n"
                             "Please ensure Docker is installed and running:\n"
                             "  - Install Docker: https://docs.docker.com/get-docker/\n"
                             "  - Start Docker daemon: sudo systemctl start docker\n"
                             "  - Add user to docker group: sudo usermod -aG docker $USER")
    
    def execute(self, code: str, test_input: str = "", timeout: int = 10) -> Tuple[str, str, int]:
        """Execute code in Docker sandbox"""
        
        # Create a temporary directory for the code
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write code to file with proper permissions
            code_path = os.path.join(tmpdir, "solution.py")
            with open(code_path, 'w') as f:
                f.write(code)
            os.chmod(code_path, 0o644)
            
            # Write input to file with proper permissions
            input_path = os.path.join(tmpdir, "input.txt")
            with open(input_path, 'w') as f:
                f.write(test_input)
            os.chmod(input_path, 0o644)
            
            stdout, stderr = "", ""
            exit_code = 0
            
            try:
                # Run container
                # Use a Python wrapper to handle stdin properly
                wrapper_code = f"""
import sys
sys.stdin = open('/code/input.txt', 'r')
exec(open('/code/solution.py').read())
"""
                wrapper_path = os.path.join(tmpdir, "wrapper.py")
                with open(wrapper_path, 'w') as f:
                    f.write(wrapper_code)
                os.chmod(wrapper_path, 0o644)
                
                container = self.docker_client.containers.run(
                    image='python:3.11-slim',
                    command=['python', '/code/wrapper.py'],
                    volumes={
                        tmpdir: {'bind': '/code', 'mode': 'rw'}  # Changed to rw
                    },
                    working_dir='/code',
                    detach=True,
                    mem_limit='512m',
                    memswap_limit='512m', 
                    cpu_quota=50000,  # 0.5 CPU
                    cpu_period=100000,
                    pids_limit=50,
                    network_disabled=True,
                    tmpfs={'/tmp': 'size=64M,mode=1777'},
                    security_opt=['no-new-privileges'],
                    cap_drop=['ALL'],
                    user='1000:1000'  # Use a non-root user
                )
                
                # Wait for completion with timeout
                try:
                    result = container.wait(timeout=timeout)
                    exit_code = result.get('StatusCode', 0)
                    
                    # Get logs
                    logs = container.logs(stdout=True, stderr=True)
                    stdout = logs.decode('utf-8', errors='replace')
                    
                except docker.errors.APIError as e:
                    stderr = "Execution timed out"
                    exit_code = 124
                finally:
                    # Clean up
                    try:
                        container.remove(force=True)
                    except:
                        pass
                
            except docker.errors.ContainerError as e:
                # Container exited with non-zero code
                stdout = e.output.decode('utf-8', errors='replace') if e.output else ""
                stderr = str(e)
                exit_code = e.exit_status
            except docker.errors.ImageNotFound:
                stderr = "Docker image not found"
                exit_code = 1
            except Exception as e:
                stderr = f"Docker execution error: {str(e)}"
                exit_code = 1
            
            return stdout, stderr, exit_code


def load_livecodebench_dataset(
    version_tag: str = "release_v5",
    split: str = "test", 
    num_examples: int = -1
) -> Dataset:
    """Load the ACTUAL LiveCodeBench dataset from HuggingFace"""
    
    dataset_name = "livecodebench/code_generation_lite"
    
    print(f"Loading LiveCodeBench dataset: {dataset_name}")
    print(f"Version: {version_tag}")
    
    # Map version tags to JSONL files
    version_to_file = {
        "release_v1": "test.jsonl",
        "release_v2": "test2.jsonl", 
        "release_v3": "test3.jsonl",
        "release_v4": "test4.jsonl",
        "release_v5": "test5.jsonl",
        "release_v6": "test6.jsonl"
    }
    
    # Get the appropriate file
    jsonl_file = version_to_file.get(version_tag, "test5.jsonl")
    
    print(f"Downloading {jsonl_file} from HuggingFace...")
    
    try:
        import urllib.request
        import json
        
        # Download JSONL file from HuggingFace
        url = f"https://huggingface.co/datasets/{dataset_name}/resolve/main/{jsonl_file}"
        
        examples = []
        with urllib.request.urlopen(url) as response:
            # Read line by line to handle large files
            for line_num, line in enumerate(response):
                if line.strip():
                    try:
                        example = json.loads(line)
                        examples.append(example)
                        
                        # Print progress every 100 examples
                        if (line_num + 1) % 100 == 0:
                            print(f"Loaded {line_num + 1} examples...")
                            
                        # Stop if we've reached the desired number
                        if num_examples > 0 and len(examples) >= num_examples:
                            break
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line {line_num}: {e}")
                        continue
        
        print(f"Successfully loaded {len(examples)} problems from LiveCodeBench")
        
        # Convert to HuggingFace Dataset
        dataset = Dataset.from_list(examples)
        
        return dataset
        
    except Exception as e:
        print(f"Error loading LiveCodeBench dataset: {e}")
        
        # Fallback: Try the standard datasets library approaches
        print("Attempting standard loading approaches...")
        
        # Try various loading methods
        for attempt_name, load_func in [
            ("with trust_remote_code", lambda: load_dataset(dataset_name, split=split, trust_remote_code=True)),
            ("without trust_remote_code", lambda: load_dataset(dataset_name, split=split)),
            ("with config", lambda: load_dataset(dataset_name, version_tag, split=split)),
            ("basic", lambda: load_dataset(dataset_name, split=split))
        ]:
            try:
                print(f"Trying {attempt_name}...")
                dataset = load_func()
                print(f"Success with {attempt_name}")
                
                # Limit examples if specified
                if num_examples > 0 and len(dataset) > num_examples:
                    dataset = dataset.select(range(num_examples))
                    
                return dataset
            except Exception as e2:
                print(f"Failed {attempt_name}: {e2}")
                continue
        
        raise Exception("All approaches to load LiveCodeBench dataset failed")


def decode_private_test_cases(encoded_str: str) -> Optional[List[Dict[str, Any]]]:
    """
    Decode private test cases using the official LiveCodeBench decoding method.
    
    The private test cases are encoded using:
    1. JSON -> 2. Pickle -> 3. Zlib compression -> 4. Base64 encoding
    
    Args:
        encoded_str: Base64-encoded and compressed string
        
    Returns:
        List of test cases or None if decoding fails
    """
    try:
        # Follow the exact decoding steps from LiveCodeBench
        # 1. Base64 decode
        decoded = base64.b64decode(encoded_str.encode("utf-8"))
        
        # 2. Zlib decompress
        decompressed = zlib.decompress(decoded)
        
        # 3. Pickle loads
        unpickled = pickle.loads(decompressed)
        
        # 4. JSON loads (unpickled is already a string)
        test_cases = json.loads(unpickled)
        
        return test_cases if isinstance(test_cases, list) else None
        
    except Exception as e:
        # If decoding fails, return None
        return None


def load_environment(
    version_tag: str = "release_v5",
    split: str = "test",
    num_examples: int = -1,
    **kwargs
) -> vf.Environment:
    """Load LiveCodeBench environment for evaluation
    
    Args:
        version_tag: LiveCodeBench version to load
        split: Dataset split (always "test" for LiveCodeBench)
        num_examples: Number of examples to load (-1 for all)
        **kwargs: Additional arguments (ignored)
    """
    # Load the actual LiveCodeBench dataset
    dataset = load_livecodebench_dataset(
        version_tag=version_tag,
        split=split,
        num_examples=num_examples
    )
    
    # Initialize sandbox executor
    sandbox = SecureSandboxExecutor()
    
    # Track unique warning types to avoid spam
    logged_warnings = set()
    
    # Convert dataset to verifiers format
    converted_dataset = []
    for idx, example in enumerate(dataset):
        # Extract problem information
        problem_id = example.get('question_id', f'problem_{idx}')
        problem_desc = example.get('question_content', '')
        starter_code = example.get('starter_code', '')
        
        # Extract test cases - they're split between public and private
        test_cases = []
        
        # Add public test cases
        if 'public_test_cases' in example and example['public_test_cases']:
            try:
                public_tests = json.loads(example['public_test_cases'])
                if isinstance(public_tests, list):
                    test_cases.extend(public_tests)
            except json.JSONDecodeError:
                print(f"Failed to parse public test cases for {problem_id}")
        
        # Add private test cases
        if 'private_test_cases' in example and example['private_test_cases']:
            private_raw = example['private_test_cases']
            
            # Check if it's already a list
            if isinstance(private_raw, list):
                test_cases.extend(private_raw)
            elif isinstance(private_raw, str):
                # Try direct JSON parsing first
                try:
                    private_tests = json.loads(private_raw)
                    if isinstance(private_tests, list):
                        test_cases.extend(private_tests)
                except json.JSONDecodeError:
                    # Try to decode using the official LiveCodeBench method
                    decoded_tests = decode_private_test_cases(private_raw)
                    if decoded_tests:
                        test_cases.extend(decoded_tests)
            
        # Format the prompt
        prompt = f"{problem_desc}\n\n"
        if starter_code:
            prompt += f"Starter code:\n```python\n{starter_code}\n```\n\n"
        if example.get('public_tests'):
            prompt += f"Example test cases:\n{example['public_tests']}\n\n"
        
        # Parse the test cases into the expected format
        parsed_tests = []
        for test in test_cases:
            if isinstance(test, dict) and 'input' in test and 'output' in test:
                parsed_tests.append({
                    'input': test['input'],
                    'output': test['output'],
                    'testtype': test.get('testtype', 'public')
                })
        
        converted_dataset.append({
            'prompt': [{'role': 'user', 'content': prompt}],
            'answer': '',  # Will be filled by the model
            'info': {
                'problem_id': problem_id,
                'problem_desc': problem_desc,
                'starter_code': starter_code,
                'test_cases': parsed_tests,
                'difficulty': example.get('difficulty', 'unknown'),
                'contest_date': example.get('contest_date', ''),
            }
        })
    
    # Create dataset
    dataset = Dataset.from_list(converted_dataset)
    
    # Create parser and rubric
    def extract_code(completion: str) -> str:
        """Extract code from model completion"""
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
        
        # If no code blocks, look for function definition
        if 'def ' in completion:
            # Extract from first def to the end
            def_index = completion.find('def ')
            return completion[def_index:]
        
        # Return the entire completion as last resort
        return completion
    
    parser = vf.Parser(extract_fn=extract_code)
    
    # Define rubric functions
    def correctness_score(completion, info, parser=None, **kwargs) -> float:
        """Evaluate code correctness using test cases"""
        if parser is None:
            parser = kwargs.get('parser', vf.Parser(extract_fn=extract_code))
            
        code = parser.parse_answer(completion)
        
        if not code or not info.get('test_cases'):
            return 0.0
        
        # Add starter code if provided
        if info.get('starter_code'):
            code = info['starter_code'] + '\n\n' + code
        
        # Run all test cases
        passed = 0
        total = len(info['test_cases'])
        
        for i, test in enumerate(info['test_cases']):
            test_input = test.get('input', '')
            expected_output = test.get('output', '').strip()
            
            # Execute code with test input
            stdout, stderr, exit_code = sandbox.execute(code, test_input)
            
            # Debug logging
            if i == 0:  # Log first test case for debugging
                print(f"\n=== Test case {i+1} ===")
                print(f"Input: {test_input[:100]}...")
                print(f"Expected: {expected_output[:100]}...")
                print(f"Got stdout: {stdout[:100]}...")
                print(f"Got stderr: {stderr[:100]}...")
                print(f"Exit code: {exit_code}")
            
            if exit_code == 0 and stdout.strip() == expected_output:
                passed += 1
        
        return passed / total if total > 0 else 0.0
    
    def execution_success(completion, info, parser=None, **kwargs) -> float:
        """Check if code executes without errors"""
        if parser is None:
            parser = kwargs.get('parser', vf.Parser(extract_fn=extract_code))
            
        code = parser.parse_answer(completion)
        
        if not code:
            return 0.0
        
        # Add starter code if provided
        if info.get('starter_code'):
            code = info['starter_code'] + '\n\n' + code
        
        # Try to execute with a simple test
        _, stderr, exit_code = sandbox.execute(code, "")
        
        return 1.0 if exit_code == 0 else 0.0
    
    # Create rubric
    rubric = vf.Rubric(
        funcs=[correctness_score, execution_success],
        weights=[1.0, 0.0],  # Only correctness_score contributes to the final score
        parser=parser
    )
    
    # Create environment
    env = vf.SingleTurnEnv(
        dataset=dataset,
        rubric=rubric, 
        parser=parser
    )
    
    return env