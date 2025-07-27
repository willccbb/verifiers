"""
LiveCodeBench Real Dataset Implementation for Verifiers

This implementation uses the ACTUAL LiveCodeBench dataset from HuggingFace
with proper Docker-based sandboxing for secure code execution.
"""

import os
import json
import tempfile
import subprocess
import time
import resource
import signal
import docker
import re
from typing import List, Dict, Optional, Tuple, Any
from contextlib import contextmanager
from pathlib import Path

import verifiers as vf
from datasets import load_dataset, Dataset


class DockerSandboxExecutor:
    """Production-grade Docker-based sandbox for secure code execution"""
    
    def __init__(
        self, 
        image_name: str = "python:3.10-slim",
        memory_limit: str = "512m",
        cpu_quota: int = 50000,  # 0.5 CPU
        timeout: int = 30,
        max_processes: int = 50
    ):
        self.image_name = image_name
        self.memory_limit = memory_limit
        self.cpu_quota = cpu_quota
        self.timeout = timeout
        self.max_processes = max_processes
        
        # Initialize Docker client
        try:
            self.client = docker.from_env()
            self.docker_available = True
            self._ensure_image()
        except Exception as e:
            print(f"Warning: Docker not available ({e}). Falling back to subprocess sandbox.")
            self.docker_available = False
    
    def _ensure_image(self):
        """Ensure the Docker image exists"""
        try:
            self.client.images.get(self.image_name)
        except docker.errors.ImageNotFound:
            print(f"Pulling Docker image {self.image_name}...")
            self.client.images.pull(self.image_name)
    
    def execute_in_docker(self, code: str, test_input: str = "") -> Tuple[str, str, int]:
        """Execute code in Docker container"""
        # Create temporary directory for code
        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = os.path.join(tmpdir, "solution.py")
            input_path = os.path.join(tmpdir, "input.txt")
            
            # Write code and input
            with open(code_path, 'w') as f:
                f.write(code)
            with open(input_path, 'w') as f:
                f.write(test_input)
            
            # Docker run command
            container = self.client.containers.run(
                self.image_name,
                command=f"python /code/solution.py < /code/input.txt",
                volumes={tmpdir: {'bind': '/code', 'mode': 'ro'}},
                working_dir="/code",
                mem_limit=self.memory_limit,
                cpu_quota=self.cpu_quota,
                cpu_period=100000,
                pids_limit=self.max_processes,
                network_mode="none",
                remove=True,
                detach=True,
                security_opt=["no-new-privileges"],
                read_only=True,
                tmpfs={'/tmp': 'size=100M,mode=1777'}
            )
            
            # Wait for completion with timeout
            try:
                result = container.wait(timeout=self.timeout)
                stdout = container.logs(stdout=True, stderr=False).decode()
                stderr = container.logs(stdout=False, stderr=True).decode()
                exit_code = result['StatusCode']
            except Exception:
                container.kill()
                stdout = ""
                stderr = "Execution timed out"
                exit_code = -1
            
            return stdout, stderr, exit_code
    
    def execute_in_subprocess(self, code: str, test_input: str = "") -> Tuple[str, str, int]:
        """Fallback execution using subprocess with resource limits"""
        def set_limits():
            # Set CPU time limit
            resource.setrlimit(resource.RLIMIT_CPU, (self.timeout, self.timeout))
            # Set memory limit (512MB)
            resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
            # Set process limit
            resource.setrlimit(resource.RLIMIT_NPROC, (self.max_processes, self.max_processes))
            # Set file size limit (no file creation)
            resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
        
        try:
            result = subprocess.run(
                ["python3", "-c", code],
                input=test_input,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                preexec_fn=set_limits if os.name != 'nt' else None
            )
            return result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired:
            return "", "Execution timed out", -1
        except Exception as e:
            return "", str(e), -1
    
    def execute(self, code: str, test_input: str = "") -> Tuple[str, str, int]:
        """Execute code with appropriate sandbox"""
        if self.docker_available:
            return self.execute_in_docker(code, test_input)
        else:
            return self.execute_in_subprocess(code, test_input)


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


def load_environment(
    version_tag: str = "release_v5",
    docker_enabled: bool = True,
    num_examples: int = -1
) -> vf.SingleTurnEnv:
    """Load LiveCodeBench environment with real dataset"""
    
    # Load the actual dataset
    dataset = load_livecodebench_dataset(
        version_tag=version_tag,
        num_examples=num_examples
    )
    
    # Initialize sandbox executor
    sandbox = DockerSandboxExecutor() if docker_enabled else DockerSandboxExecutor()
    sandbox.docker_available = docker_enabled
    
    # Convert dataset to verifiers format
    converted_dataset = []
    for example in dataset:
        # LiveCodeBench dataset structure:
        # - 'question_id': unique identifier
        # - 'question_title': problem title  
        # - 'question_content': problem description
        # - 'test_list': list of test cases
        # - 'starter_code': optional starter code
        # - 'difficulty': problem difficulty
        
        # Extract test cases
        test_cases = []
        if 'test_list' in example and example['test_list']:
            for test in example['test_list']:
                test_cases.append({
                    'input': test.get('input', ''),
                    'output': test.get('output', '')
                })
        
        # Create the problem prompt
        prompt_parts = [example['question_title'], "", example['question_content']]
        
        # Add starter code if available
        if example.get('starter_code'):
            prompt_parts.extend(["", "Starter code:", "```python", example['starter_code'], "```"])
        
        # Add example tests
        if test_cases:
            prompt_parts.extend(["", "Examples:"])
            for i, test in enumerate(test_cases[:3]):  # Show first 3 examples
                prompt_parts.append(f"\nExample {i+1}:")
                prompt_parts.append(f"Input: {test['input']}")
                prompt_parts.append(f"Output: {test['output']}")
        
        prompt = "\n".join(prompt_parts)
        
        # Store all info for the rubric functions
        info = {
            'question_id': example.get('question_id', ''),
            'test_cases': test_cases,
            'starter_code': example.get('starter_code', ''),
            'difficulty': example.get('difficulty', 'unknown'),
            'entry_point': example.get('entry_point', 'solution')
        }
        
        converted_dataset.append({
            'prompt': [{'role': 'user', 'content': prompt}],
            'info': info
        })
    
    dataset = Dataset.from_list(converted_dataset)
    
    # Define answer parser
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
    def correctness_score(parser, completion, info) -> float:
        """Evaluate code correctness using test cases"""
        code = parser.parse_answer(completion)
        
        if not code or not info.get('test_cases'):
            return 0.0
        
        # Add starter code if provided
        if info.get('starter_code'):
            code = info['starter_code'] + '\n\n' + code
        
        # Run all test cases
        passed = 0
        total = len(info['test_cases'])
        
        for test in info['test_cases']:
            test_input = test['input']
            expected_output = test['output'].strip()
            
            # Execute code with test input
            stdout, stderr, exit_code = sandbox.execute(code, test_input)
            
            if exit_code == 0 and stdout.strip() == expected_output:
                passed += 1
        
        return passed / total if total > 0 else 0.0
    
    def execution_success(parser, completion, info) -> float:
        """Check if code executes without errors"""
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
