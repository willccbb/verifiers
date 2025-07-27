"""
LiveCodeBench Implementation Example for Verifiers Environment

This example demonstrates a concrete implementation of LiveCodeBench
within the verifiers framework, focusing on secure sandbox execution.
"""

import os
import json
import tempfile
import subprocess
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from contextlib import contextmanager
import docker
import verifiers as vf
from datasets import Dataset, load_dataset


@dataclass
class SandboxConfig:
    """Configuration for sandbox execution"""
    memory_limit: str = "512m"
    cpu_quota: int = 50000  # 0.5 CPU
    timeout: int = 30
    max_processes: int = 50
    network_enabled: bool = False
    read_only_root: bool = True


class SecureSandboxExecutor:
    """Handles secure execution of code in isolated environments"""
    
    def __init__(self, config: SandboxConfig):
        self.config = config
        self.docker_available = self._check_docker_availability()
        
    def _check_docker_availability(self) -> bool:
        """Check if Docker is available and accessible"""
        try:
            client = docker.from_env()
            client.ping()
            return True
        except:
            return False
    
    @contextmanager
    def docker_sandbox(self, language: str, code: str, test_input: str):
        """Execute code in Docker container with security constraints"""
        if not self.docker_available:
            raise RuntimeError("Docker not available, use process sandbox instead")
            
        client = docker.from_env()
        
        # Create temporary directory for code
        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = os.path.join(tmpdir, "solution.py")
            with open(code_path, 'w') as f:
                f.write(code)
            
            # Run container with security constraints
            container = client.containers.run(
                f"python:3.10-alpine",  # Minimal image
                command=["python", "/code/solution.py"],
                volumes={tmpdir: {'bind': '/code', 'mode': 'ro'}},
                mem_limit=self.config.memory_limit,
                cpu_quota=self.config.cpu_quota,
                network_mode='none' if not self.config.network_enabled else 'bridge',
                read_only=self.config.read_only_root,
                user='nobody',  # Run as non-root
                detach=True,
                stdin_open=True,
                environment={'PYTHONDONTWRITEBYTECODE': '1'}
            )
            
            try:
                # Send input and get output
                result = container.exec_run(
                    ["python", "/code/solution.py"],
                    stdin=True,
                    stdout=True,
                    stderr=True,
                    environment={'PYTHONDONTWRITEBYTECODE': '1'}
                )
                
                # Wait for completion with timeout
                container.wait(timeout=self.config.timeout)
                
                yield {
                    'stdout': result.output.decode('utf-8'),
                    'stderr': '',
                    'exit_code': result.exit_code
                }
                
            finally:
                # Clean up
                container.stop()
                container.remove(force=True)
    
    def process_sandbox(self, language: str, code: str, test_input: str) -> Dict:
        """Fallback execution using process isolation with resource limits"""
        import resource
        
        def set_limits():
            # CPU time limit
            resource.setrlimit(resource.RLIMIT_CPU, (self.config.timeout, self.config.timeout))
            # Memory limit (in bytes)
            memory_bytes = int(self.config.memory_limit.rstrip('m')) * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
            # Process limit
            resource.setrlimit(resource.RLIMIT_NPROC, (self.config.max_processes, self.config.max_processes))
        
        try:
            proc = subprocess.run(
                ["python", "-c", code],
                input=test_input,
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
                preexec_fn=set_limits if os.name != 'nt' else None
            )
            
            return {
                'stdout': proc.stdout,
                'stderr': proc.stderr,
                'exit_code': proc.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                'stdout': '',
                'stderr': 'Execution timed out',
                'exit_code': -1
            }
        except Exception as e:
            return {
                'stdout': '',
                'stderr': str(e),
                'exit_code': -1
            }
    
    def run_tests(self, code: str, test_cases: List[Dict], language: str) -> Dict:
        """Run multiple test cases against code"""
        results = []
        
        for test in test_cases:
            test_input = test.get('input', '')
            expected_output = test.get('output', '')
            
            # Execute in sandbox
            if self.docker_available:
                with self.docker_sandbox(language, code, test_input) as result:
                    actual_output = result['stdout'].strip()
                    passed = actual_output == expected_output.strip()
                    results.append({
                        'passed': passed,
                        'input': test_input,
                        'expected': expected_output,
                        'actual': actual_output,
                        'error': result['stderr']
                    })
            else:
                result = self.process_sandbox(language, code, test_input)
                actual_output = result['stdout'].strip()
                passed = actual_output == expected_output.strip()
                results.append({
                    'passed': passed,
                    'input': test_input,
                    'expected': expected_output,
                    'actual': actual_output,
                    'error': result['stderr']
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


class LiveCodeBenchEnv(vf.MultiTurnEnv):
    """LiveCodeBench environment for verifiers"""
    
    def __init__(
        self,
        dataset_version: str = "release_v1",
        language: str = "python",
        scenario: str = "code_generation",
        sandbox_config: Optional[SandboxConfig] = None,
        **kwargs
    ):
        self.language = language
        self.scenario = scenario
        self.sandbox = SecureSandboxExecutor(sandbox_config or SandboxConfig())
        
        # Load and prepare dataset
        dataset = self._load_dataset(dataset_version)
        
        # Create parser for code extraction
        parser = vf.Parser(extract_fn=self._extract_code)
        
        # Define rubric based on scenario
        rubric = self._create_rubric(scenario, parser)
        
        super().__init__(
            dataset=dataset,
            rubric=rubric,
            parser=parser,
            max_turns=1 if scenario == "code_generation" else 3,
            **kwargs
        )
    
    def _load_dataset(self, version: str) -> Dataset:
        """Load LiveCodeBench dataset and convert to verifiers format"""
        # This would load from HuggingFace in production
        # For now, create example data
        examples = [
            {
                'prompt': [{'role': 'user', 'content': self._format_problem({
                    'description': 'Write a function that returns the sum of two numbers.',
                    'examples': [
                        {'input': '1 2', 'output': '3'},
                        {'input': '5 7', 'output': '12'}
                    ]
                })}],
                'info': {
                    'test_cases': [
                        {'input': '1 2', 'output': '3'},
                        {'input': '5 7', 'output': '12'},
                        {'input': '0 0', 'output': '0'},
                        {'input': '-1 1', 'output': '0'}
                    ],
                    'language': 'python',
                    'difficulty': 'easy',
                    'source': 'example',
                    'release_date': '2024-01-01'
                }
            }
        ]
        
        return Dataset.from_list(examples)
    
    def _format_problem(self, problem: Dict) -> str:
        """Format problem statement for prompt"""
        prompt = f"{problem['description']}\n\n"
        prompt += "Examples:\n"
        for ex in problem['examples']:
            prompt += f"Input: {ex['input']}\n"
            prompt += f"Output: {ex['output']}\n\n"
        return prompt
    
    def _extract_code(self, completion: List[Dict]) -> str:
        """Extract code from model completion"""
        if not completion:
            return ""
        
        last_message = completion[-1]['content']
        
        # Extract code between ```python and ```
        import re
        code_match = re.search(r'```python\n(.*?)```', last_message, re.DOTALL)
        if code_match:
            return code_match.group(1)
        
        # Fallback: assume entire message is code
        return last_message
    
    def _create_rubric(self, scenario: str, parser: vf.Parser) -> vf.Rubric:
        """Create evaluation rubric based on scenario"""
        if scenario == "code_generation":
            return vf.Rubric(
                funcs=[
                    self._correctness_score,
                    self._execution_time,
                    self._code_quality
                ],
                weights=[1.0, 0.0, 0.0],
                parser=parser
            )
        elif scenario == "self_repair":
            return vf.Rubric(
                funcs=[
                    self._correctness_score,
                    self._repair_success,
                    self._num_attempts
                ],
                weights=[1.0, 0.5, -0.1],
                parser=parser
            )
        else:
            raise ValueError(f"Unknown scenario: {scenario}")
    
    def _correctness_score(self, parser, completion, info) -> float:
        """Evaluate code correctness using test cases"""
        code = parser.parse_answer(completion)
        
        if not code:
            return 0.0
        
        # Run tests in sandbox
        test_results = self.sandbox.run_tests(
            code=code,
            test_cases=info['test_cases'],
            language=info['language']
        )
        
        return test_results['pass_rate']
    
    def _execution_time(self, parser, completion, info) -> float:
        """Measure execution time (for analysis, not scoring)"""
        # This would measure actual execution time in production
        return 0.0
    
    def _code_quality(self, parser, completion, info) -> float:
        """Assess code quality metrics"""
        # This could include style checks, complexity analysis, etc.
        return 0.0
    
    def _repair_success(self, parser, completion, info) -> float:
        """Check if code was successfully repaired"""
        # For self-repair scenario
        return 0.0
    
    def _num_attempts(self, parser, completion, info) -> float:
        """Count number of repair attempts"""
        # For self-repair scenario
        return float(len([m for m in completion if m['role'] == 'assistant']))
    
    def is_completed(self, messages: List[Dict], state: Dict, **kwargs) -> bool:
        """Check if evaluation is complete"""
        if self.scenario == "code_generation":
            # Single turn: complete after first response
            return any(msg['role'] == 'assistant' for msg in messages)
        else:
            # Multi-turn: complete after success or max attempts
            if state.get('all_tests_passed'):
                return True
            
            num_attempts = len([m for m in messages if m['role'] == 'assistant'])
            return num_attempts >= self.max_turns
    
    def env_response(self, messages: List[Dict], state: Dict, **kwargs) -> Tuple[List[Dict], Dict]:
        """Provide environment feedback for multi-turn scenarios"""
        if self.scenario == "self_repair" and not state.get('all_tests_passed'):
            # Get last code submission
            code = self.parser.parse_answer(messages)
            
            # Run tests
            test_results = self.sandbox.run_tests(
                code=code,
                test_cases=state['info']['test_cases'],
                language=state['info']['language']
            )
            
            # Update state
            state['all_tests_passed'] = test_results['pass_rate'] == 1.0
            
            # Provide feedback
            if not state['all_tests_passed']:
                failed_test = next(r for r in test_results['results'] if not r['passed'])
                feedback = f"Test failed:\nInput: {failed_test['input']}\n"
                feedback += f"Expected: {failed_test['expected']}\n"
                feedback += f"Got: {failed_test['actual']}\n"
                if failed_test['error']:
                    feedback += f"Error: {failed_test['error']}"
                
                return [{'role': 'user', 'content': feedback}], state
        
        return [], state


def load_environment(**kwargs) -> LiveCodeBenchEnv:
    """Entry point for verifiers framework"""
    return LiveCodeBenchEnv(**kwargs)


# Example usage
if __name__ == "__main__":
    # Create environment
    env = LiveCodeBenchEnv(
        dataset_version="release_v1",
        language="python",
        scenario="code_generation"
    )
    
    # Simulate evaluation
    from openai import OpenAI
    client = OpenAI()
    
    results = env.evaluate(
        client=client,
        model="gpt-4",
        num_examples=10,
        rollouts_per_example=1
    )
    
    print(f"Average pass rate: {results['metrics']['correctness_score']:.2%}")