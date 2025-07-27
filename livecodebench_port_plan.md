# LiveCodeBench Port to Verifiers Environment: Implementation Plan

## Overview
This document outlines a robust, clean approach for porting LiveCodeBench (LCB) to the verifiers environment while maintaining exact ground truth methodology and leveraging verifiers' infrastructure.

## Key Design Principles

### 1. Containerized Sandbox Execution
- Use Docker containers for isolated code execution (similar to LiveCodeBench's approach)
- Implement resource limits (CPU, memory, time) to prevent abuse
- Network isolation to prevent external communication
- Filesystem isolation with read-only system directories

### 2. Ground Truth Methodology Alignment
- Preserve LCB's test-based evaluation approach
- Support multiple programming languages (Python, C++, Java, etc.)
- Maintain exact scoring metrics (pass@1, pass@5)
- Keep problem metadata (release dates, difficulty, platform source)

### 3. Integration with Verifiers Infrastructure
- Leverage `MultiTurnEnv` for interactive problem-solving scenarios
- Use `ToolEnv` for code execution with sandboxed Python tool
- Implement custom `Rubric` for test-based evaluation
- Support dataset versioning and time-windowed evaluation

## Architecture Design

### Core Components

```
LiveCodeBench Environment
├── Dataset Management
│   ├── Problem Loader (from HF datasets)
│   ├── Version Manager (release_v1 - release_latest)
│   └── Time Window Filter
├── Sandbox Execution Engine
│   ├── Docker Container Manager
│   ├── Language-specific Runners
│   └── Resource Monitor
├── Evaluation Framework
│   ├── Test Runner
│   ├── Output Validator
│   └── Metric Calculator
└── Verifiers Integration
    ├── Environment Classes
    ├── Parser Implementation
    └── Rubric Functions
```

### Sandbox Execution Strategy

#### Option 1: Docker-in-Docker (Recommended for Production)
```python
# Use official Docker SDK
import docker
from contextlib import contextmanager

@contextmanager
def sandboxed_execution(language, code, test_cases, limits):
    client = docker.from_env()
    container = client.containers.run(
        f"livecodebench/{language}:latest",
        command=["python", "-c", execution_script],
        detach=True,
        mem_limit=limits['memory'],
        cpu_quota=limits['cpu_quota'],
        network_mode='none',
        read_only=True,
        volumes={
            '/tmp/code': {'bind': '/workspace', 'mode': 'rw'},
            '/tmp/tests': {'bind': '/tests', 'mode': 'ro'}
        }
    )
    # ... execution and cleanup
```

#### Option 2: Process Isolation with Resource Limits
```python
# For environments where Docker is not available
import resource
import subprocess
from multiprocessing import Process, Queue

def limited_execution(code, test_input, limits):
    def set_limits():
        # CPU time limit
        resource.setrlimit(resource.RLIMIT_CPU, (limits['cpu_seconds'], limits['cpu_seconds']))
        # Memory limit
        resource.setrlimit(resource.RLIMIT_AS, (limits['memory_mb'] * 1024 * 1024, limits['memory_mb'] * 1024 * 1024))
        # Disable network
        os.setgid(limits['sandbox_gid'])
        os.setuid(limits['sandbox_uid'])
    
    # Execute in isolated process
    proc = subprocess.Popen(
        [sys.executable, '-c', code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=set_limits
    )
```

### Environment Implementation

```python
# environments/livecodebench/livecodebench.py
import verifiers as vf
from typing import Dict, List, Tuple
from .sandbox import SandboxExecutor
from .dataset_loader import LiveCodeBenchDataset

class LiveCodeBenchEnv(vf.MultiTurnEnv):
    def __init__(
        self,
        dataset_version: str = "release_latest",
        languages: List[str] = ["python", "cpp", "java"],
        sandbox_config: Dict = None,
        **kwargs
    ):
        self.sandbox = SandboxExecutor(sandbox_config or default_sandbox_config())
        self.languages = languages
        
        # Load dataset with version control
        dataset = LiveCodeBenchDataset.load(version=dataset_version)
        
        # Create rubric for test-based evaluation
        rubric = vf.Rubric(
            funcs=[
                self.correctness_score,
                self.execution_time,
                self.memory_usage,
                self.test_coverage
            ],
            weights=[1.0, 0.0, 0.0, 0.0]
        )
        
        super().__init__(
            dataset=dataset,
            rubric=rubric,
            max_turns=1,  # Single-turn for basic code generation
            **kwargs
        )
    
    def correctness_score(self, prompt, completion, info) -> float:
        """Run tests and calculate pass rate"""
        code = self.parser.extract_code(completion)
        test_results = self.sandbox.run_tests(
            code=code,
            tests=info['test_cases'],
            language=info['language']
        )
        return test_results['pass_rate']
```

### Sandbox Security Best Practices

1. **Container Security**
   - Use minimal base images (distroless/alpine)
   - Drop all capabilities except required ones
   - Run as non-root user
   - Read-only root filesystem
   - No network access

2. **Resource Limits**
   ```yaml
   # Default limits per execution
   limits:
     cpu_quota: 50000  # 0.5 CPU
     memory: "512m"
     pids: 50
     timeout: 30s
     disk_quota: "100m"
   ```

3. **Code Sanitization**
   - Filter dangerous imports/includes
   - Restrict file system access
   - Prevent system calls
   - Validate input/output sizes

### Dataset Management

```python
class LiveCodeBenchDataset:
    """Handles LCB dataset with versioning and filtering"""
    
    @classmethod
    def load(cls, version="release_latest", start_date=None, end_date=None):
        # Load from HuggingFace
        dataset = load_dataset("livecodebench/livecodebench", version)
        
        # Apply time window filtering
        if start_date or end_date:
            dataset = cls.filter_by_date(dataset, start_date, end_date)
        
        # Convert to verifiers format
        return cls.to_verifiers_format(dataset)
    
    @staticmethod
    def to_verifiers_format(lcb_dataset):
        """Convert LCB format to verifiers Dataset format"""
        examples = []
        for item in lcb_dataset:
            examples.append({
                'prompt': cls.format_prompt(item),
                'info': {
                    'test_cases': item['test_cases'],
                    'language': item['language'],
                    'difficulty': item['difficulty'],
                    'source': item['source'],
                    'release_date': item['release_date']
                }
            })
        return Dataset.from_list(examples)
```

### Multi-Language Support

```python
class LanguageRunner:
    """Base class for language-specific execution"""
    
    @abstractmethod
    def prepare_code(self, code: str, test_harness: str) -> str:
        """Prepare code with test harness"""
        pass
    
    @abstractmethod
    def compile_if_needed(self, code_path: str) -> bool:
        """Compile code if required"""
        pass
    
    @abstractmethod
    def execute(self, executable_path: str, test_input: str) -> str:
        """Execute code with test input"""
        pass

class PythonRunner(LanguageRunner):
    def prepare_code(self, code: str, test_harness: str) -> str:
        return f"{code}\n\n{test_harness}"
    
    def compile_if_needed(self, code_path: str) -> bool:
        return True  # No compilation needed
    
    def execute(self, executable_path: str, test_input: str) -> str:
        return subprocess.run(
            ["python", executable_path],
            input=test_input,
            capture_output=True,
            text=True,
            timeout=30
        ).stdout
```

### Evaluation Scenarios

1. **Code Generation**
   ```python
   vf_env = LiveCodeBenchEnv(
       scenario="code_generation",
       dataset_version="release_v6",
       languages=["python"]
   )
   ```

2. **Self-Repair**
   ```python
   class LiveCodeBenchSelfRepairEnv(LiveCodeBenchEnv):
       def __init__(self, **kwargs):
           super().__init__(max_turns=3, **kwargs)
       
       def env_response(self, messages, state):
           # Provide error feedback for repair
           if state.get('last_execution_failed'):
               error_msg = state['last_error']
               return [{"role": "user", "content": f"Error: {error_msg}"}], state
   ```

3. **Test Output Prediction**
   ```python
   class LiveCodeBenchTestPredictionEnv(LiveCodeBenchEnv):
       def correctness_score(self, prompt, completion, info):
           predicted_output = self.parser.extract_output(completion)
           actual_output = self.sandbox.get_test_output(
               code=info['solution'],
               test_input=info['test_input']
           )
           return 1.0 if predicted_output.strip() == actual_output.strip() else 0.0
   ```

## Implementation Timeline

### Phase 1: Core Infrastructure (Week 1-2)
- [ ] Implement sandbox execution engine
- [ ] Create Docker images for each language
- [ ] Set up security policies and resource limits
- [ ] Test isolation and security measures

### Phase 2: Dataset Integration (Week 2-3)
- [ ] Build dataset loader with version management
- [ ] Implement time-window filtering
- [ ] Convert LCB format to verifiers format
- [ ] Add problem metadata handling

### Phase 3: Environment Implementation (Week 3-4)
- [ ] Implement base LiveCodeBenchEnv
- [ ] Add language-specific runners
- [ ] Create evaluation rubrics
- [ ] Implement all LCB scenarios

### Phase 4: Testing & Optimization (Week 4-5)
- [ ] Comprehensive security testing
- [ ] Performance benchmarking
- [ ] Comparison with original LCB results
- [ ] Documentation and examples

## Security Considerations

1. **Defense in Depth**
   - Container isolation (primary)
   - Process isolation (secondary)
   - Resource limits (tertiary)
   - Code sanitization (quaternary)

2. **Monitoring**
   - Log all executions
   - Track resource usage
   - Alert on anomalies
   - Regular security audits

3. **Updates**
   - Regular base image updates
   - Security patch management
   - Vulnerability scanning
   - Dependency updates

## Testing Strategy

1. **Security Tests**
   - Attempt container escape
   - Resource exhaustion attacks
   - Network access attempts
   - File system manipulation

2. **Functionality Tests**
   - Compare results with original LCB
   - Test all supported languages
   - Verify scoring accuracy
   - Edge case handling

3. **Performance Tests**
   - Measure overhead vs native execution
   - Concurrent execution scaling
   - Resource utilization
   - Latency benchmarks

## Conclusion

This implementation plan provides a robust, secure, and scalable approach to porting LiveCodeBench to the verifiers environment. By leveraging Docker containers and following security best practices, we can maintain the integrity of the ground truth methodology while providing a safe execution environment for untrusted code.

The modular design allows for easy extension to new languages and evaluation scenarios, while the integration with verifiers' infrastructure ensures compatibility with existing RL training pipelines.