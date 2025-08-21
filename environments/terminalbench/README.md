# Terminal-Bench Environment

This environment runs Terminal-Bench tasks from the local repository using the Terminal-Bench harness (Docker + tmux). Tasks are read from `terminal-bench/tasks` and provided to the verifiers framework. Terminal-Bench evaluates AI agents on end-to-end terminal tasks ranging from compiling code and training models to setting up servers and debugging systems.

## Requirements

- Docker installed and running
- Python packages: `docker`, `PyYAML`, `terminal-bench` (installed per `environments/terminalbench/pyproject.toml`)

## Tasks

A subset of tasks can be selected via `num_examples` (default: all). The Terminal-Bench task suite contains tasks across various categories:

- **algorithms**: Algorithm implementation and optimization
- **data-science**: Data processing and analysis tasks  
- **debugging**: System debugging and troubleshooting
- **file-operations**: File system manipulation
- **games**: Game development and implementation
- **machine-learning**: Model training and evaluation
- **mathematics**: Mathematical computation tasks
- **model-training**: ML model training pipelines
- **personal-assistant**: Task automation scenarios


## Usage

Please be sure to set a sufficiently high max-tokens.
Example terminal usage:
```bash
uv run vf-eval --api-base-url https://openrouter.ai/api/v1 --api-key-var OPENROUTER_API_KEY --model openai/gpt-5-mini --num-examples 10 --rollouts-per-example 1 --max-tokens 16384 environments.terminalbench.vf_terminalbench
```

```python
from environments.terminalbench.vf_terminalbench import load_environment

# Load the environment (tasks read locally from terminal-bench/tasks)
env = load_environment(
    num_examples=10  # Load first 10 tasks; use -1 for all
)

# The environment exposes a single tool to the agent: execute_commands(commands: List[str], reasoning: str = "")
# Agent/tool calls should provide non-interactive shell commands to complete the task inside the container.
```

## Task Structure

Each task is extracted from a gzipped tarball containing:

```
task_id/
├── Dockerfile              # Container definition
├── task.yaml              # Task configuration  
├── solution.sh            # Reference solution
├── tests/                 # Test validation
│   └── test_outputs.py   # pytest test file
├── run-tests.sh          # Optional custom test runner
└── [additional files]    # Data, binaries, models, etc.
```

## Evaluation Process

1. **Archive Extraction**: Task archive is extracted and integrity verified
2. **Container Setup**: Docker image built from task Dockerfile
3. **Agent Execution**: Model-generated commands executed in container
4. **Test Validation**: Test suite run to verify task completion
5. **Cleanup**: Cleanup is handled automatically

## Scoring

- **Binary Success**: 1.0 if all tests pass, 0.0 otherwise
- **Detailed Output**: Agent output and test results available in state

## Docker Requirements

Tasks may require:
- Internet connectivity for package installation
- Elevated privileges for certain operations
- Specific base images (Ubuntu/Debian with apt)
- Resource limits (2GB memory, 1 CPU by default)
