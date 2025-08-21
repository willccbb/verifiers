# Terminal-Bench Environment

This environment runs Terminal-Bench tasks from the local repository using the Terminal-Bench harness (Docker + tmux). Tasks are read from `terminal-bench/tasks` and provided to the verifiers framework. Terminal-Bench evaluates AI agents on end-to-end terminal tasks ranging from compiling code and training models to setting up servers and debugging systems.

## Features

- **Real Terminal Tasks**: Complete terminal environment tasks with Docker sandboxing
- **Archive Extraction**: Automatic extraction and integrity verification of task archives
- **Docker Integration**: Secure containerized execution of terminal commands
- **Comprehensive Evaluation**: Binary pass/fail scoring based on test suite execution
- **Resource Management**: Automatic cleanup of Docker containers and images

## Requirements

- Docker installed and running
- Python packages: `docker`, `PyYAML`, `terminal-bench` (installed per `environments/terminalbench/pyproject.toml`)

## Tasks

Tasks are discovered from the local `terminal-bench/tasks` folder in this repository checkout. A subset can be selected via `num_examples` (default: all). The Terminal-Bench task suite contains tasks across various categories:

- **algorithms**: Algorithm implementation and optimization
- **data-science**: Data processing and analysis tasks  
- **debugging**: System debugging and troubleshooting
- **file-operations**: File system manipulation
- **games**: Game development and implementation
- **machine-learning**: Model training and evaluation
- **mathematics**: Mathematical computation tasks
- **model-training**: ML model training pipelines
- **personal-assistant**: Task automation scenarios

Each task difficulty is rated as:
- **easy**: Simple, straightforward tasks
- **medium**: Moderate complexity requiring some expertise
- **hard**: Complex tasks requiring advanced knowledge

## Usage

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



## Task Categories

The dataset includes diverse task types across multiple domains:

- **Systems Programming**: C/C++ compilation, debugging, optimization
- **Machine Learning**: Model training, data processing, evaluation  
- **DevOps**: Server setup, containerization, monitoring
- **Security**: Vulnerability analysis, cryptography, forensics
- **Data Science**: ETL pipelines, statistical analysis, visualization
- **Legacy Systems**: COBOL modernization, assembly programming

## Examples

Tasks range from simple file operations to complex multi-step workflows:

- **hello-world**: Basic program compilation and execution
- **pytorch-model-training**: Train neural network from scratch
- **database-setup**: Set up and configure database server
- **vulnerability-analysis**: Analyze and patch security issues
- **data-pipeline**: Build ETL pipeline for data processing

Each task is completely self-contained with all necessary files, dependencies, and test cases included in the archive. 