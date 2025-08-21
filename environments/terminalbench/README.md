# Terminal-Bench Environment

This environment implements the Terminal-Bench dataset from HuggingFace (`ia03/terminal-bench`) for the verifiers framework. Terminal-Bench evaluates AI agents on real-world, end-to-end terminal tasks ranging from compiling code and training models to setting up servers and debugging systems.

## Features

- **Real Terminal Tasks**: Complete terminal environment tasks with Docker sandboxing
- **Archive Extraction**: Automatic extraction and integrity verification of task archives
- **Docker Integration**: Secure containerized execution of terminal commands
- **Comprehensive Evaluation**: Binary pass/fail scoring based on test suite execution
- **Resource Management**: Automatic cleanup of Docker containers and images

## Requirements

- Docker installed and running
- Python packages: `datasets`, `docker`, `PyYAML`

## Dataset

The Terminal-Bench dataset contains 112 tasks across various categories:

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
import environments.terminalbench as tb

# Load the environment
env = tb.load_environment(
    dataset_name="ia03/terminal-bench",
    split="test", 
    num_examples=10,  # Load first 10 tasks, -1 for all
    pool_size=5       # Docker container pool size
)

# The environment expects shell commands as model output
# Example model response should contain commands like:
# ```bash
# cd /path/to/directory
# make build
# python test.py
# ```
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
5. **Cleanup**: Container and images automatically removed

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