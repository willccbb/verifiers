# LiveCodeBench Environment for Verifiers

A Verifiers environment implementation for evaluating code generation models on the [LiveCodeBench](https://livecodebench.github.io/) benchmark.

> **⚠️ Important Disclaimer**: This environment was ported to the Verifiers framework by Claude 4 Opus (an AI assistant). While it has been tested and appears to function correctly, it should be carefully vetted by users before relying on it for production evaluations. Please review the implementation and test thoroughly for your specific use case.

## Overview

LiveCodeBench is a holistic and contamination-free benchmark for evaluating Large Language Models (LLMs) on code generation tasks. This port mirrors the official LiveCodeBench ground truth methodology while utilizing the Verifiers infrastructure for evaluation.

### Key Features

- **Real Dataset**: Uses the official LiveCodeBench dataset from HuggingFace
- **Docker-Based Sandboxing**: Secure code execution with resource limits and isolation
- **Multiple Problem Types**: Supports various competitive programming problems
- **Accurate Scoring**: Implements pass@1, correctness score, and execution success metrics

## Installation

```bash
# From the workspace root
uv run vf-init livecodebench
cd environments/livecodebench
uv sync
cd ../..
uv run vf-install livecodebench
```

## Usage

### Basic Evaluation

```bash
# Evaluate with default settings (all problems, 1 rollout each)
uv run vf-eval livecodebench -m gpt-4

# Evaluate with specific number of problems and rollouts
uv run vf-eval livecodebench -m gpt-4 -n 20 -r 5 --env-args '{"num_examples": 20}'

# Use a different dataset version
uv run vf-eval livecodebench -m gpt-4 --env-args '{"version_tag": "release_v4"}'
```

### Available Parameters

- `num_examples`: Number of problems to evaluate (default: all)
- `version_tag`: LiveCodeBench version to use (default: "release_v5")
- `seed`: Random seed for problem selection (default: 42)

## Implementation Details

### Docker Sandboxing

The environment uses Docker containers for secure code execution with:
- **Network isolation**: No internet access
- **Resource limits**: 512MB memory, 0.5 CPU, 50 processes max
- **Time limits**: 10-second timeout per execution
- **Filesystem isolation**: Read-write temporary directory
- **Security**: Non-root user, dropped capabilities

### Dataset Loading

The implementation downloads JSONL files directly from HuggingFace due to the deprecation of `trust_remote_code` in the datasets library. It supports multiple LiveCodeBench versions:
- `release_v1` through `release_v5`
- `release_v2.1`

### Evaluation Metrics

1. **Pass@1**: Whether the generated code passes all test cases
2. **Correctness Score**: Fraction of test cases passed
3. **Execution Success**: Whether the code executes without errors

## Requirements

- Docker must be installed and running
- User must be in the docker group (or have sudo access)
- Python 3.11+ (for consistency with the Docker image)

## Known Limitations

1. Some private test cases may fail to parse due to non-standard JSON formatting in the dataset
2. Network DNS resolution issues may temporarily prevent dataset downloading
3. The environment requires Docker and cannot fall back to subprocess execution

## Debugging

To see detailed output during evaluation:
```bash
# This will show test case parsing warnings and execution details
uv run vf-eval livecodebench -m gpt-4 -n 5 -r 1
```

## Contributing

If you find issues or have improvements:
1. Test your changes thoroughly with multiple models
2. Ensure Docker sandboxing remains secure
3. Verify that scoring matches the official LiveCodeBench methodology

## References

- [LiveCodeBench Paper](https://arxiv.org/abs/2403.07974)
- [LiveCodeBench GitHub](https://github.com/LiveCodeBench/LiveCodeBench)
- [LiveCodeBench Website](https://livecodebench.github.io/)

---

**Remember**: This is an AI-generated port. Please validate the implementation against the official LiveCodeBench before using for critical evaluations.