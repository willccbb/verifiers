# LiveCodeBench Port for Verifiers

This is a production-ready port of LiveCodeBench to the verifiers evaluation framework. It provides secure, containerized execution of untrusted code with exact replication of LiveCodeBench's evaluation methodology.

## Features

### ✅ Real Dataset
- Uses the **actual** LiveCodeBench dataset from HuggingFace (`livecodebench/code_generation_lite`)
- Supports all LiveCodeBench versions (v1-v5)
- Handles both public and hidden test cases
- Preserves exact problem metadata (difficulty, contest info, dates)

### 🔒 Secure Sandboxing
- **Docker-based isolation** (preferred)
  - Network isolation
  - Resource limits (CPU, memory, processes)
  - Non-root execution
  - Read-only filesystem where possible
  - Dropped capabilities
- **Subprocess fallback** with resource limits
  - CPU time limits
  - Memory limits  
  - Process limits
  - File size limits

### 📊 Exact Evaluation
- Replicates LiveCodeBench's pass@1 metric
- Supports code extraction from various formats
- Handles starter code injection
- Exact output matching

## Installation

```bash
# Install the environment
uv run vf-install livecodebench-port -p environments
```

## Usage

### Basic Evaluation
```bash
# Run evaluation with 5 examples
uv run vf-eval livecodebench-port -m gpt-4 -n 5

# Use specific LiveCodeBench version
uv run vf-eval livecodebench-port -m gpt-4 -n 10 --env-kwargs '{"version_tag": "release_v3"}'

# Disable Docker (use subprocess sandboxing)
uv run vf-eval livecodebench-port -m gpt-4 -n 5 --env-kwargs '{"docker_enabled": false}'
```

### Available Options

- `version_tag`: LiveCodeBench dataset version (`release_v1` through `release_v5`, default: `release_v5`)
- `split`: Dataset split to use (default: `test`)
- `num_examples`: Number of examples to evaluate (-1 for all)
- `docker_enabled`: Whether to use Docker sandboxing (default: `true`)

## Docker Sandbox Details

The Docker sandbox provides defense-in-depth security:

1. **Container Isolation**
   - Separate container per execution
   - No network access (`network_mode: none`)
   - Non-root user execution

2. **Resource Limits**
   - Memory: 512MB
   - CPU: 50% of one core
   - Processes: 50 max
   - File size: 50MB max
   - Execution timeout: 30 seconds

3. **Security Features**
   - All capabilities dropped
   - No new privileges
   - Read-only root filesystem (where possible)
   - No package installation

## Differences from Original LiveCodeBench

This port maintains exact compatibility with LiveCodeBench's evaluation methodology while adding:

1. **Enhanced Security**: Docker-based sandboxing vs basic subprocess
2. **Verifiers Integration**: Compatible with verifiers' evaluation pipeline
3. **Flexible Execution**: Can run with or without Docker
4. **Resource Management**: Better handling of resource limits

## Troubleshooting

### Docker Not Available
If Docker is not installed or not running, the environment will automatically fall back to subprocess-based sandboxing with resource limits. While less secure, this still provides basic isolation.

### Dataset Loading Issues
The environment will attempt to load the official LiveCodeBench dataset from HuggingFace. Ensure you have internet connectivity and valid HuggingFace credentials if required.

### Out of Memory
If you encounter OOM errors, you can adjust the memory limit in the environment initialization or reduce the number of parallel evaluations.

## Security Considerations

This implementation follows defense-in-depth principles:

1. **Primary Defense**: Docker containerization
2. **Secondary Defense**: Resource limits (rlimits)
3. **Tertiary Defense**: Timeout enforcement
4. **Code Validation**: Careful parsing and validation of generated code

Even with these protections, running untrusted code carries inherent risks. Always run evaluations in an isolated environment.

## Citation

If you use this port in your research, please cite both the original LiveCodeBench paper and the verifiers framework:

```bibtex
@article{jain2024livecodebench,
  title={LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code},
  author={Jain, Naman and Han, King and Gu, Alex and Li, Wen-Ding and Yan, Fanjia and Zhang, Tianjun and Wang, Sida and Solar-Lezama, Armando and Sen, Koushik and Stoica, Ion},
  journal={arXiv preprint arXiv:2403.07974},
  year={2024}
}
```
