# Modal Setup and Training Guide

This guide explains how to use the current `run_modal.py` script for training verifiers with GPU support on Modal.

## Prerequisites

1. **Modal Account**: Sign up at [modal.com](https://modal.com)
2. **Modal CLI**: Install with `pip install modal`
3. **Authentication**: Run `modal setup` and follow the prompts
4. **Environment Variables**: Create a `.env` file based on `.env.example`:
   ```bash
   cp .env.example .env
   ```
   
   Then edit `.env` to include your API keys:
   ```bash
   WANDB_API_KEY="your-wandb-key"
   HF_TOKEN="your-huggingface-token"  # Optional, for private models
   OPENAI_API_KEY="your-openai-key"  # Optional, used by vLLM as dummy key
   EXPERIMENTER="your-name"  # Used for experiment naming and volume isolation
   EXPERIMENT_NAME="experiment-description"  # Optional, defaults to "default"
   ```

   The script automatically loads these from your `.env` file and passes them to Modal via `modal.Secret.from_dotenv()`.

## Configuration Files

The script supports YAML configuration files in the `configs/` directory for reproducible training runs:

### Available Configs

- `configs/modal_training_config.yaml` - General purpose training configuration
- `configs/modal_training_config_wordle.yaml` - Optimized for Wordle environment
- `configs/modal_training_config_bfcl.yaml` - Optimized for BFCL function calling

### Config Structure

```yaml
# Modal compute resources
modal:
  cpus: 8.0
  memory: 32768  # MB
  gpu:
    count: 1
    type: "A100-40GB"  # T4, A10G, A100, A100-40GB, A100-80GB, H100
  timeout: 14400  # 4 hours

# Model configuration  
model:
  base_model: "Qwen/Qwen2.5-1.5B-Instruct"
  max_token_length: 4096
  dtype: "bfloat16"
  attn_implementation: "sdpa"
  # Environment-specific model overrides
  overrides:
    vf-bfcl-single-turn: "Salesforce/xLAM-2-3b-fc-r"

# Training parameters
training:
  environment: "vf-wordle"
  steps: 200
  batch_size: 8
  gradient_accumulation_steps: 8
  learning_rate: 3e-6
  max_grad_norm: 0.1

# vLLM server settings (for RL training)
vllm:
  gpu_memory_utilization: 0.5
  max_model_len: 4096
  block_size: 16
  enforce_eager: true
  enable_auto_tool_choice: true  # For function calling
  tool_call_parser: "hermes"

# Weights & Biases
wandb:
  project: "verifiers"
  enabled: true
```

## Training Commands

The `run_modal.py` script provides several entry points for different training scenarios:

### Method 1: Direct Parameters (Quick Start)

Train with command-line parameters:
```bash
# Basic training with defaults
modal run run_modal.py::train --env wordle --steps 200

# Specify GPU configuration
modal run run_modal.py::train --env vf-bfcl-single-turn --steps 500 --gpu-count 1 --gpu-type A100-40GB

# Evaluation
modal run run_modal.py::evaluate --env wordle --model Qwen/Qwen2.5-0.5B-Instruct --num-examples 50
```

### Method 2: Configuration Files (Recommended)

Train using YAML configuration files for reproducible experiments:
```bash
# Using config file (recommended approach)
modal run run_modal.py::train --config-path configs/modal_training_config.yaml

# Override specific parameters from config
modal run run_modal.py::train --config-path configs/modal_training_config_wordle.yaml --env vf-bfcl-single-turn

# Alternative entry point with dynamic resource allocation
modal run run_modal.py::train_with_config --config-path configs/modal_training_config.yaml
```

### Method 3: Mixed Configuration

Combine config files with parameter overrides:
```bash
# Use config as base, override environment
modal run run_modal.py::train --config-path configs/modal_training_config.yaml --env gsm8k

# Use config as base, override training steps
modal run run_modal.py::train --config-path configs/modal_training_config.yaml --steps 1000
```

## Technical Details

### GRPO Training with Embedded vLLM

The script automatically sets up embedded vLLM servers for GRPO (reinforcement learning) training:

1. **Automatic Detection**: Detects environments that need RL training (wordle, bfcl-single-turn, etc.)
2. **Model Selection**: Uses appropriate models based on environment:
   - `vf-bfcl-single-turn`: `Salesforce/xLAM-2-3b-fc-r` (function calling specialist)
   - `vf-wordle`: `Qwen/Qwen2.5-0.5B-Instruct` (general purpose)
   - Custom models via config files
3. **vLLM Configuration**: Optimized for training with:
   - GPU memory utilization: 50% (configurable)
   - Eager execution for better memory management
   - Tool calling support for function calling environments

### Environment Handling

The script automatically:
- Converts environment names (`vf-wordle` → `vf_wordle`)
- Installs verifier environments from the `environments/` directory
- Updates training scripts to use correct model names and configurations
- Sets up Python paths for environment discovery

### GPU Configuration

Default GPU allocation:
- **Single GPU**: `A100-40GB` (suitable for 3B model training)
- **Memory**: 32GB RAM, 8 CPUs
- **Timeout**: 4 hours (configurable)

Available GPU types in config files:
- `T4`: Development and small models
- `A10G`: Medium models
- `A100`, `A100-40GB`, `A100-80GB`: Large models and multi-GPU
- `H100`: Latest generation, best performance

## Persistent Storage

The script uses experiment-specific persistent volumes:

### Volume Naming
Volumes are automatically named using your environment variables:
- Format: `verifiers-{cache/outputs}-{EXPERIMENTER}-{EXPERIMENT_NAME}`
- Example: `verifiers-cache-alice-wordle-experiment`

### Mounted Paths
- `/cache`: Model weights, transformers cache, pip packages
- `/outputs`: Training outputs, checkpoints, logs

This ensures experiment isolation and prevents conflicts between different users or experiments.

## Output Management

Training outputs are automatically saved to the `outputs/` persistent volume:
```bash
# Outputs are saved to /outputs/ in the Modal container
# Access them in subsequent runs or download manually via Modal dashboard
```

## Troubleshooting

### Common Issues

1. **Environment Variables**: Ensure `.env` file is properly configured with required keys
2. **vLLM Server Issues**: The embedded vLLM server may take 60+ seconds to start
3. **Memory Issues**: Default config uses 50% GPU memory for vLLM, 50% for training
4. **Environment Installation**: Script automatically installs environments from `environments/` directory

### Debugging and Monitoring

Modal provides real-time logs with structured output:

```bash
# Check what's happening during training
modal run run_modal.py::train --config-path configs/modal_training_config.yaml

# Look for these log sections:
# 🚀 Starting Verifiers RL training with embedded vLLM
# 📦 Installing verifiers...
# 🌐 Starting vLLM server with {model}...
# 🏃 Starting GRPO training...
```

### vLLM Server Diagnostics

If training fails, check for:
- Model loading issues (wrong model name, insufficient memory)
- Port conflicts (script uses 127.0.0.1:8000)
- GPU memory allocation problems

The script includes automatic retries and detailed error reporting for vLLM issues.

## Cost Optimization

- **Timeout Management**: Default timeout is 4 hours (configurable in config files)
- **GPU Selection**: Use smaller GPUs (T4, A10G) for development, A100/H100 for production
- **Experiment Isolation**: Volume naming prevents conflicts, allows parallel experiments
- **Efficient Caching**: Persistent volumes cache models and dependencies between runs

## Advanced Configuration

### Custom Dependencies

Add Python packages to the Modal image by editing `run_modal.py`:
```python
.pip_install([
    "your-custom-package>=1.0.0",
    # Core packages already included: torch, transformers, wandb, etc.
])
```

### Multiple Concurrent Experiments

Each experimenter can run multiple experiments simultaneously:
```bash
# Terminal 1: Wordle experiment
EXPERIMENTER=alice EXPERIMENT_NAME=wordle-v1 modal run run_modal.py::train --config-path configs/modal_training_config_wordle.yaml

# Terminal 2: BFCL experiment (different volumes, no conflicts)
EXPERIMENTER=alice EXPERIMENT_NAME=bfcl-v1 modal run run_modal.py::train --config-path configs/modal_training_config_bfcl.yaml
```

### Custom Training Scripts

The script supports custom training scripts in the `examples/grpo/` directory:
- `train_wordle.py` - Wordle environment training
- `train_bfcl.py` - BFCL function calling training

Add new scripts by following the existing pattern and updating the `TRAINING_SCRIPTS` mapping in `run_modal.py:43-47`.

## Quick Reference

### Environment Variables Required
```bash
WANDB_API_KEY="your-wandb-key"
HF_TOKEN="your-huggingface-token" 
OPENAI_API_KEY="dummy-key-for-vllm"
EXPERIMENTER="your-name"
EXPERIMENT_NAME="experiment-description"  # Optional
```

### Common Commands
```bash
# Quick start with config
modal run run_modal.py::train --config-path configs/modal_training_config.yaml

# Direct parameters
modal run run_modal.py::train --env vf-wordle --steps 500

# Evaluation
modal run run_modal.py::evaluate --env wordle --model Qwen/Qwen2.5-0.5B-Instruct

# Override config parameters
modal run run_modal.py::train --config-path configs/modal_training_config.yaml --env vf-bfcl-single-turn
```

### Supported Environments
- `vf-wordle` - Wordle puzzle solving
- `vf-bfcl-single-turn` - Function calling (BFCL)
- `vf-tool-test` - Legacy function calling alias

### Available Models
- **General**: `Qwen/Qwen2.5-0.5B-Instruct`, `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-3B-Instruct`
- **Function Calling**: `Salesforce/xLAM-2-3b-fc-r`

## Support

- Modal documentation: https://modal.com/docs
- Verifiers repository: Current working directory
- Configuration examples: `configs/` directory