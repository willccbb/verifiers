# Modal Setup and Training Guide

This guide explains how to use Modal for training verifiers with GPU support.

## Prerequisites

1. **Modal Account**: Sign up at [modal.com](https://modal.com)
2. **Modal CLI**: Install with `pip install modal`
3. **Authentication**: Run `modal setup` and follow the prompts
4. **API Keys**: Set the following environment variables locally:
   ```bash
   export WANDB_API_KEY="your-wandb-key"
   export OPENAI_API_KEY="your-openai-key"  # Optional, for vLLM
   export HF_TOKEN="your-huggingface-token"  # Optional, for private models
   ```

## First-Time Setup

1. **Initial Sync**: Upload your local files to Modal's persistent storage:
   ```bash
   modal run run_modal.py --sync-only
   ```

2. **Verify Upload**: Check that files were uploaded correctly:
   ```bash
   modal run run_modal.py --list
   ```

3. **Install Environments**: Install any verifier environments you need:
   ```bash
   # Install a specific environment
   modal run run_modal.py --cmd "vf-install vf-wordle"
   
   # Install from repository
   modal run run_modal.py --cmd "vf-install vf-wordle --from-repo"
   
   # Install a custom environment
   modal run run_modal.py --cmd "vf-install vf-tool-test"
   ```

## Training Commands

### Basic Training

For standard training without vLLM:
```bash
# Single GPU training
modal run run_modal.py --cmd "python examples/your_script.py"

# Multi-GPU training with accelerate
modal run run_modal.py --cmd "accelerate launch --config-file configs/zero3.yaml examples/your_script.py"
```

### GRPO Training (with vLLM)

GRPO training requires a vLLM server. The script automatically detects and starts vLLM for:
- Any script in the `grpo/` directory
- Scripts containing `train_wordle.py` or `train_tool_test.py`
- Commands with `--use-vllm` flag
- Commands with `VLLM_MODEL=` environment variable

```bash
# Train Wordle with GRPO (uses default Wordle model)
modal run run_modal.py --cmd "python examples/grpo/train_wordle.py"

# Train tool test with GRPO (uses default ToolTest model)
modal run run_modal.py --cmd "accelerate launch --num-processes 1 --config-file configs/zero3.yaml examples/grpo/train_tool_test.py"

# Specify a custom model for vLLM
modal run run_modal.py --cmd "VLLM_MODEL=meta-llama/Llama-2-7b-hf python examples/grpo/train_custom.py"

# Force vLLM for any script
modal run run_modal.py --cmd "python my_script.py --use-vllm"
```

### vLLM Server Issues

If you see "Server is not up yet" errors, the script now provides detailed diagnostics:
- Checks if the vLLM process crashed
- Shows stdout/stderr from the server
- Provides suggestions for fixing common issues

Common fixes:
1. **Wrong model name**: Check the model exists on HuggingFace
2. **Out of memory**: Use a smaller model or reduce batch size
3. **Model download timeout**: Pre-download with:
   ```bash
   modal run run_modal.py --cmd "vf-vllm --model <model-name> --download-only"
   ```

## GPU Configuration

The default configuration uses 2x H100 GPUs. Modal automatically handles GPU allocation:
- GPU 0: Reserved for vLLM server (when needed)
- GPU 1: Used for training

To change GPU type, edit line 89 in `run_modal.py`:
```python
gpu="H100:2",  # Options: "A100:1", "A100:2", "H100:1", "H100:2", etc.
```

## Environment Variables

Pass custom environment variables using `--env-vars`:
```bash
modal run run_modal.py --cmd "python train.py" --env-vars "BATCH_SIZE=32,LEARNING_RATE=1e-4"
```

## Downloading Results

Download files or directories from Modal:
```bash
# Download a specific file
modal run run_modal.py --download "outputs/model.pt"

# Download a directory
modal run run_modal.py --download "outputs/"
```

## Troubleshooting

### Common Issues

1. **"No virtual environment found"**: This is handled automatically by the script
2. **vLLM server not starting**: Check logs for port conflicts or GPU allocation issues
3. **Out of memory**: Reduce batch size or use gradient accumulation
4. **Permission errors**: The script uses `/tmp` for caches to avoid permission issues

### Debugging Commands

```bash
# Check GPU availability
modal run run_modal.py --cmd "nvidia-smi"

# Check Python environment
modal run run_modal.py --cmd "which python && python --version"

# Check installed packages
modal run run_modal.py --cmd "pip list | grep verifiers"

# Test vLLM installation
modal run run_modal.py --cmd "python -c 'import vllm; print(vllm.__version__)'"
```

### Logs

Modal streams logs in real-time. For debugging:
- Check the "Environment Check" section for CUDA/PyTorch status
- Look for "vLLM server is ready!" when using GRPO
- Monitor GPU usage with the initial `nvidia-smi` output

## Cost Optimization

- Use `--timeout` to limit run time (default: 2 hours)
- Modal charges per second of GPU usage
- Stop runs early with Ctrl+C to save costs
- Use smaller GPUs (A10G, T4) for development

## Making vLLM Work with Any Training Script

The current setup works well for most cases, but here's how to ensure it works with ANY training script:

### Method 1: Use Environment Variables (Recommended)
```bash
# Basic usage - specify the model
VLLM_MODEL=meta-llama/Llama-2-7b-hf modal run run_modal.py --cmd "python my_training_script.py"

# With custom vLLM port (if your script expects different port)
VLLM_MODEL=gpt2 VLLM_PORT=8080 modal run run_modal.py --cmd "python train.py --api-url http://localhost:8080"

# Multiple environment variables
modal run run_modal.py --cmd "VLLM_MODEL=facebook/opt-1.3b python train.py" --env-vars "VLLM_ARGS=--max-model-len 2048"
```

### Method 2: Force vLLM with Flag
```bash
# Add --use-vllm to any command
modal run run_modal.py --cmd "python any_script.py --use-vllm"
```

### Method 3: Modify Detection Logic
If you have many scripts that need vLLM, edit `run_modal.py` line 210:
```python
needs_vllm = any([
    "grpo/" in cmd,
    "train_" in cmd,  # Any training script
    "--use-vllm" in cmd,
    "VLLM_MODEL=" in cmd,
    # Add your custom conditions here
    "your_pattern" in cmd,
])
```

### Common Pitfalls and Solutions

1. **Wrong Model Architecture**: Ensure your training script expects the same model architecture as what vLLM loads
   ```bash
   # Check model architecture first
   modal run run_modal.py --cmd "python -c 'from transformers import AutoModel; print(AutoModel.from_pretrained(\"your-model\").config)'"
   ```

2. **Port Conflicts**: If your script expects a different port than 8000:
   - Modify your script to use port 8000, OR
   - Edit `run_modal.py` to use a different port in the vllm_cmd

3. **Model Not Found**: Pre-download models to avoid timeout:
   ```bash
   modal run run_modal.py --cmd "huggingface-cli download your-model-name"
   ```

4. **Memory Issues**: Use smaller models or quantization:
   ```bash
   VLLM_MODEL=gpt2 modal run run_modal.py --cmd "python train.py"  # Start with small model
   ```

### Example: Custom Training Script

Here's a complete example for any custom training script:

```bash
# Step 1: Ensure your script uses the vLLM API endpoint
# In your script, use: api_url = "http://localhost:8000"

# Step 2: Run with Modal, specifying the model
VLLM_MODEL=meta-llama/Llama-2-7b-hf modal run run_modal.py --cmd \
  "python my_custom_rlhf_training.py --num-epochs 3 --batch-size 8"

# Step 3: If it fails, check the logs for the specific model path vLLM expects
# Then adjust your training script accordingly
```

## Advanced Usage

### Custom Image Modifications

To add more dependencies, edit the `image` definition in `run_modal.py`:
```python
.pip_install(["your-package>=1.0.0"])
```

### Persistent Storage

Files are saved to persistent volumes:
- `/workspace`: Your code and outputs
- `/cache`: Model weights and pip packages

These persist between runs to save download time.

### Multiple Concurrent Runs

Each Modal run is isolated. You can run multiple training jobs:
```bash
# Terminal 1
modal run run_modal.py --cmd "python train_model_a.py"

# Terminal 2  
modal run run_modal.py --cmd "python train_model_b.py"
```

## Support

- Modal documentation: https://modal.com/docs
- Verifiers documentation: https://github.com/willccbb/verifiers
- Report issues: Create an issue in the verifiers repository