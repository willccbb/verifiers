#!/usr/bin/env python3
"""
Modal runner for verifiers training.

Usage:
    # Train with direct parameters
    modal run run_modal.py::train --env wordle --steps 200
    
    # Train with config file (recommended)
    modal run run_modal.py::train --config-path configs/modal_training_config.yaml
    modal run run_modal.py::train_with_config --config-path configs/modal_training_config.yaml
    
    # Train with mixed (config + override)
    modal run run_modal.py::train --config-path configs/modal_training_config.yaml --env gsm8k
    
    # Run evaluation
    modal run run_modal.py::evaluate --env wordle --model Qwen/Qwen2.5-0.5B-Instruct
"""

import modal
import os
import sys
import re
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Explicitly grab required keys from environment
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
WANDB_API_KEY = os.environ.get("WANDB_API_KEY") 
HF_TOKEN = os.environ.get("HF_TOKEN")

# Get experimenter and experiment names for app naming
experimenter = os.environ.get("EXPERIMENTER", "default")
experiment_name = os.environ.get("EXPERIMENT_NAME", "default")
app_name = f"verifiers_{experimenter}_{experiment_name}"
app = modal.App(app_name)

# Training scripts mapping
TRAINING_SCRIPTS = {
    "vf-wordle": "examples/grpo/train_wordle.py",
    "vf-tool-test": "examples/grpo/train_bfcl.py",  # Legacy alias
    "vf-bfcl-single-turn": "examples/grpo/train_bfcl.py",  # BFCL function calling training
}

# GPU configurations
GPU_CONFIGS = {
    1: "T4",
    2: "A10G:2",
    4: "A100-40GB:4",
    8: "H100:8",
}

# Build optimized image with local code
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .pip_install([
        # Core dependencies
        "torch>=2.3.0",
        "transformers>=4.44.0",
        "accelerate>=1.4.0",
        "datasets",
        "peft>=0.8.0",
        "wandb",
        "rich",
        "trl>=0.17.0",
        
        # Verifiers specific
        "openai",
        "nltk",
        "textarena",
        "python-dotenv",
        "liger-kernel>=0.5.10",
        "deepspeed",
        "einops",
        "sentencepiece",
        "protobuf",
        "vllm>=0.6.0",
    ])
    .add_local_dir(".", remote_path="/app/verifiers")
)

# Persistent volumes - experiment-specific
cache_volume = modal.Volume.from_name(f"verifiers-cache-{experimenter}-{experiment_name}", create_if_missing=True)
outputs_volume = modal.Volume.from_name(f"verifiers-outputs-{experimenter}-{experiment_name}", create_if_missing=True)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    # Handle both absolute and relative paths
    if not os.path.isabs(config_path):
        # If we're in the Modal container, adjust the path
        if os.path.exists("/app/verifiers"):
            config_path = os.path.join("/app/verifiers", config_path)
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def _train_with_vllm(env: str, gpu_type: str, steps: int, gpus: int, config: dict = None):
    """Training with embedded vLLM server for RL."""
    import subprocess
    import time
    
    print(f"🚀 Starting Verifiers RL training with embedded vLLM")
    print(f"📚 Environment: {env}")
    print(f"📏 GPU TYPE: {gpu_type}")
    print(f"🖥️ GPUs: {gpus}")
    
    # Set environment variables
    os.environ["HF_HOME"] = "/cache"
    os.environ["TRANSFORMERS_CACHE"] = "/cache"
    os.environ["WANDB_PROJECT"] = "verifiers"
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY or "dummy-key-for-vllm"
    
    # Change to the mounted verifiers repository
    print("\n📥 Setting up verifiers repository...")
    os.chdir("/app/verifiers")
    print(f"Current directory: {os.getcwd()}")
    print(f"Contents: {os.listdir('.')}")
    
    # Install verifiers
    print("📦 Installing verifiers...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
    
    # Install environment
    env_name = env.replace("-", "_")
    env_path = f"environments/{env_name}"
    print(f"🔍 Looking for environment: {env_name} at {env_path}")
    print(f"🔍 Environment exists: {os.path.exists(env_path)}")
    if os.path.exists(env_path):
        print(f"📦 Installing {env_name} environment...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", "-e", env_path], check=True, capture_output=True, text=True)
        print("✅ Environment installation completed")
        print(f"Install output: {result.stdout}")
    else:
        print(f"❌ Environment {env_name} not found at {env_path}")
        print("Available environments:")
        if os.path.exists("environments"):
            for env_dir in os.listdir("environments"):
                if os.path.isdir(f"environments/{env_dir}"):
                    print(f"  - {env_dir}")
        else:
            print("  No environments directory found")
    
    # Fix training scripts to use correct model names
    script_path = TRAINING_SCRIPTS.get(env, f"examples/grpo/train_{env}.py")
    if os.path.exists(script_path):
        with open(script_path, 'r') as f:
            content = f.read()
        
        modified = False
        
        # Fix environment ID (vf-wordle -> vf_wordle)
        if 'env_id="vf-' in content:
            content = re.sub(r'env_id="vf-([^"]+)"', r'env_id="vf_\1"', content)
            modified = True
        
        # Fix model loading with attn_implementation
        if 'model, tokenizer = vf.get_model_and_tokenizer(model_name)' in content and 'model_kwargs=' not in content:
            content = content.replace(
                'model, tokenizer = vf.get_model_and_tokenizer(model_name)',
                'model, tokenizer = vf.get_model_and_tokenizer(model_name, model_kwargs={"attn_implementation": "sdpa"})'
            )
            modified = True
        
        # Replace old model references
        if 'willcb/Qwen3' in content or 'model_name = f"willcb/Qwen3-{size}-Wordle"' in content:
            # Find and replace the model_name assignment
            content = re.sub(
                r'model_name = .*Qwen3.*',
                '''# Use Qwen2.5 models based on size
    if size == "1.7B":
        model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    elif size == "3B":
        model_name = "Qwen/Qwen2.5-3B-Instruct"
    elif size == "4B":
        model_name = "Qwen/Qwen2.5-3B-Instruct"
    else:
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"''',
                content
            )
            modified = True
        
        if modified:
            with open(script_path, 'w') as f:
                f.write(content)
    
    # Use model from config or default based on environment
    if config and 'model' in config:
        model_config = config['model']
        # Check for environment-specific overrides
        if 'overrides' in model_config and env in model_config['overrides']:
            model_name = model_config['overrides'][env]
        else:
            model_name = model_config.get('base_model', 'Qwen/Qwen2.5-0.5B-Instruct')
    else:
        # Fallback to original logic
        if env == "vf-bfcl-single-turn":
            model_name = "Salesforce/xLAM-2-3b-fc-r"
        else:
            model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    # Start standard vLLM OpenAI API server (no weight sync needed)
    print(f"\n🌐 Starting vLLM server with {model_name}...")
    
    # Get vLLM config from config file or use defaults
    vllm_config = config.get('vllm', {}) if config else {}
    
    vllm_cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--host", "127.0.0.1",
        "--port", "8000",
        "--disable-log-requests",
        "--trust-remote-code",
        "--max-model-len", str(vllm_config.get('max_model_len', 4096)),
        "--gpu-memory-utilization", str(vllm_config.get('gpu_memory_utilization', 0.5)),
        "--block-size", str(vllm_config.get('block_size', 16)),
    ]
    
    # Add optional flags based on config
    if vllm_config.get('enforce_eager', True):
        vllm_cmd.append("--enforce-eager")
    if vllm_config.get('enable_auto_tool_choice', True):
        vllm_cmd.append("--enable-auto-tool-choice")
    if 'tool_call_parser' in vllm_config:
        vllm_cmd.extend(["--tool-call-parser", vllm_config['tool_call_parser']])
    
    # Start vLLM in background
    vllm_process = subprocess.Popen(vllm_cmd)
    
    # Wait for vLLM to start
    print("⏳ Waiting for vLLM server to start...")
    time.sleep(60)  # Give vLLM more time to load model
    
    # Disable flash attention and optimize memory
    os.environ["TRANSFORMERS_DISABLE_FLASH_ATTN"] = "1"
    os.environ["ATTN_IMPLEMENTATION"] = "sdpa"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
    
    # Add environments to Python path for module discovery
    os.environ["PYTHONPATH"] = f"/app/verifiers/environments:{os.environ.get('PYTHONPATH', '')}"
    
    # Create configuration to use OpenAI client instead of VLLMClient
    vllm_config = {
        "use_openai_client": True,
        "base_url": "http://127.0.0.1:8000/v1",
        "api_key": "dummy-key"
    }
    
    # Save config to environment for the training script to use
    os.environ["VLLM_USE_OPENAI_CLIENT"] = "1"
    os.environ["VLLM_BASE_URL"] = vllm_config["base_url"]
    os.environ["VLLM_API_KEY"] = vllm_config["api_key"]
    
    print(f"\n🏃 Starting GRPO training...")
    print("==" * 30)
    
    try:
        # Run training script directly
        script = TRAINING_SCRIPTS.get(env, f"examples/grpo/train_{env}.py")
        train_cmd = [sys.executable, script]
        
        # Add steps if provided
        if steps:
            train_cmd.extend(["--steps", str(steps)])
        result = subprocess.run(train_cmd)
        
        print("==" * 30)
        print(f"✅ Training completed with exit code: {result.returncode}")
        
        # Save outputs
        if os.path.exists("outputs"):
            print("💾 Saving outputs...")
            subprocess.run(["cp", "-r", "outputs", "/outputs/"], check=False)
        
        return result.returncode == 0
        
    finally:
        # Kill vLLM server
        print("🛑 Stopping vLLM server...")
        vllm_process.terminate()
        vllm_process.wait()

# Simplified entry point - single GPU by default
@app.function(
    image=image,
    gpu="A100-40GB",  # Use A100 for 3B model training (40GB memory)
    cpu=8.0,
    memory=32768,
    timeout=14400,  # 4 hours for longer training runs
    volumes={
        "/cache": cache_volume,
        "/outputs": outputs_volume,
    },
    secrets=[modal.Secret.from_dotenv()],
)
def train(env: str = "wordle", gpu_count: int = 1, gpu_type: str = "T4", steps: int = 200, config_path: str = None):
    """Default training function - single GPU.
    
    Args:
        env: Environment name (e.g., 'wordle', 'gsm8k')
        gpu_count: Number of GPUs to use
        gpu_type: Type of GPU to use ('T4', 'A10G', 'A100', 'H100')
        steps: Number of training steps
        config_path: Path to configuration YAML file
    """
    
    config = None
    if config_path:
        print(f"📄 Loading configuration from {config_path}")
        config = load_config(config_path)
        
        # Override parameters with config values if not explicitly provided
        training_config = config.get('training', {})
        modal_config = config.get('modal', {})
        
        # Use config values if CLI args are defaults
        if env == "wordle" and 'environment' in training_config:
            env = training_config['environment']
        if steps == 200 and 'steps' in training_config:
            steps = training_config['steps']
        if gpu_count == 1 and 'gpu' in modal_config:
            gpu_count = modal_config['gpu'].get('count', 1)
        if gpu_type == "T4" and 'gpu' in modal_config:
            gpu_type = modal_config['gpu'].get('type', 'T4')
    
    print("🚀 Using RL (GRPO) training with embedded vLLM server")
    # For RL, use the original environment name
    return _train_with_vllm(env, gpu_type, steps, gpus=gpu_count, config=config)
    

@app.function(
    image=image,
    gpu="T4",
    timeout=600,
    volumes={"/cache": cache_volume},
    secrets=[modal.Secret.from_dotenv()],
)
def evaluate(env: str = "wordle", model: str = None, num_examples: int = 10):
    """Run verifiers evaluation on Modal."""
    import subprocess
    
    print(f"🧪 Running evaluation")
    print(f"📚 Environment: {env}")
    print(f"🤖 Model: {model or 'default'}")
    
    # Change to the mounted verifiers repository
    os.chdir("/app/verifiers")
    print(f"Current directory: {os.getcwd()}")
    print(f"Contents: {os.listdir('.')}")
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
    
    # Install environment
    # For SFT versions, use the base environment name
    env_name = env.replace("-", "_")
    env_path = f"environments/{env_name}"
    if os.path.exists(env_path):
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", env_path], check=True)
    
    # Build eval command
    cmd = [sys.executable, "-m", "verifiers.scripts.eval", env_name]
    if model:
        cmd.extend(["-m", model])
    cmd.extend(["-n", str(num_examples)])
    
    # Run evaluation
    print(f"\n🏃 Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    return result.returncode == 0

# Train with config file - dynamically allocates resources
@app.function(
    image=image,
    cpu=1.0,  # Minimal CPU for initial config loading
    timeout=60,
    secrets=[modal.Secret.from_dotenv()],
)
def train_with_config(config_path: str):
    """Load config and create a new Modal function with specified resources."""
    import subprocess
    
    print(f"📄 Loading configuration from {config_path}")
    config = load_config(config_path)
    
    # Extract Modal configuration
    modal_config = config.get('modal', {})
    gpu_config = modal_config.get('gpu', {})
    
    # Format GPU string
    gpu_count = gpu_config.get('count', 1)
    gpu_type = gpu_config.get('type', 'A100-40GB')
    if gpu_count == 1:
        gpu_str = gpu_type
    else:
        gpu_str = f"{gpu_type}:{gpu_count}"
    
    # Extract training parameters
    training_config = config.get('training', {})
    env = training_config.get('environment', 'wordle')
    steps = training_config.get('steps', 200)
    
    print(f"🚀 Launching training with config:")
    print(f"   Environment: {env}")
    print(f"   Steps: {steps}")
    print(f"   GPU: {gpu_str}")
    print(f"   CPUs: {modal_config.get('cpus', 8.0)}")
    print(f"   Memory: {modal_config.get('memory', 32768)} MB")
    
    # Create the actual training command with proper resource allocation
    cmd = [
        "modal", "run", "run_modal.py::train",
        "--env", env,
        "--steps", str(steps),
        "--gpu-count", str(gpu_count),
        "--gpu-type", gpu_type,
        "--config-path", config_path
    ]
    
    # Run the command
    result = subprocess.run(cmd)
    return result.returncode == 0

# Simple local entry point for testing
@app.local_entrypoint()
def main():
    """Local testing."""
    print("🎯 Modal runner for verifiers")
    print("\nUsage examples:")
    print("  # Using direct parameters:")
    print("  modal run run_modal.py::train --env wordle --steps 200")
    print("  ")
    print("  # Using config file:")
    print("  modal run run_modal.py::train --config-path configs/modal_training_config.yaml")
    print("  modal run run_modal.py::train_with_config --config-path configs/modal_training_config.yaml")
    print("  ")
    print("  # Evaluate:")
    print("  modal run run_modal.py::evaluate --env wordle")
    
if __name__ == "__main__":
    with app.run():
        main()
