#!/usr/bin/env python3
"""
Modal runner for verifiers training.

Usage:
    # Train Wordle environment
    modal run run_modal.py::train --env wordle --size 0.5B
    
    # Train GSM8K environment  
    modal run run_modal.py::train --env gsm8k
    
    # Train with custom config
    modal run run_modal.py::train --env math-python --gpus 2 --steps 500
    
    # Run evaluation
    modal run run_modal.py::evaluate --env wordle --model Qwen/Qwen2.5-0.5B-Instruct
"""

import modal
import os
import sys
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

app = modal.App("verifiers")

# Training scripts mapping
TRAINING_SCRIPTS = {
    "wordle": "examples/grpo/train_wordle.py",
    "wordle-sft": "examples/sft/train_wordle_sft.py",  # SFT version without vLLM
    "tool-test": "examples/grpo/train_tool_test.py",
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

# Persistent volumes
cache_volume = modal.Volume.from_name("verifiers-cache", create_if_missing=True)
outputs_volume = modal.Volume.from_name("verifiers-outputs", create_if_missing=True)

# Single GPU training function
@app.function(
    image=image,
    gpu="T4",
    cpu=8.0,
    memory=32768,
    timeout=14400,  # Increase timeout to 4 hours
    volumes={
        "/cache": cache_volume,
        "/outputs": outputs_volume,
    },
    secrets=[modal.Secret.from_dotenv()],
)
def train_1gpu(env: str, size: str = "0.5B", steps: int = 200, batch_size: int = None, 
               learning_rate: float = None, wandb_project: str = None, wandb_run_name: str = None):
    """Single GPU training."""
    return _train(env, size, steps, batch_size, learning_rate, wandb_project, wandb_run_name, gpus=1)

# 2 GPU training function
@app.function(
    image=image,
    gpu="A10G:2",
    cpu=8.0,
    memory=32768,
    timeout=14400,  # Increase timeout to 4 hours
    volumes={
        "/cache": cache_volume,
        "/outputs": outputs_volume,
    },
    secrets=[modal.Secret.from_dotenv()],
)
def train_2gpu(env: str, size: str = "0.5B", steps: int = 200, batch_size: int = None,
               learning_rate: float = None, wandb_project: str = None, wandb_run_name: str = None):
    """2 GPU training."""
    return _train(env, size, steps, batch_size, learning_rate, wandb_project, wandb_run_name, gpus=2)

# 4 GPU training function
@app.function(
    image=image,
    gpu="A100-40GB:4",
    cpu=8.0,
    memory=32768,
    timeout=14400,  # Increase timeout to 4 hours
    volumes={
        "/cache": cache_volume,
        "/outputs": outputs_volume,
    },
    secrets=[modal.Secret.from_dotenv()],
)
def train_4gpu(env: str, size: str = "1.7B", steps: int = 200, batch_size: int = None,
               learning_rate: float = None, wandb_project: str = None, wandb_run_name: str = None):
    """4 GPU training."""
    return _train(env, size, steps, batch_size, learning_rate, wandb_project, wandb_run_name, gpus=4)

# 8 GPU training function
@app.function(
    image=image,
    gpu="H100:8",
    cpu=8.0,
    memory=32768,
    timeout=14400,  # Increase timeout to 4 hours
    volumes={
        "/cache": cache_volume,
        "/outputs": outputs_volume,
    },
    secrets=[modal.Secret.from_dotenv()],
)
def train_8gpu(env: str, size: str = "4B", steps: int = 200, batch_size: int = None,
               learning_rate: float = None, wandb_project: str = None, wandb_run_name: str = None):
    """8 GPU training."""
    return _train(env, size, steps, batch_size, learning_rate, wandb_project, wandb_run_name, gpus=8)

def _train(env: str, size: str, steps: int, batch_size: int, learning_rate: float, 
           wandb_project: str, wandb_run_name: str, gpus: int):
    """Shared training logic."""
    import subprocess
    import torch
    
    print(f"🚀 Starting Verifiers training")
    print(f"📚 Environment: {env}")
    print(f"📏 Model size: {size}")
    print(f"🖥️ GPUs: {gpus}")
    
    # Set environment variables
    os.environ["HF_HOME"] = "/cache"
    os.environ["TRANSFORMERS_CACHE"] = "/cache"
    os.environ["WANDB_PROJECT"] = wandb_project or "verifiers"
    if wandb_run_name:
        os.environ["WANDB_RUN_NAME"] = wandb_run_name
    
    # Log GPU info
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Change to the mounted verifiers repository
    print("\n📥 Setting up verifiers repository...")
    os.chdir("/app/verifiers")
    print(f"Current directory: {os.getcwd()}")
    print(f"Contents: {os.listdir('.')}")
    
    # Install verifiers
    print("📦 Installing verifiers...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
    
    # Create SFT training script if it doesn't exist
    sft_dir = "examples/sft"
    os.makedirs(sft_dir, exist_ok=True)
    sft_script_path = "examples/sft/train_wordle_sft.py"
    if not os.path.exists(sft_script_path):
        with open(sft_script_path, 'w') as f:
            f.write('''import argparse
import verifiers as vf
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
import torch
import os

def main(args):
    size = args.size
    # Use Qwen models with proper HF authentication
    if size == "1.7B":
        model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    elif size == "4B":
        model_name = "Qwen/Qwen2.5-3B-Instruct"
    else:
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    print(f"Loading model: {model_name}")
    model, tokenizer = vf.get_model_and_tokenizer(model_name, model_kwargs={"attn_implementation": "sdpa"})
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load wordle environment to get dataset
    vf_env = vf.load_environment(env_id="vf_wordle")
    
    # Get train dataset
    dataset = vf_env.dataset
    print(f"Dataset size: {len(dataset)}")
    
    # Tokenize dataset
    def tokenize_function(examples):
        # Format the prompts
        texts = []
        for i in range(len(examples["prompt"])):
            messages = examples["prompt"][i]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            text += f"Let me think about this Wordle puzzle...\\n\\nAnswer: {examples['answer'][i]}"
            texts.append(text)
        
        # Tokenize
        model_inputs = tokenizer(
            texts,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors=None,
        )
        
        # Set labels (same as input_ids for causal LM)
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        
        return model_inputs
    
    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
    )
    
    # Training arguments
    run_name = f"wordle-sft-{size}"
    training_args = TrainingArguments(
        output_dir=f"./outputs/{run_name}",
        run_name=run_name,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        logging_steps=1,
        save_steps=50,
        eval_strategy="no",
        save_strategy="steps",
        warmup_steps=10,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        remove_unused_columns=True,
        max_steps=int(os.environ.get("VERIFIERS_MAX_STEPS", "20")),
        logging_dir=f"./logs/{run_name}",
    )
    
    # Create standard Trainer (not SFTTrainer to avoid compatibility issues)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save model and tokenizer
    print("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    print(f"Model saved to {training_args.output_dir}")
    
    # Copy outputs to Modal volume if available
    if os.path.exists("/outputs"):
        import shutil
        shutil.copytree(training_args.output_dir, f"/outputs/{run_name}", dirs_exist_ok=True)
        print(f"Model copied to Modal volume: /outputs/{run_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", "-s", type=str, default="0.5B")
    args = parser.parse_args()
    main(args)
''')
    
    # Fix training scripts to use correct model names and environment IDs
    for _, script_path in TRAINING_SCRIPTS.items():
        if os.path.exists(script_path):
            with open(script_path, 'r') as f:
                content = f.read()
            
            modified = False
            
            # Fix environment ID (vf-wordle -> vf_wordle)
            if 'env_id="vf-' in content:
                content = re.sub(r'env_id="vf-([^"]+)"', r'env_id="vf_\1"', content)
                modified = True
            
            # Fix old model references
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
    
    # Install environment
    # For SFT versions, use the base environment name
    base_env = env.replace("-sft", "")
    env_name = f"vf_{base_env.replace('-', '_')}"
    env_path = f"environments/{env_name}"
    if os.path.exists(env_path):
        print(f"📦 Installing {env_name} environment...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", env_path], check=True)
    else:
        print(f"❌ Environment {env_name} not found at {env_path}")
        print("Available environments:")
        if os.path.exists("environments"):
            for env_dir in os.listdir("environments"):
                if os.path.isdir(f"environments/{env_dir}"):
                    print(f"  - {env_dir}")
        else:
            print("  No environments directory found")
    
    # Build training command
    script = TRAINING_SCRIPTS.get(env)
    if not script:
        raise ValueError(f"Unknown environment: {env}")
    
    # Configure based on GPU count
    if gpus > 1:
        # Multi-GPU with accelerate
        cmd = [
            "accelerate", "launch",
            "--num-processes", str(gpus),
            "--config-file", "configs/zero3.yaml",
            script
        ]
    else:
        # Single GPU
        cmd = [sys.executable, script]
    
    # Add arguments
    cmd.extend(["--size", size])
    
    # Override training args if provided
    if steps:
        os.environ["VERIFIERS_MAX_STEPS"] = str(steps)
        # Also update the training script to use the max_steps
        if script and os.path.exists(script):
            with open(script, 'r') as f:
                content = f.read()
            # Replace max_steps value
            content = re.sub(r'training_args\.max_steps = \d+', f'training_args.max_steps = {steps}', content)
            with open(script, 'w') as f:
                f.write(content)
    if batch_size:
        os.environ["VERIFIERS_BATCH_SIZE"] = str(batch_size)
    if learning_rate:
        os.environ["VERIFIERS_LR"] = str(learning_rate)
    
    # Disable flash attention if needed
    os.environ["TRANSFORMERS_DISABLE_FLASH_ATTN"] = "1"
    os.environ["ATTN_IMPLEMENTATION"] = "sdpa"
    os.environ["FLASH_ATTENTION_SKIP_IMPORT"] = "1"
    
    # Add attn_implementation to model loading
    if script and os.path.exists(script):
        with open(script, 'r') as f:
            content = f.read()
        # Add attn_implementation parameter
        content = content.replace(
            'model, tokenizer = vf.get_model_and_tokenizer(model_name)',
            'model, tokenizer = vf.get_model_and_tokenizer(model_name, model_kwargs={"attn_implementation": "sdpa"})'
        )
        with open(script, 'w') as f:
            f.write(content)
    
    # Run training
    print(f"\n🏃 Running: {' '.join(cmd)}")
    print("=" * 60)
    
    result = subprocess.run(cmd)
    
    print("=" * 60)
    print(f"✅ Training completed with exit code: {result.returncode}")
    
    # Save outputs
    if os.path.exists("outputs"):
        print("💾 Saving outputs...")
        subprocess.run(["cp", "-r", "outputs", "/outputs/"], check=False)
    
    return result.returncode == 0

def _train_with_vllm(env: str, size: str, steps: int, gpus: int):
    """Training with embedded vLLM server for RL."""
    import subprocess
    import time
    
    print(f"🚀 Starting Verifiers RL training with embedded vLLM")
    print(f"📚 Environment: {env}")
    print(f"📏 Model size: {size}")
    print(f"🖥️ GPUs: {gpus}")
    
    # Set environment variables
    os.environ["HF_HOME"] = "/cache"
    os.environ["TRANSFORMERS_CACHE"] = "/cache"
    os.environ["WANDB_PROJECT"] = "verifiers"
    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "dummy-key-for-vllm")
    
    # Change to the mounted verifiers repository
    print("\n📥 Setting up verifiers repository...")
    os.chdir("/app/verifiers")
    print(f"Current directory: {os.getcwd()}")
    print(f"Contents: {os.listdir('.')}")
    
    # Install verifiers
    print("📦 Installing verifiers...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
    
    # Install environment
    base_env = env.replace("-sft", "")
    env_name = f"vf_{base_env.replace('-', '_')}"
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
    
    # Use xLAM model for function calling tasks
    if env == "bfcl-single-turn":
        model_name = "Salesforce/xLAM-2-3b-fc-r"
    else:
        # Model name mapping for other environments
        if size == "1.7B":
            model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        elif size == "3B":
            model_name = "Qwen/Qwen2.5-3B-Instruct"
        elif size == "4B":
            model_name = "Qwen/Qwen2.5-3B-Instruct"
        else:
            model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    # Start standard vLLM OpenAI API server (no weight sync needed)
    print(f"\n🌐 Starting vLLM server with {model_name}...")
    vllm_cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--host", "127.0.0.1",
        "--port", "8000",
        "--enforce-eager",
        "--disable-log-requests",
        "--trust-remote-code",
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.5",  # Increased for 3B model
        "--block-size", "16",
        "--enable-auto-tool-choice",  # Required for tool calling
        "--tool-call-parser", "hermes",  # Parser for tool calls
    ]
    
    # Start vLLM in background
    vllm_process = subprocess.Popen(vllm_cmd)
    
    # Wait for vLLM to start
    print("⏳ Waiting for vLLM server to start...")
    time.sleep(30)  # Give vLLM time to load model
    
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
        train_cmd = [sys.executable, script, "--size", size]
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
def train(env: str = "wordle", size: str = "0.5B", steps: int = 200, rl: bool = False):
    """Default training function - single GPU.
    
    Args:
        env: Environment name (e.g., 'wordle', 'gsm8k')
        size: Model size ('0.5B', '1.7B', '4B')
        steps: Number of training steps
        rl: Use RL (GRPO) training. If False, uses SFT training.
    """
    if rl:
        print("🚀 Using RL (GRPO) training with embedded vLLM server")
        # For RL, use the original environment name
        return _train_with_vllm(env, size, steps, gpus=1)
    else:
        print("📚 Using SFT (Supervised Fine-Tuning)")
        # For SFT, add -sft suffix if not present
        if not env.endswith("-sft"):
            env = f"{env}-sft"
        return _train(env, size, steps, None, None, None, None, gpus=1)

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
    base_env = env.replace("-sft", "")
    env_name = f"vf_{base_env.replace('-', '_')}"
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

# Simple local entry point for testing
@app.local_entrypoint()
def main():
    """Local testing."""
    print("🎯 Modal runner for verifiers")
    print("\nUsage examples:")
    print("  modal run run_modal.py::train --env wordle --size 0.5B")
    print("  modal run run_modal.py::train_2gpu --env gsm8k --size 1.7B")
    print("  modal run run_modal.py::evaluate --env wordle")
    
if __name__ == "__main__":
    with app.run():
        main()
