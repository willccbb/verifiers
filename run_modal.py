#!/usr/bin/env python3
"""
Modal wrapper script for verifiers training with GPU support.
Usage: 
  modal run run_modal.py --cmd "python examples/grpo/train_wordle.py"
  modal run run_modal.py --cmd "vf-install vf-wordle --from-repo"
  modal run run_modal.py --cmd "vf-eval vf-wordle"
  modal run run_modal.py --download "path/to/file"
  modal run run_modal.py --sync-only
"""

import argparse
import modal
import sys
import os
import tarfile
import io
from pathlib import Path

app = modal.App("verifiers-training")

# Build image with all necessary dependencies
image = (
    # Use NVIDIA CUDA image with development tools
    modal.Image.from_registry("nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install([
        "git", "build-essential", "cmake", "wget", "curl", "unzip", 
        "ninja-build", "libssl-dev", "libffi-dev", "rustc", "cargo"
    ])
    # Install uv and make it available globally
    .run_commands(
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "mv /root/.local/bin/uv /usr/local/bin/uv",
        "mv /root/.local/bin/uvx /usr/local/bin/uvx",
        "chmod +x /usr/local/bin/uv /usr/local/bin/uvx"
    )
    # Install PyTorch with CUDA 12.1 support (version >=2.7.0 as required)
    .pip_install(
        "torch>=2.7.0",
        "torchvision",
        "torchaudio",
        extra_index_url="https://download.pytorch.org/whl/cu121"
    )
    # Install build dependencies for flash-attn compilation
    .pip_install(["packaging", "wheel", "setuptools", "ninja"])
    # Install flash-attn with proper compilation settings
    .run_commands(
        "MAX_JOBS=4 pip install flash-attn --no-build-isolation"
    )
    # Install verifiers package with training dependencies from GitHub
    .pip_install("verifiers[train] @ git+https://github.com/willccbb/verifiers.git")
    # Install additional dependencies that may not be covered
    .pip_install([
        "transformers>=4.44.0",
        "accelerate>=1.4.0", 
        "datasets",
        "peft",
        "wandb",
        "rich",
        "trl>=0.17.0",
        "openai",
        "pydantic>=2.11.7",
        "requests",
        "nest-asyncio>=1.6.0",
        "packaging",
        "huggingface-hub",
        "einops",
        "sentencepiece",
        "protobuf",
        "scipy",
        "scikit-learn",
        "vllm>=0.9.2",
        "ray>=2.9.0",
        "liger-kernel>=0.5.10",
        "deepspeed",
        "math-verify>=0.8.0",
        "nltk",
        "textarena",
    ])
)

# Create persistent volumes
storage_volume = modal.Volume.from_name("verifiers-storage", create_if_missing=True)
cache_volume = modal.Volume.from_name("verifiers-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",  # Can change to H100, A100, etc.
    timeout=7200,
    volumes={
        "/workspace": storage_volume,
        "/cache": cache_volume,  # Changed from /root/.cache
    },
)
def run_training_command(cmd: str, env_vars: dict = None):
    """Run training command with full environment setup."""
    import subprocess
    import shutil
    
    # Set up workspace
    workspace_dir = "/workspace/verifiers"
    work_dir = "/app/verifiers"
    
    # Copy from persistent storage if exists
    if os.path.exists(workspace_dir):
        print("Loading existing workspace from volume...")
        os.makedirs(os.path.dirname(work_dir), exist_ok=True)
        shutil.copytree(workspace_dir, work_dir, dirs_exist_ok=True)
    else:
        print("No existing workspace found, creating new one...")
        os.makedirs(work_dir, exist_ok=True)
    
    # Verify files exist
    train_file = os.path.join(work_dir, "examples/grpo/train_wordle.py")
    print(f"Checking for training file: {train_file}")
    print(f"File exists: {os.path.exists(train_file)}")
    
    if not os.path.exists(train_file):
        print("Training file not found! Listing workspace contents:")
        if os.path.exists(work_dir):
            for root, dirs, files in os.walk(work_dir):
                level = root.replace(work_dir, '').count(os.sep)
                indent = ' ' * 2 * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = ' ' * 2 * (level + 1)
                for file in files[:5]:  # Show first 5 files per directory
                    print(f"{subindent}{file}")
        else:
            print("Work directory doesn't exist!")
    
    # Ensure environment directories have __init__.py files
    environments_dir = os.path.join(work_dir, "environments")
    if os.path.exists(environments_dir):
        for env_dir in os.listdir(environments_dir):
            env_path = os.path.join(environments_dir, env_dir)
            if os.path.isdir(env_path):
                init_file = os.path.join(env_path, "__init__.py")
                if not os.path.exists(init_file):
                    print(f"Creating __init__.py for {env_dir}")
                    with open(init_file, 'w') as f:
                        f.write("# Auto-generated __init__.py for environment\n")
                        # Import the main module
                        f.write(f"from .{env_dir} import *\n")
    
    os.chdir(work_dir)
    
    # Set up environment
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{work_dir}:{work_dir}/environments"
    env['TRANSFORMERS_CACHE'] = '/cache/huggingface'
    env['HF_HOME'] = '/cache/huggingface'
    env['UV_CACHE_DIR'] = '/cache/uv'  # Also cache uv downloads
    
    # Add any custom env vars
    if env_vars:
        env.update(env_vars)
    
    # Set CUDA environment variables
    env['CUDA_VISIBLE_DEVICES'] = '0'  # Modal handles GPU allocation
    
    # Enable flash attention
    env['USE_FLASH_ATTENTION'] = 'true'
    
    # If no OPENAI_API_KEY is set, use a dummy one for vLLM
    if 'OPENAI_API_KEY' not in env:
        env['OPENAI_API_KEY'] = 'dummy-key-for-vllm'
    
    print("=== Environment Check ===")
    print(f"Working directory: {os.getcwd()}")
    subprocess.run(["nvidia-smi"], check=False)
    
    # Check PyTorch and CUDA
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA devices: {torch.cuda.device_count()}")
            print(f"Current device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"PyTorch check failed: {e}")
    
    # Check if verifiers is installed
    try:
        import verifiers
        print(f"Verifiers package found at: {verifiers.__file__}")
    except ImportError:
        print("WARNING: verifiers package not found!")
    
    print("\nDirectory structure:")
    subprocess.run(["find", ".", "-maxdepth", "3", "-type", "d", "-name", "__pycache__", "-prune", "-o", "-type", "d", "-print"], check=False)
    
    # Handle special commands
    if cmd.startswith("vf-"):
        # These are verifiers CLI commands
        print(f"\n=== Running Verifiers CLI Command ===")
        # Ensure verifiers CLI tools are in PATH
        if os.path.exists("scripts"):
            env['PATH'] = f"{work_dir}/scripts:{env['PATH']}"
    
    # Run the command
    try:
        print(f"\n=== Running Command ===")
        print(f"Command: {cmd}")
        
        # For multi-GPU commands, parse and adjust CUDA_VISIBLE_DEVICES
        if "CUDA_VISIBLE_DEVICES=" in cmd:
            # Modal handles GPU allocation, so we need to adjust this
            cmd = cmd.replace("CUDA_VISIBLE_DEVICES=0,1,2,3,4,5", "CUDA_VISIBLE_DEVICES=0")
            cmd = cmd.replace("CUDA_VISIBLE_DEVICES=6,7", "CUDA_VISIBLE_DEVICES=0")
            print(f"Adjusted command for Modal: {cmd}")
        
        result = subprocess.run(
            cmd,
            shell=True,
            text=True,
            env=env,
            timeout=7200
        )
        
        # Save workspace back to volume
        print("\n=== Saving Results ===")
        if os.path.exists(workspace_dir):
            shutil.rmtree(workspace_dir)
        shutil.copytree(work_dir, workspace_dir)
        storage_volume.commit()
        print("Workspace saved to persistent volume")
        
        return result.returncode
        
    except subprocess.TimeoutExpired:
        print("Command timed out after 2 hours")
        return 1
    except Exception as e:
        print(f"Error running command: {e}")
        import traceback
        traceback.print_exc()
        return 1


@app.function(
    image=image,
    volumes={"/workspace": storage_volume},
    timeout=300,
)
def sync_local_to_modal(files_data: bytes):
    """Sync local files to Modal storage."""
    import shutil
    
    print("Syncing local files to Modal...")
    
    # Extract files to temporary directory
    temp_dir = "/tmp/local_files"
    os.makedirs(temp_dir, exist_ok=True)
    
    with tarfile.open(fileobj=io.BytesIO(files_data), mode='r:gz') as tar:
        tar.extractall(temp_dir)
    
    # Copy to persistent storage
    workspace_dir = "/workspace/verifiers"
    if os.path.exists(workspace_dir):
        shutil.rmtree(workspace_dir)
    
    shutil.copytree(temp_dir, workspace_dir)
    storage_volume.commit()
    
    print(f"Synced files to workspace. Items: {len(os.listdir(workspace_dir))}")
    
    # List key directories
    for item in ['verifiers', 'environments', 'examples', 'configs']:
        path = os.path.join(workspace_dir, item)
        if os.path.exists(path):
            print(f"  ✓ {item}/")
    
    return True


@app.function(
    image=image,
    volumes={"/workspace": storage_volume},
    timeout=300,
)
def download_from_modal(file_path: str) -> bytes:
    """Download file from Modal storage."""
    workspace_dir = "/workspace/verifiers"
    full_path = os.path.join(workspace_dir, file_path)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if os.path.isfile(full_path):
        with open(full_path, 'rb') as f:
            return f.read()
    else:
        # Directory - create tar
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode='w:gz') as tar:
            tar.add(full_path, arcname=os.path.basename(full_path))
        return tar_buffer.getvalue()


@app.function(
    image=image,
    volumes={"/workspace": storage_volume},
    timeout=300,
)
def list_workspace():
    """List contents of Modal workspace."""
    workspace_dir = "/workspace/verifiers"
    
    print(f"Contents of {workspace_dir}:")
    if not os.path.exists(workspace_dir):
        print("  (empty)")
        return
    
    for root, dirs, files in os.walk(workspace_dir):
        level = root.replace(workspace_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:10]:  # Limit files shown
            print(f"{subindent}{file}")
        if len(files) > 10:
            print(f"{subindent}... and {len(files) - 10} more files")


def create_file_archive():
    """Create compressed archive of local files."""
    print("Creating archive of local files...")
    
    tar_buffer = io.BytesIO()
    file_count = 0
    total_size = 0
    
    # Files and directories to exclude
    exclude_patterns = {
        '.git', '__pycache__', '.pytest_cache', 'node_modules',
        '.venv', 'venv', 'wandb', '.modal', '.mypy_cache',
        '.DS_Store', 'outputs', 'checkpoints'
    }
    
    # File extensions to exclude
    exclude_extensions = {'.pyc', '.pyo', '.pyd', '.so', '.dylib', '.pt', '.bin'}
    
    with tarfile.open(fileobj=tar_buffer, mode='w:gz') as tar:
        for root, dirs, files in os.walk('.'):
            # Skip unwanted directories
            dirs[:] = [d for d in dirs if d not in exclude_patterns and not d.startswith('.')]
            
            for file in files:
                # Skip unwanted files
                if (file in exclude_patterns or 
                    file.startswith('.') or 
                    any(file.endswith(ext) for ext in exclude_extensions)):
                    continue
                
                file_path = os.path.join(root, file)
                try:
                    file_size = os.path.getsize(file_path)
                    # Skip very large files
                    if file_size > 100 * 1024 * 1024:  # 100MB
                        print(f"  Skipping large file: {file_path} ({file_size / 1024 / 1024:.1f} MB)")
                        continue
                    
                    arcname = file_path[2:] if file_path.startswith('./') else file_path
                    tar.add(file_path, arcname=arcname)
                    file_count += 1
                    total_size += file_size
                    
                    # Show important files being archived
                    if 'train_wordle.py' in file or file.endswith('.py') and file_count <= 10:
                        print(f"  Adding: {arcname}")
                        
                except Exception as e:
                    print(f"  Warning: Could not add {file_path}: {e}")
    
    data = tar_buffer.getvalue()
    compressed_size_mb = len(data) / 1024 / 1024
    total_size_mb = total_size / 1024 / 1024
    print(f"Archived {file_count} files ({total_size_mb:.1f} MB -> {compressed_size_mb:.1f} MB compressed)")
    return data


@app.local_entrypoint()
def main(cmd: str = None, download: str = None, sync_only: bool = False, list: bool = False, env_vars: str = None):
    """Main entry point."""
    
    # Parse environment variables
    parsed_env_vars = {}
    if env_vars:
        for var in env_vars.split(','):
            if '=' in var:
                key, value = var.split('=', 1)
                parsed_env_vars[key] = value
    
    # Handle list request
    if list:
        list_workspace.remote()
        return
    
    # Handle download request
    if download:
        print(f"Downloading: {download}")
        try:
            file_data = download_from_modal.remote(download)
            local_path = Path(download)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            if download.endswith('.tar.gz'):
                # Extract if it's a tar file
                with tarfile.open(fileobj=io.BytesIO(file_data), mode='r:gz') as tar:
                    tar.extractall('.')
                print(f"Extracted archive to current directory")
            else:
                with open(local_path, 'wb') as f:
                    f.write(file_data)
                print(f"Downloaded to: {local_path}")
        except Exception as e:
            print(f"Download failed: {e}")
            sys.exit(1)
        return
    
    # Sync local files to Modal
    file_data = create_file_archive()
    sync_local_to_modal.remote(file_data)
    
    if sync_only:
        print("Files synced to Modal.")
        return
    
    # Run command if provided
    if cmd:
        print(f"Running command on Modal: {cmd}")
        exit_code = run_training_command.remote(cmd, parsed_env_vars)
        sys.exit(exit_code)
    else:
        print("Files synced. Use --cmd to run a command.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modal wrapper for verifiers")
    parser.add_argument("--cmd", help="Command to run")
    parser.add_argument("--download", help="Download file/directory from Modal")
    parser.add_argument("--sync-only", action="store_true", help="Only sync files")
    parser.add_argument("--list", action="store_true", help="List workspace contents")
    parser.add_argument("--env-vars", help="Environment variables (comma-separated KEY=VALUE pairs)")
    
    args = parser.parse_args()
    main(cmd=args.cmd, download=args.download, sync_only=args.sync_only, list=args.list, env_vars=args.env_vars)