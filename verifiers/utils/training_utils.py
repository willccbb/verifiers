"""
Utility functions for training configuration and error handling.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def get_optimal_training_config(
    model_size: str = "default",
    gpu_count: int = 1,
    gpu_memory_gb: int = 24,
    expected_response_length: str = "medium"
) -> Dict[str, Any]:
    """
    Get recommended training configuration based on model size and hardware.
    
    Args:
        model_size: Size of the model ("small", "medium", "large", "xlarge")
        gpu_count: Number of GPUs available
        gpu_memory_gb: GPU memory in GB
        expected_response_length: Expected response length ("short", "medium", "long")
        
    Returns:
        Dictionary with recommended configuration values
    """
    # Base configuration
    config = {
        "max_concurrent": 128,
        "async_generation_timeout": 300.0,
        "vllm_server_timeout": 300.0,
        "max_tokens": 1024,
        "temperature": 0.7,
    }
    
    # Adjust based on model size
    if model_size == "small":
        config["max_concurrent"] = min(512, gpu_count * 256)
        config["max_tokens"] = 512
    elif model_size == "medium":
        config["max_concurrent"] = min(256, gpu_count * 128)
        config["max_tokens"] = 1024
    elif model_size == "large":
        config["max_concurrent"] = min(128, gpu_count * 64)
        config["max_tokens"] = 1024
    elif model_size == "xlarge":
        config["max_concurrent"] = min(64, gpu_count * 32)
        config["max_tokens"] = 512
    
    # Adjust based on expected response length
    if expected_response_length == "short":
        config["max_tokens"] = 512
        config["async_generation_timeout"] = 180.0
    elif expected_response_length == "long":
        config["max_tokens"] = 2048
        config["async_generation_timeout"] = 600.0
    
    # Adjust based on GPU memory
    if gpu_memory_gb < 16:
        config["max_concurrent"] = min(config["max_concurrent"], 64)
        config["max_tokens"] = min(config["max_tokens"], 512)
    elif gpu_memory_gb > 40:
        config["max_concurrent"] = min(config["max_concurrent"], 512)
    
    logger.info(f"Recommended configuration for {model_size} model with {gpu_count} GPUs: {config}")
    return config


def handle_training_timeout_error(error: Exception) -> Dict[str, Any]:
    """
    Provide guidance when encountering timeout errors during training.
    
    Args:
        error: The timeout exception that occurred
        
    Returns:
        Dictionary with suggested fixes
    """
    suggestions = {
        "reduce_concurrency": "Try reducing max_concurrent parameter",
        "increase_timeout": "Increase async_generation_timeout in GRPOConfig",
        "check_server": "Verify vLLM server is running and responsive",
        "adjust_model_params": "Consider reducing max_tokens or using a smaller model",
        "system_tuning": "Increase system limits with 'ulimit -n 4096'"
    }
    
    logger.warning(f"Training timeout error: {error}")
    logger.info("Suggested fixes:")
    for key, suggestion in suggestions.items():
        logger.info(f"  - {suggestion}")
        
    return suggestions


def validate_training_config(config: Dict[str, Any]) -> bool:
    """
    Validate training configuration for common issues.
    
    Args:
        config: Training configuration dictionary
        
    Returns:
        True if configuration appears valid, False otherwise
    """
    issues = []
    
    # Check max_concurrent
    max_concurrent = config.get("max_concurrent", 128)
    if max_concurrent > 1024:
        issues.append("max_concurrent is very high (>1024), consider reducing")
    
    # Check timeouts
    timeout = config.get("async_generation_timeout", 300.0)
    if timeout < 60.0:
        issues.append("async_generation_timeout is very low (<60s), consider increasing")
        
    # Check max_tokens
    max_tokens = config.get("max_tokens", 1024)
    if max_tokens > 4096:
        issues.append("max_tokens is very high (>4096), may cause timeout issues")
        
    if issues:
        logger.warning("Potential configuration issues found:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        return False
    
    logger.info("Configuration validation passed")
    return True
    