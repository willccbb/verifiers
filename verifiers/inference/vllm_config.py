import os
import argparse
from dataclasses import dataclass, field
from typing import Optional, Literal, Union

@dataclass
class VLLMServerConfig:
    # Model configuration
    model_name_or_path: str = field(
        default="",
        metadata={"help": "Model name or path to load the model from."}
    )
    revision: Optional[str] = field(
        default=None,
        metadata={"help": "Revision to use for the model. If not specified, the default branch will be used."}
    )
    
    # Parallelization settings
    tensor_parallel_size: int = field(
        default_factory=lambda: os.environ.get("CUDA_VISIBLE_DEVICES", "").count(",") + 1,
        metadata={"help": "Number of tensor parallel workers to use."}
    )
    data_parallel_size: int = field(
        default=1,
        metadata={"help": "Number of data parallel workers to use."}
    )
    
    # Server settings
    host: str = field(
        default="0.0.0.0",
        metadata={"help": "Host address to run the server on."}
    )
    port: int = field(
        default=8000,
        metadata={"help": "Port to run the server on."}
    )
    
    # Memory and performance settings
    gpu_memory_utilization: float = field(
        default=0.95,
        metadata={"help": "Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache."}
    )
    dtype: str = field(
        default="auto",
        metadata={"help": "Data type to use for vLLM generation. If set to 'auto', the data type will be automatically determined."}
    )
    max_model_len: int = field(
        default=8192,
        metadata={"help": "The max_model_len to use for vLLM. This can be useful when running with reduced gpu_memory_utilization."}
    )
    kv_cache_dtype: str = field(
        default="auto",
        metadata={"help": "Data type to use for KV cache. If set to 'auto', the dtype will default to the model data type."}
    )
    
    # Feature flags
    enable_prefix_caching: bool = field(
        default=True,
        metadata={"help": "Whether to enable prefix caching in vLLM."}
    )
    enforce_eager: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether to enforce eager execution. If True, disable CUDA graph and always execute in eager mode."}
    )
    
    # Batching and request handling
    max_batch_size: int = field(
        default=1024,
        metadata={"help": "Maximum number of requests to process in one LLM call from the active pool."}
    )
    batch_request_timeout_seconds: int = field(
        default=300,
        metadata={"help": "Timeout in seconds for a single request waiting for its turn and completion."}
    )
    token_chunk_size: int = field(
        default=64,
        metadata={"help": "Number of tokens to generate per iteration per request in token-chunk dynamic batching."}
    )
    
    # Logging
    log_level: Literal["critical", "error", "warning", "info", "debug", "trace"] = field(
        default="info",
        metadata={"help": "Log level for uvicorn."}
    )

    def __post_init__(self):
        self.model = self.model_name_or_path