"""
OpenAI-compatible vLLM server with weight synchronization.

Usage:

```bash
uv run python vllm_serve.py --model <model_name> --port <port>
```

Supports:
- /v1/chat/completions
- /v1/completions
"""
import argparse
import logging
import os
import time 
import asyncio 
import threading 
import inspect 
from collections.abc import Sequence
from collections import defaultdict 
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any, Literal, Optional
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection as MPConnection
from typing import Any as AnyType
from uuid import uuid4

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse 
import torch
from trl import TrlParser
from pydantic import BaseModel
import uvicorn
from vllm import LLM, SamplingParams
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.parallel_state import get_world_group
from vllm.distributed.utils import StatelessProcessGroup
from vllm.sampling_params import GuidedDecodingParams
from vllm.utils import get_open_port


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Ensure logger is defined

# We use CUDA with multiprocessing, so we must use the 'spawn' start method. Otherwise, we will get the following
# error: RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use
# the 'spawn' start method
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# At the global level, after imports and logger setup:
pipe_lock = threading.Lock()  # Global lock for pipe operations
request_queue: Optional[asyncio.Queue] = None
batch_processor_task: Optional[asyncio.Task] = None

# Generation tracking
active_generation_count = 0
generation_count_lock = asyncio.Lock()

# Weight update throttling
MAX_CONCURRENT_WEIGHT_UPDATES = 5
weight_update_semaphore = asyncio.Semaphore(MAX_CONCURRENT_WEIGHT_UPDATES)

# -------- OpenAI /v1/chat/completions Pydantic Models ---------- #
class OAChatMessage(BaseModel):
    role: str
    content: str

class OAChatCompletionRequest(BaseModel):
    model: str
    messages: list[OAChatMessage]
    temperature: float | None = 0.7
    top_p: float | None = 1.0
    presence_penalty: float | None = 0.0
    frequency_penalty: float | None = 0.0
    max_tokens: int | None  = 1024
    n: int | None = 1
    stop: str | list[str] | None = None
    stream: bool = False # not supported
    extra_body: dict | None = None 
    # supported by vLLM:
    # guided_decoding, include_stop_str_in_output, skip_special_tokens, spaces_between_special_tokens

class OAChatChoice(BaseModel):
    index: int
    message: OAChatMessage
    finish_reason: str | None = "stop"

class OAChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[OAChatChoice]

# -------- OpenAI /v1/completions Pydantic Models ---------- #
class OACompletionRequest(BaseModel):
    model: str
    prompt: str | list[str] 
    temperature: float | None = 0.7
    top_p: float | None = 1.0
    presence_penalty: float | None = 0.0
    frequency_penalty: float | None = 0.0
    max_tokens: int | None  = 1024
    n: int  = 1
    stop: str | list[str] | None = None
    stream: bool = False # not supported
    extra_body: dict | None = None

class OACompletionChoice(BaseModel):
    index: int
    text: str
    logprobs: dict | None = None 
    finish_reason: str | None = "length" 

class OACompletionResponse(BaseModel):
    id: str
    object: str = "completion"
    created: int
    model: str
    choices: list[OACompletionChoice]
# ---------------------------------------------------------------------- #

def send_and_recv(conn: MPConnection, payload: dict):
    """Helper to send a payload and receive a response over a pipe."""
    # Use the global pipe_lock
    with pipe_lock:
        conn.send(payload)
        return conn.recv()

class WeightSyncWorkerExtension:
    """
    A vLLM worker extension that enables weight synchronization between a client and multiple server workers.

    This worker uses a `StatelessProcessGroup` to establish communication and a `PyNcclCommunicator` to handle
    efficient GPU-based communication using NCCL. The primary purpose of this class is to receive updated model weights
    from a client process and distribute them to all worker processes participating in model inference.
    """

    # The following attributes are initialized when `init_communicator` method is called.
    pynccl_comm = None  # Communicator for weight updates
    client_rank = None  # Source rank for broadcasting updated weights

    def init_communicator(self, host: str, port: int, world_size: int) -> None:
        """
        Initializes the weight update communicator using a stateless process group.

        This method creates a `StatelessProcessGroup` that allows external training processes to
        communicate with vLLM workers without interfering with the global torch distributed group.

        Args:
            host (`str`):
                Hostname or IP address of the master node.
            port (`int`):
                Port number to be used for communication.
            world_size (`int`):
                Total number of participating processes in the update group.
        """
        if self.pynccl_comm is not None:
            raise RuntimeError("Weight update group already initialized. Call close_communicator first.")

        # Get the rank of the current worker in the global world group.
        rank = get_world_group().rank
        
        # Log device information for debugging
        logger.info(f"[WORKER] Initializing communicator: rank={rank}, device={self.device}, world_size={world_size}") # type: ignore

        # Create a stateless process group to manage communication between training processes and vLLM workers.
        pg = StatelessProcessGroup.create(host=host, port=port, rank=rank, world_size=world_size)

        # Initialize the NCCL-based communicator for weight synchronization.
        self.pynccl_comm = PyNcclCommunicator(pg, device=self.device) # type: ignore

        # The client process that sends updated weights has the highest rank (world_size - 1).
        self.client_rank = world_size - 1

    def update_named_param(self, name: str, dtype: torch.dtype, shape: Sequence[int]) -> None:
        """
        Receives updated weights from the client process and updates the named parameter in the model.

        Args:
            name (`str`):
                Name of the weight tensor being updated.
            dtype (`torch.dtype`):
                Data type of the weight tensor (e.g., `torch.float32`).
            shape (`Sequence[int]`):
                Shape of the weight tensor.
        """
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"[WORKER] Received weight update request for {name}, dtype={dtype}, shape={shape}")
        
        if self.pynccl_comm is None:
            raise RuntimeError("Communicator not initialized. Call `init_communicator` first.")

        # Allocate memory for the incoming weight tensor on the correct device.
        weight = torch.empty(shape, dtype=dtype, device=self.device) # type: ignore

        logger.debug(f"[WORKER] Starting NCCL broadcast for {name}")
        # Use NCCL to broadcast the updated weights from the client (src) to all workers.
        self.pynccl_comm.broadcast(weight, src=self.client_rank) # type: ignore 
        logger.debug(f"[WORKER] NCCL broadcast complete, waiting at barrier for {name}")
        self.pynccl_comm.group.barrier()
        logger.debug(f"[WORKER] Barrier passed, loading weights for {name}")

        # Load the received weights into the model.
        self.model_runner.model.load_weights(weights=[(name, weight)]) # type: ignore
        logger.debug(f"[WORKER] Weight update complete for {name}")

    def close_communicator(self) -> None:
        """
        Closes the communicator when weight synchronization is no longer needed.

        This method deletes the NCCL communicator to release associated resources.
        """

        if self.pynccl_comm is not None:
            del self.pynccl_comm
            self.pynccl_comm = None  # Ensure attribute is reset to None
            self.client_rank = None  # Ensure attribute is reset to None


@dataclass
class ScriptArguments:
    r"""
    Arguments for the script.

    Args:
        model (`str`):
            Model name or path to load the model from.
        revision (`str` or `None`, *optional*, defaults to `None`):
            Revision to use for the model. If not specified, the default branch will be used.
        tensor_parallel_size (`int`, *optional*, defaults to `1`):
            Number of tensor parallel workers to use.
        data_parallel_size (`int`, *optional*, defaults to `1`):
            Number of data parallel workers to use.
        host (`str`, *optional*, defaults to `"0.0.0.0"`):
            Host address to run the server on.
        port (`int`, *optional*, defaults to `8000`):
            Port to run the server on.
        gpu_memory_utilization (`float`, *optional*, defaults to `0.95`):
            Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache on the
            device dedicated to generation powered by vLLM. Higher values will increase the KV cache size and thus
            improve the model's throughput. However, if the value is too high, it may cause out-of-memory (OOM) errors
            during initialization.
        dtype (`str`, *optional*, defaults to `"auto"`):
            Data type to use for vLLM generation. If set to `"auto"`, the data type will be automatically determined
            based on the model configuration. Find the supported values in the vLLM documentation.
        max_model_len (`int` or `None`, *optional*, defaults to `None`):
            If set, the `max_model_len` to use for vLLM. This can be useful when running with reduced
            `vllm_gpu_memory_utilization`, leading to a reduced KV cache size. If not set, vLLM will use the model
            context size, which might be much larger than the KV cache, leading to inefficiencies.
        enable_prefix_caching (`bool` or `None`, *optional*, defaults to `None`):
            Whether to enable prefix caching in vLLM. If set to `True`, ensure that the model and the hardware support
            this feature.
        enforce_eager (`bool` or `None`, *optional*, defaults to `None`):
            Whether to enforce eager execution. If set to `True`, we will disable CUDA graph and always execute the
            model in eager mode. If `False` (default behavior), we will use CUDA graph and eager execution in hybrid.
        kv_cache_dtype (`str`, *optional*, defaults to `"auto"`):
            Data type to use for KV cache. If set to `"auto"`, the dtype will default to the model data type.
        log_level (`str`, *optional*, defaults to `"info"`):
            Log level for uvicorn. Possible choices: `"critical"`, `"error"`, `"warning"`, `"info"`, `"debug"`,
            `"trace"`.
        max_batch_size (int):
            Maximum number of requests to process in one LLM call from the active pool.
        batch_request_timeout_seconds (int):
            Timeout in seconds for a single request waiting for its turn and completion.
        token_chunk_size (int):
            Number of tokens to generate per iteration per request in token-chunk dynamic batching.
        """

    model: str = field(metadata={"help": "Model name or path to load the model from."})
    revision: Optional[str] = field(
        default=None,
        metadata={"help": "Revision to use for the model. If not specified, the default branch will be used."},
    )
    tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Number of tensor parallel workers to use."},
    )
    data_parallel_size: int = field(
        default=1,
        metadata={"help": "Number of data parallel workers to use."},
    )
    host: str = field(
        default="0.0.0.0",
        metadata={"help": "Host address to run the server on."},
    )
    port: int = field(
        default=8000,
        metadata={"help": "Port to run the server on."},
    )
    gpu_memory_utilization: float = field(
        default=0.95,
        metadata={
            "help": "Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV "
            "cache on the device dedicated to generation powered by vLLM. Higher values will increase the KV cache "
            "size and thus improve the model's throughput. However, if the value is too high, it may cause "
            "out-of-memory (OOM) errors during initialization."
        },
    )
    dtype: str = field(
        default="auto",
        metadata={
            "help": "Data type to use for vLLM generation. If set to 'auto', the data type will be automatically "
            "determined based on the model configuration. Find the supported values in the vLLM documentation."
        },
    )
    max_model_len: Optional[int] = field(
        default=None,
        metadata={
            "help": "If set, the `max_model_len` to use for vLLM. This can be useful when running with reduced "
            "`vllm_gpu_memory_utilization`, leading to a reduced KV cache size. If not set, vLLM will use the model "
            "context size, which might be much larger than the KV cache, leading to inefficiencies."
        },
    )
    enable_prefix_caching: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether to enable prefix caching in vLLM. If set to `True`, ensure that the model and the "
            "hardware support this feature."
        },
    )
    enforce_eager: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to enforce eager execution. If set to `True`, we will disable CUDA graph and always "
            "execute the model in eager mode. If `False` (default behavior), we will use CUDA graph and eager "
            "execution in hybrid."
        },
    )
    kv_cache_dtype: str = field(
        default="auto",
        metadata={
            "help": "Data type to use for KV cache. If set to 'auto', the dtype will default to the model data type."
        },
    )
    log_level: str = field(
        default="info",
        metadata={
            "help": "Log level for uvicorn. Possible choices: 'critical', 'error', 'warning', 'info', 'debug', "
            "'trace'."
        },
    )
    max_batch_size: int = field(
        default=32,
        metadata={"help": "Maximum number of requests to process in one LLM call from the active pool."},
    )
    batch_request_timeout_seconds: int = field(
        default=300,
        metadata={"help": "Timeout in seconds for a single request waiting for its turn and completion."},
    )
    token_chunk_size: int = field(
        default=64,
        metadata={"help": "Number of tokens to generate per iteration in token-chunk dynamic batching."},
    )

# Global/module-level variables for token-chunk dynamic batching
_SAMPLING_PARAM_NAMES: Optional[frozenset[str]] = None

@dataclass(frozen=True)
class PoolSignature:
    model_name: str
    request_type: Literal["chat", "completion"]
    # Excludes max_tokens and stream
    sampling_params_tuple: tuple[tuple[str, AnyType], ...]
    extra_body_params_tuple: tuple[tuple[str, AnyType], ...]

@dataclass 
class PooledRequestState:
    original_request: AnyType # OAChatCompletionRequest or OACompletionRequest
    completion_event: asyncio.Event
    result_container: list 
    request_id: str 
    request_type: Literal["chat", "completion"]
    pool_signature: PoolSignature # Store the signature for quick checks
    effective_max_tokens: int
    accumulated_content: str = ""
    generated_token_count: int = 0
    original_chat_messages: Optional[list[OAChatMessage]] = None
    original_prompt: Optional[str | list[str]] = None 
    error: Optional[Exception] = None 
    finish_reason: Optional[str] = None
    completed_and_signaled: bool = False
    timed_out: bool = False

pending_requests_by_signature: defaultdict[PoolSignature, list[PooledRequestState]] = defaultdict(list)
active_pool_signature: Optional[PoolSignature] = None
active_pool_requests: list[PooledRequestState] = []


def _get_sampling_param_names() -> frozenset[str]:
    global _SAMPLING_PARAM_NAMES
    if _SAMPLING_PARAM_NAMES is None:
        _SAMPLING_PARAM_NAMES = frozenset(inspect.signature(SamplingParams).parameters.keys())
    return _SAMPLING_PARAM_NAMES

def create_pool_signature(
    model_name: str,
    request_type: Literal["chat", "completion"],
    raw_request_params: dict[str, AnyType], # Contains all original request fields like temp, top_p, etc.
    extra_body: Optional[dict[str, AnyType]]
) -> PoolSignature:
    valid_sampling_keys = _get_sampling_param_names()
    
    sig_sampling_items = []
    key_openai_to_vllm_map = {
        "temperature": "temperature", "top_p": "top_p", "n": "n", 
        "presence_penalty": "presence_penalty", "frequency_penalty": "frequency_penalty",
        "stop": "stop", "seed": "seed", "ignore_eos": "ignore_eos", "min_tokens": "min_tokens",
    }
    
    # Use defaults from Pydantic models if not provided in request
    param_defaults_for_sig = {
        "temperature": OAChatCompletionRequest.model_fields["temperature"].default,
        "top_p": OAChatCompletionRequest.model_fields["top_p"].default,
        "presence_penalty": OAChatCompletionRequest.model_fields["presence_penalty"].default,
        "frequency_penalty": OAChatCompletionRequest.model_fields["frequency_penalty"].default,
        "n": OAChatCompletionRequest.model_fields["n"].default,
        "stop": OAChatCompletionRequest.model_fields["stop"].default,
        # stop: None, seed: None, ignore_eos: False, min_tokens: 0
    }

    for oa_key, vllm_key in key_openai_to_vllm_map.items():
        if vllm_key in valid_sampling_keys:
            value = raw_request_params.get(oa_key, param_defaults_for_sig.get(oa_key)) # Use default if not in request
            # For 'stop', ensure it's a tuple if it's a list for hashability
            if vllm_key == "stop" and isinstance(value, list):
                value = tuple(value)
            if value is not None: # Only add if value is meaningfully set or defaulted for signature
                 sig_sampling_items.append((vllm_key, value))
    
    # Sort for stable signature
    sig_sampling_items.sort(key=lambda item: item[0])

    filtered_extra_body_items = []
    if extra_body:
        for k, v in sorted(extra_body.items()):
            # Only include extra_body items that are NOT already part of vLLM SamplingParams
            # to avoid them influencing signature if they are just alternative ways to pass standard params.
            # This primarily targets things like 'guided_decoding_regex'.
            if k not in valid_sampling_keys: 
                 filtered_extra_body_items.append((k,v))
                 
    return PoolSignature(
        model_name=model_name,
        request_type=request_type,
        sampling_params_tuple=tuple(sig_sampling_items),
        extra_body_params_tuple=tuple(filtered_extra_body_items)
    )

def llm_worker(
    script_args: ScriptArguments, data_parallel_rank: int, master_port: int, connection: MPConnection
) -> None:
    # Set required environment variables for DP to work with vLLM
    os.environ["VLLM_DP_RANK"] = str(data_parallel_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(data_parallel_rank)
    os.environ["VLLM_DP_SIZE"] = str(script_args.data_parallel_size)
    os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)

    llm = LLM(
        model=script_args.model,
        revision=script_args.revision,
        tensor_parallel_size=script_args.tensor_parallel_size,
        gpu_memory_utilization=script_args.gpu_memory_utilization,
        enforce_eager=script_args.enforce_eager,
        dtype=script_args.dtype,
        enable_prefix_caching=script_args.enable_prefix_caching,
        max_model_len=script_args.max_model_len,
        worker_extension_cls="verifiers.inference.vllm_openai.WeightSyncWorkerExtension",
    )

    # Send ready signal to parent process
    connection.send({"status": "ready"})

    while True:
        # Wait for commands from the parent process
        try:
            command = connection.recv()
        except KeyboardInterrupt:
            llm.collective_rpc(method="close_communicator")
            break

        # Handle commands
        if command["type"] in ["call", "fire_and_forget"]:
            method_name = command["method"]
            args, kwargs = command.get("args", ()), command.get("kwargs", {})
            method = getattr(llm, method_name)
            result = method(*args, **kwargs)
            if command["type"] == "call":
                connection.send(result)
        elif command["type"] == "shutdown":
            break


def chunk_list(lst: list, n: int) -> list[list]:
    """
    Split list `lst` into `n` evenly distributed sublists.

    Example:
        >>> chunk_list([1, 2, 3, 4, 5, 6], 2)
        [[1, 2, 3], [4, 5, 6]]
        >>> chunk_list([1, 2, 3, 4, 5, 6], 4)
        [[1, 2], [3, 4], [5], [6]]
        >>> chunk_list([1, 2, 3, 4, 5, 6], 8)
        [[1], [2], [3], [4], [5], [6], [], []]
    """
    k, r = divmod(len(lst), n)
    return [lst[i * k + min(i, r) : (i + 1) * k + min(i + 1, r)] for i in range(n)]

async def batch_processing_loop(
    script_args: ScriptArguments, 
    connections: list[AnyType], 
    queue: asyncio.Queue, # This queue now receives PooledRequestState
    logger_instance: logging.Logger
):
    global pending_requests_by_signature, active_pool_signature, active_pool_requests
    global active_generation_count, generation_count_lock

    if not connections:
        logger_instance.error("Batch Processor: No LLM workers available. Shutting down loop.")
        # We cannot process anything if connections are not there from the start.
        # New requests added to queue will eventually timeout or error when picked by lifespan shutdown.
        return

    while True:
        try:
            # 1. Ingest new requests (non-blocking or short timeout)
            # This part tries to fill pending_requests_by_signature from the main request_queue
            try:
                while True: # Loop to drain current items in asyncio.Queue
                    pooled_req_state: PooledRequestState = queue.get_nowait()
                    pending_requests_by_signature[pooled_req_state.pool_signature].append(pooled_req_state)
                    logger_instance.debug(f"Request {pooled_req_state.request_id} enqueued to pending pool with sig {pooled_req_state.pool_signature}")
                    queue.task_done() # Signal that this item from main queue is taken
            except asyncio.QueueEmpty:
                pass # No new requests in the main queue right now

            # 2. Activate a new pool if current is empty and pending requests exist
            if not active_pool_requests and pending_requests_by_signature:
                # Simple strategy: pick the first signature that has pending requests
                # More sophisticated strategies (e.g. largest batch) could be implemented here
                # items() gives a view, convert to list to pop
                available_signatures = list(pending_requests_by_signature.keys())
                if available_signatures:
                    active_pool_signature = available_signatures[0] # Pick one
                    active_pool_requests = pending_requests_by_signature.pop(active_pool_signature)
                    logger_instance.info(f"Activated new pool with signature: {active_pool_signature}. Size: {len(active_pool_requests)}")
            
            # 3. Merge new matching requests into the active pool
            if active_pool_signature and active_pool_signature in pending_requests_by_signature:
                newly_matching_requests = pending_requests_by_signature.pop(active_pool_signature)
                active_pool_requests.extend(newly_matching_requests)
                logger_instance.info(f"Merged {len(newly_matching_requests)} requests into active pool. New size: {len(active_pool_requests)}")

            # 4. Process active pool chunk
            if active_pool_requests:
                # Take a sub-batch from the active pool
                # If active_pool_requests is not empty, active_pool_signature must be set.
                assert active_pool_signature is not None, "active_pool_signature cannot be None if active_pool_requests is populated"
                sub_batch_to_process: list[PooledRequestState] = []
                sub_batch_size = min(len(active_pool_requests), script_args.max_batch_size)
                sub_batch_to_process = active_pool_requests[:sub_batch_size]
                
                if not sub_batch_to_process: # Should not happen if active_pool_requests was not empty
                    await asyncio.sleep(0.01)
                    continue

                logger_instance.debug(f"[BATCH_PROCESSOR] Processing sub-batch of {len(sub_batch_to_process)} for sig: {active_pool_signature}")

                # Track active generation
                async with generation_count_lock:
                    active_generation_count += 1
                    logger_instance.debug(f"[BATCH_PROCESSOR] Active generation count increased to {active_generation_count}")

                try:
                    # Prepare inputs for LLM
                    # All requests in sub_batch_to_process share active_pool_signature
                    # So, sampling params (except max_tokens) are the same.
                    
                    # Construct SamplingParams from the active_pool_signature
                    # The signature stores param tuples. Convert back to dict for SamplingParams.
                    sig_sampling_dict = dict(active_pool_signature.sampling_params_tuple)
                    sig_extra_body_dict = dict(active_pool_signature.extra_body_params_tuple)

                    # Override 'n' to 1 for chunked generation as per design.
                    # Log if original 'n' was different.
                    original_n = sig_sampling_dict.get('n', 1)
                    if original_n != 1:
                        logger_instance.warning(f"Pool {active_pool_signature}: Original 'n={original_n}' overridden to n=1 for chunked generation.")
                    
                    # Create a new dict for **kwargs, excluding 'n' as it's set explicitly
                    kwargs_for_sampling_params = {k: v for k, v in sig_sampling_dict.items() if k != 'n'}
                    
                    vllm_sampling_params = SamplingParams(
                        **kwargs_for_sampling_params,
                        n=1, # Generate one sequence continuation per request in the chunk
                        max_tokens=script_args.token_chunk_size, # Generate one chunk
                        # Ensure guided_decoding is correctly set up if present in extra_body
                        guided_decoding=GuidedDecodingParams(backend="outlines", regex=sig_extra_body_dict["guided_decoding_regex"]) if "guided_decoding_regex" in sig_extra_body_dict else None,
                        # Remove any params from extra_body that might also be in SamplingParams if they were not filtered by create_pool_signature
                        **{k: v for k, v in sig_extra_body_dict.items() if k in _get_sampling_param_names() and k != "guided_decoding_regex"}
                    )

                    # --- Bucket chat requests by first chunk vs continuing ---
                    first_chunk_inputs = []
                    first_chunk_states = []
                    continue_chunk_states = []
                    prompts_for_vllm = []
                    is_chat_pool = active_pool_signature.request_type == "chat"

                    for req_state in sub_batch_to_process:
                        if is_chat_pool:
                            current_messages = []
                            if req_state.original_chat_messages:
                                current_messages.extend([m.model_dump() for m in req_state.original_chat_messages])
                            if req_state.accumulated_content:
                                current_messages.append({"role": "assistant", "content": req_state.accumulated_content})
                            else:
                                if not current_messages or current_messages[-1]["role"] == "assistant":
                                    logger_instance.warning(f"Request {req_state.request_id} in chat pool has no user prompt to start generation.")
                                    current_messages.append({"role": "user", "content": ""})
                            if req_state.generated_token_count == 0:
                                first_chunk_inputs.append(current_messages)
                                first_chunk_states.append(req_state)
                            else:
                                continue_chunk_states.append(req_state)
                        else:
                            if isinstance(req_state.original_prompt, str):
                                current_prompt = req_state.original_prompt
                            elif isinstance(req_state.original_prompt, list):
                                current_prompt = req_state.original_prompt[0] if req_state.original_prompt else ""
                            else:
                                current_prompt = str(req_state.original_prompt or "")
                            prompts_for_vllm.append(current_prompt + req_state.accumulated_content)

                    # Only process first-chunk chat requests in this tick, then continuing if no first-chunk left
                    llm_results = []
                    processed_states = []
                    if is_chat_pool:
                        loop = asyncio.get_running_loop()
                        if first_chunk_inputs:
                            flags = dict(add_generation_prompt=True, continue_final_message=False)
                            payload = {
                                "type": "call",
                                "method": "chat",
                                "kwargs": {
                                    "messages": first_chunk_inputs,
                                    "sampling_params": vllm_sampling_params,
                                    **flags,
                                },
                            }
                            llm_results = await loop.run_in_executor(None, send_and_recv, connections[0], payload)
                            processed_states = first_chunk_states
                        elif continue_chunk_states:
                            # No first-chunk requests, process continuing requests
                            continue_chunk_inputs = []
                            for req_state in continue_chunk_states:
                                current_messages = []
                                if req_state.original_chat_messages:
                                    current_messages.extend([m.model_dump() for m in req_state.original_chat_messages])
                                if req_state.accumulated_content:
                                    current_messages.append({"role": "assistant", "content": req_state.accumulated_content})
                                else:
                                    if not current_messages or current_messages[-1]["role"] == "assistant":
                                        logger_instance.warning(f"Request {req_state.request_id} in chat pool has no user prompt to start generation.")
                                        current_messages.append({"role": "user", "content": ""})
                                continue_chunk_inputs.append(current_messages)
                            flags = dict(add_generation_prompt=False, continue_final_message=True)
                            payload = {
                                "type": "call",
                                "method": "chat",
                                "kwargs": {
                                    "messages": continue_chunk_inputs,
                                    "sampling_params": vllm_sampling_params,
                                    **flags,
                                },
                            }
                            llm_results = await loop.run_in_executor(None, send_and_recv, connections[0], payload)
                            processed_states = continue_chunk_states
                        # else: nothing to process this tick
                    else:
                        # completion – unchanged
                        loop = asyncio.get_running_loop()
                        payload = {
                            "type": "call",
                            "method": "generate",
                            "kwargs": {"prompts": prompts_for_vllm, "sampling_params": vllm_sampling_params},
                        }
                        llm_results = await loop.run_in_executor(None, send_and_recv, connections[0], payload)
                        processed_states = sub_batch_to_process

                    # Now, update state for each request in the processed_states
                    temp_failed_requests_in_sub_batch: list[PooledRequestState] = []

                    if is_chat_pool:
                        if processed_states and (len(llm_results) != len(processed_states)):
                            logger_instance.error(f"LLM result count mismatch. Expected {len(processed_states)}, got {len(llm_results)} for sig {active_pool_signature}. Marking affected requests in sub-batch as error.")
                            for req_state in processed_states:
                                if not req_state.completed_and_signaled:
                                    req_state.error = RuntimeError("LLM result mismatch in batch processing.")
                                    req_state.finish_reason = "error"
                                    temp_failed_requests_in_sub_batch.append(req_state)
                        else:
                            real_idx = 0
                            for req_state in processed_states:
                                if req_state.completed_and_signaled:
                                    continue
                                request_output = llm_results[real_idx]
                                real_idx += 1
                                if not request_output.outputs or len(request_output.outputs) == 0:
                                    logger_instance.warning(f"Request {req_state.request_id} (idx {real_idx-1}) received no output from vLLM in chunk.")
                                    req_state.finish_reason = request_output.outputs[0].finish_reason if request_output.outputs else "error_no_output"
                                    temp_failed_requests_in_sub_batch.append(req_state)
                                    continue
                                completion_output = request_output.outputs[0]
                                new_text_chunk = completion_output.text
                                req_state.accumulated_content += new_text_chunk
                                new_token_count = len(completion_output.token_ids)
                                req_state.generated_token_count += new_token_count
                                req_state.finish_reason = completion_output.finish_reason
                                is_finished = False
                                
                                # If we got fewer tokens than requested chunk size, generation is done
                                if new_token_count < script_args.token_chunk_size:
                                    is_finished = True
                                    if req_state.finish_reason is None or req_state.finish_reason == "length":
                                        # Set appropriate finish reason if not already set
                                        if req_state.generated_token_count >= req_state.effective_max_tokens:
                                            req_state.finish_reason = "length"
                                        else:
                                            # vLLM stopped for some other reason (possibly internal limit)
                                            req_state.finish_reason = "length"
                                    logger_instance.debug(f"Request {req_state.request_id} finished due to incomplete chunk. Generated {new_token_count}/{script_args.token_chunk_size} tokens in chunk, total: {req_state.generated_token_count}")
                                elif req_state.finish_reason == "stop":
                                    is_finished = True
                                elif req_state.finish_reason and req_state.finish_reason != "length":
                                    is_finished = True
                                elif req_state.generated_token_count >= req_state.effective_max_tokens:
                                    is_finished = True
                                    req_state.finish_reason = "length"
                                
                                if is_finished:
                                    logger_instance.debug(f"Request {req_state.request_id} finished. Reason: {req_state.finish_reason}. Total tokens: {req_state.generated_token_count}")
                                else:
                                    logger_instance.debug(f"Request {req_state.request_id} got chunk. New total tokens: {req_state.generated_token_count}. Reason: {req_state.finish_reason}")
                    else:
                        # completion – unchanged
                        if len(llm_results) != len(processed_states):
                            logger_instance.error(f"LLM result count mismatch. Expected {len(processed_states)}, got {len(llm_results)} for sig {active_pool_signature}. Marking affected requests in sub-batch as error.")
                            for req_state in processed_states:
                                if not req_state.completed_and_signaled:
                                    req_state.error = RuntimeError("LLM result mismatch in batch processing.")
                                    req_state.finish_reason = "error"
                                    temp_failed_requests_in_sub_batch.append(req_state)
                        else:
                            real_idx = 0
                            for req_state in processed_states:
                                if req_state.completed_and_signaled:
                                    continue
                                request_output = llm_results[real_idx]
                                real_idx += 1
                                if not request_output.outputs or len(request_output.outputs) == 0:
                                    logger_instance.warning(f"Request {req_state.request_id} (idx {real_idx-1}) received no output from vLLM in chunk.")
                                    req_state.finish_reason = request_output.outputs[0].finish_reason if request_output.outputs else "error_no_output"
                                    temp_failed_requests_in_sub_batch.append(req_state)
                                    continue
                                completion_output = request_output.outputs[0]
                                new_text_chunk = completion_output.text
                                req_state.accumulated_content += new_text_chunk
                                new_token_count = len(completion_output.token_ids)
                                req_state.generated_token_count += new_token_count
                                req_state.finish_reason = completion_output.finish_reason
                                is_finished = False
                                
                                # If we got fewer tokens than requested chunk size, generation is done
                                if new_token_count < script_args.token_chunk_size:
                                    is_finished = True
                                    if req_state.finish_reason is None or req_state.finish_reason == "length":
                                        # Set appropriate finish reason if not already set
                                        if req_state.generated_token_count >= req_state.effective_max_tokens:
                                            req_state.finish_reason = "length"
                                        else:
                                            # vLLM stopped for some other reason (possibly internal limit)
                                            req_state.finish_reason = "length"
                                    logger_instance.debug(f"Request {req_state.request_id} finished due to incomplete chunk. Generated {new_token_count}/{script_args.token_chunk_size} tokens in chunk, total: {req_state.generated_token_count}")
                                elif req_state.finish_reason == "stop":
                                    is_finished = True
                                elif req_state.finish_reason and req_state.finish_reason != "length":
                                    is_finished = True
                                elif req_state.generated_token_count >= req_state.effective_max_tokens:
                                    is_finished = True
                                    req_state.finish_reason = "length"
                                
                                if is_finished:
                                    logger_instance.debug(f"Request {req_state.request_id} finished. Reason: {req_state.finish_reason}. Total tokens: {req_state.generated_token_count}")
                                else:
                                    logger_instance.debug(f"Request {req_state.request_id} got chunk. New total tokens: {req_state.generated_token_count}. Reason: {req_state.finish_reason}")
                    
                    # Now, handle all finished or errored requests from this sub_batch
                    # These need to be removed from active_pool_requests and have their events set.
                    completed_in_sub_batch: list[PooledRequestState] = []
                    remaining_in_sub_batch: list[PooledRequestState] = []

                    # Iterate over sub_batch_to_process to decide their fate
                    updated_active_pool = []
                    
                    # Create a set of request_ids from sub_batch_to_process for quick lookup
                    sub_batch_ids = {r.request_id for r in sub_batch_to_process}

                    for req_state in active_pool_requests: # Iterate main active pool
                        if req_state.request_id not in sub_batch_ids:
                            updated_active_pool.append(req_state) # Keep if not in current sub-batch
                            continue

                        # req_state is from the current sub_batch. Check its status.
                        is_definitely_finished = (req_state.finish_reason is not None and req_state.finish_reason != "length") or \
                                                 (req_state.finish_reason == "length" and req_state.generated_token_count >= req_state.effective_max_tokens) or \
                                                 (req_state.generated_token_count >= req_state.effective_max_tokens) or \
                                                 (req_state.error is not None)
                        
                        if is_definitely_finished and not req_state.completed_and_signaled:
                            completed_in_sub_batch.append(req_state)
                        elif not req_state.completed_and_signaled: # Not finished, and not yet signaled
                            updated_active_pool.append(req_state) # Keep for next chunk
                        # If already signaled (completed_and_signaled is True), it means it was handled (e.g. timeout), so don't re-add or re-signal.
                    
                    active_pool_requests = updated_active_pool
                    if not active_pool_requests:
                        active_pool_signature = None # Deactivate pool if empty
                        logger_instance.info(f"Deactivated pool {active_pool_signature} as it is now empty.")


                    for req_state in completed_in_sub_batch:
                        if req_state.completed_and_signaled: continue # Already handled

                        response_content: OAChatCompletionResponse | OACompletionResponse | JSONResponse
                        if req_state.error:
                            logger_instance.error(f"Request {req_state.request_id} failed with error: {req_state.error}")
                            response_content = JSONResponse(status_code=500, content={"error": f"Processing error: {str(req_state.error)}", "request_id": req_state.request_id})
                        elif req_state.request_type == "chat":
                            final_choices = [OAChatChoice(
                                index=0,
                                message=OAChatMessage(role="assistant", content=req_state.accumulated_content),
                                finish_reason=req_state.finish_reason
                            )]
                            response_content = OAChatCompletionResponse(
                                id=f"chatcmpl-{uuid4().hex}", # Use original request_id if available? For now, new UUID.
                                created=int(datetime.now(tz=timezone.utc).timestamp()),
                                model=req_state.original_request.model,
                                choices=final_choices
                            )
                        else: # Completion
                            final_choices = [OACompletionChoice(
                                index=0, 
                                text=req_state.accumulated_content, 
                                finish_reason=req_state.finish_reason
                            )]
                            response_content = OACompletionResponse(
                                id=f"cmpl-{uuid4().hex}",
                                created=int(datetime.now(tz=timezone.utc).timestamp()),
                                model=req_state.original_request.model,
                                choices=final_choices
                            )
                        
                        req_state.result_container[0] = response_content
                        req_state.completion_event.set()
                        req_state.completed_and_signaled = True
                
                finally:
                    # Always decrement active generation count
                    async with generation_count_lock:
                        active_generation_count -= 1
                        logger_instance.debug(f"[BATCH_PROCESSOR] Active generation count decreased to {active_generation_count}")
            
            else: # No active pool
                await asyncio.sleep(0.01) # Small sleep if no active pool and pending queue was empty

        except asyncio.CancelledError:
            logger_instance.info("Batch processing loop cancelled.")
            all_requests_to_cancel = list(active_pool_requests)
            active_pool_requests.clear()
            active_pool_signature = None
            for sig_list in pending_requests_by_signature.values():
                all_requests_to_cancel.extend(sig_list)
            pending_requests_by_signature.clear()
            
            for req_state in all_requests_to_cancel:
                if not req_state.completed_and_signaled:
                    req_state.result_container[0] = JSONResponse(status_code=503, content={"error": "Server shutting down, request cancelled."})
                    req_state.completion_event.set()
                    req_state.completed_and_signaled = True
            break
        except Exception as e:
            logger_instance.error(f"Critical error in batch processing loop: {e}", exc_info=True)
            all_requests_to_fail = list(active_pool_requests)
            active_pool_requests.clear()
            active_pool_signature = None
            for sig_list in pending_requests_by_signature.values():
                all_requests_to_fail.extend(sig_list)
            pending_requests_by_signature.clear()

            for req_state in all_requests_to_fail:
                 if not req_state.completed_and_signaled:
                    req_state.result_container[0] = JSONResponse(status_code=500, content={"error": f"Critical batch processor error: {str(e)}"})
                    req_state.completion_event.set()
                    req_state.completed_and_signaled = True
            await asyncio.sleep(1) # Pause before retrying loop

def main(script_args: ScriptArguments):
    global request_queue, batch_processor_task # Allow lifespan to assign to these

    # Spawn dp workers, and setup pipes for communication
    master_port = get_open_port()
    connections: list[AnyType] = [] # Use Any type to avoid PipeConnection vs Connection mismatch
    processes = []
    for data_parallel_rank in range(script_args.data_parallel_size):
        # Use duplex=True to allow bidirectional communication
        # This is needed for "call" type commands that expect responses
        parent_connection, child_connection = Pipe(duplex=True)
        process = Process(target=llm_worker, args=(script_args, data_parallel_rank, master_port, child_connection))
        process.start()
        connections.append(parent_connection)
        processes.append(process)

    @asynccontextmanager 
    async def lifespan(app: FastAPI):
        nonlocal processes # Capture from outer scope
        global request_queue, batch_processor_task # Defined at module level

        logger.info(f"Lifespan: Waiting for {script_args.data_parallel_size} LLM worker(s) to be ready...")
        ready_connections = set()
        
        # Timeout for waiting for workers to get ready (e.g., 5 minutes)
        timeout_seconds = 300 
        start_wait_time = time.time()

        while len(ready_connections) < script_args.data_parallel_size:
            if time.time() - start_wait_time > timeout_seconds:
                logger.error(f"Lifespan: Timeout waiting for all LLM workers. Expected {script_args.data_parallel_size}, got {len(ready_connections)} ready.")
                raise RuntimeError("LLM workers failed to initialize in time")

            for i, connection in enumerate(connections):
                if connection not in ready_connections:
                    # Use poll() with a short timeout to avoid blocking indefinitely if a worker is stuck
                    if connection.poll(0.1): # Check if data is available, with a 0.1s timeout
                        try:
                            msg = connection.recv()
                            logger.info(f"Lifespan: Received message from worker {i}: {msg}")
                            if isinstance(msg, dict) and msg.get("status") == "ready":
                                logger.info(f"Lifespan: LLM worker {i} reported ready.")
                                ready_connections.add(connection)
                            else:
                                logger.warning(f"Lifespan: Received unexpected message from worker {i}: {msg}")
                        except Exception as e:
                            logger.error(f"Lifespan: Error receiving message from worker {i}: {e}")

            if len(ready_connections) < script_args.data_parallel_size:
                time.sleep(0.5) # Brief sleep to avoid busy-waiting if not all workers are ready yet

        if len(ready_connections) == script_args.data_parallel_size:
            logger.info(f"Lifespan: All {script_args.data_parallel_size} LLM worker(s) are ready. Proceeding to yield.")
            # Initialize request queue and start batch processor task
            request_queue = asyncio.Queue()
            logger.info("Lifespan: Initialized request queue for batched chat completions.")
            batch_processor_task = asyncio.create_task(
                batch_processing_loop(script_args, connections, request_queue, logger)
            )
            logger.info("Lifespan: Started batch processing task for chat completions.")
        else:
            logger.error(f"Lifespan: Not all LLM workers became ready. Expected {script_args.data_parallel_size}, got {len(ready_connections)}. Uvicorn might not function correctly. Batch processor NOT started.")
        
        yield
        logger.info("Lifespan: Uvicorn server is shutting down. Cleaning up resources.")

        if batch_processor_task is not None and not batch_processor_task.done():
            logger.info("Lifespan: Cancelling batch processor task...")
            batch_processor_task.cancel()
            try:
                await batch_processor_task
                logger.info("Lifespan: Batch processor task finished.")
            except asyncio.CancelledError:
                logger.info("Lifespan: Batch processor task was cancelled as expected.")
            except Exception as e:
                logger.error(f"Lifespan: Exception during batch processor task shutdown: {e}", exc_info=True)

        # Wait for processes to terminate
        for process in processes:
            process.join(timeout=10)  # Wait for 10 seconds for the process to terminate
            if process.is_alive():
                logger.warning(f"Process {process} is still alive after 10 seconds, attempting to terminate...")
                process.terminate()
                process.join()  # ensure process termination after calling terminate()
                
    app = FastAPI(lifespan=lifespan) 
 
    # Define the endpoints for the model server
    @app.get("/health/")
    async def health():
        """
        Health check endpoint to verify that the server is running.
        """
        return {"status": "ok"}

    @app.get("/get_world_size/")
    async def get_world_size():
        """
        Retrieves the world size of the LLM engine, which is `tensor_parallel_size * data_parallel_size`.

        Returns:
            `dict`:
                A dictionary containing the world size.

        Example response:
        ```json
        {"world_size": 8}
        ```
        """
        return {"world_size": script_args.tensor_parallel_size * script_args.data_parallel_size}

    # -------- OpenAI-chat ---------- #
    @app.post("/v1/chat/completions", response_model=OAChatCompletionResponse)
    async def openai_chat(req: OAChatCompletionRequest):
        global request_queue 

        if request_queue is None:
            logger.error("/v1/chat/completions: Request queue not initialized.")
            return JSONResponse(status_code=503, content={"error": "Server not ready, batch processing queue not initialized."})

        request_id = f"chatcmpl-{uuid4().hex}"
        logger.debug(f"Received chat request {request_id}, model: {req.model}")

        # Create signature for pooling
        # The OAChatCompletionRequest fields are: model, messages, temperature, top_p, max_tokens, stream, extra_body
        # We need to pass the relevant ones to create_pool_signature
        raw_params_for_sig = {
            "temperature": req.temperature,
            "top_p": req.top_p,
            "presence_penalty": req.presence_penalty,
            "frequency_penalty": req.frequency_penalty,
            # "n" is not in OAChatCompletionRequest, defaults to 1 for chat in OpenAI spec
            "n": 1, 
        }
        # Add other SamplingParams-mappable fields if they were part of OAChatCompletionRequest
        # For example, if we added 'stop', 'presence_penalty' etc. to OAChatCompletionRequest.
        # For now, the above are the main ones.

        default_max = OAChatCompletionRequest.model_fields["max_tokens"].default
        effective_max_tokens = req.max_tokens or default_max

        pool_sig = create_pool_signature(
            model_name=req.model,
            request_type="chat",
            raw_request_params=raw_params_for_sig,
            extra_body=req.extra_body
        )
        
        completion_event = asyncio.Event()
        result_container = [None] 
        
        pooled_state = PooledRequestState(
            original_request=req,
            completion_event=completion_event,
            result_container=result_container,
            request_id=request_id,
            request_type="chat",
            pool_signature=pool_sig,
            effective_max_tokens=effective_max_tokens,
            original_chat_messages=req.messages, # Store original messages
        )

        try:
            await request_queue.put(pooled_state)
        except Exception as e:
            logger.error(f"Enqueueing error for {request_id}: {e}", exc_info=True)
            return JSONResponse(status_code=500, content={"error": "Internal server error while queueing request."})
        
        try:
            await asyncio.wait_for(completion_event.wait(), timeout=script_args.batch_request_timeout_seconds)
        except asyncio.TimeoutError:
            logger.error(f"Timeout for chat request {request_id} (model {req.model}).")
            pooled_state.timed_out = True
            pooled_state.completed_and_signaled = True
            pooled_state.completion_event.set()
            return JSONResponse(status_code=504, content={"error": "Request timed out."})
        except Exception as e:
            logger.error(f"Error waiting for completion event for {request_id}: {e}", exc_info=True)
            return JSONResponse(status_code=500, content={"error": "Internal server error while waiting for completion."})

        response_data = result_container[0]
        if response_data is None: # Should ideally be set to an error by processor if timeout internally
            logger.error(f"No result for {request_id} (model {req.model}) after event set. Internal error.")
            return JSONResponse(status_code=500, content={"error": "Internal error: No result from processor."})

        if isinstance(response_data, JSONResponse):
            return response_data
        
        if isinstance(response_data, OAChatCompletionResponse):
             # Must return JSONResponse for FastAPI
            return JSONResponse(response_data.model_dump())
        else:
            logger.error(f"Unexpected result type {type(response_data)} for {request_id}.")
            return JSONResponse(status_code=500, content={"error": "Internal error: Unexpected result format."})

    @app.get("/v1/models")
    async def list_models():
        return {"data": [{"id": script_args.model, "object": "model", "owned_by": "vllm"}]}

    @app.post("/v1/completions", response_model=OACompletionResponse)
    async def openai_completions(req: OACompletionRequest):
        global request_queue

        if request_queue is None:
            logger.error("/v1/completions: Request queue not initialized.")
            return JSONResponse(status_code=503, content={"error": "Server not ready, batch processing queue not initialized."})

        request_id = f"cmpl-{uuid4().hex}"
        logger.debug(f"Received completion request {request_id}, model: {req.model}")

        # OACompletionRequest fields: model, prompt, temperature, top_p, max_tokens, n, extra_body
        raw_params_for_sig = {
            "temperature": req.temperature,
            "top_p": req.top_p,
            "presence_penalty": req.presence_penalty,
            "frequency_penalty": req.frequency_penalty,
            "n": req.n, # Pass 'n' from the request
        }
        # Add other SamplingParams-mappable fields from OACompletionRequest if they exist
        # e.g., req.stop, req.presence_penalty etc. if we add them to OACompletionRequest model
        # For now, the above are the main ones.
        # We need to get ALL fields of OACompletionRequest that are also valid for SamplingParams
        # This is safer:
        valid_sp_keys = _get_sampling_param_names()
        for field_name, field_value in req.model_dump().items():
            if field_name in valid_sp_keys and field_name not in raw_params_for_sig:
                raw_params_for_sig[field_name] = field_value

        default_max = OACompletionRequest.model_fields["max_tokens"].default
        effective_max_tokens = req.max_tokens or default_max

        pool_sig = create_pool_signature(
            model_name=req.model,
            request_type="completion",
            raw_request_params=raw_params_for_sig,
            extra_body=req.extra_body
        )

        completion_event = asyncio.Event()
        result_container = [None]

        # Check for list prompts for completion, which is problematic for current chunking.
        # vLLM's generate can take list of prompts, but our chunking logic (appending to prompt) assumes single string.
        if isinstance(req.prompt, list):
            if len(req.prompt) > 1:
                 logger.warning(f"Request {request_id} for completion has a list of prompts. Only the first prompt will be used for chunked generation.")
                 current_prompt = req.prompt[0] if req.prompt else ""
            elif not req.prompt: # empty list (simplified condition)
                current_prompt = ""
            else: # list with one element
                current_prompt = req.prompt[0]
        else: #string
            current_prompt = req.prompt


        pooled_state = PooledRequestState(
            original_request=req,
            completion_event=completion_event,
            result_container=result_container,
            request_id=request_id,
            request_type="completion",
            pool_signature=pool_sig,
            effective_max_tokens=effective_max_tokens,
            original_prompt=current_prompt, # Store single prompt for chunking
        )

        try:
            await request_queue.put(pooled_state)
        except Exception as e:
            logger.error(f"Enqueueing error for completion {request_id}: {e}", exc_info=True)
            return JSONResponse(status_code=500, content={"error": "Internal server error while queueing request."})

        try:
            await asyncio.wait_for(completion_event.wait(), timeout=script_args.batch_request_timeout_seconds)
        except asyncio.TimeoutError:
            logger.error(f"Timeout for completion request {request_id} (model {req.model}).")
            pooled_state.timed_out = True
            pooled_state.completed_and_signaled = True
            pooled_state.completion_event.set()
            return JSONResponse(status_code=504, content={"error": "Request timed out."})
        except Exception as e:
            logger.error(f"Error waiting for completion event for {request_id}: {e}", exc_info=True)
            return JSONResponse(status_code=500, content={"error": "Internal server error while waiting for completion."})

        response_data = result_container[0]
        if response_data is None:
            logger.error(f"No result for completion {request_id} (model {req.model}) after event set. Internal error.")
            return JSONResponse(status_code=500, content={"error": "Internal error: No result from processor."})

        if isinstance(response_data, JSONResponse):
            return response_data
        
        if isinstance(response_data, OACompletionResponse):
            return JSONResponse(response_data.model_dump()) # Must return JSONResponse for FastAPI
        else:
            logger.error(f"Unexpected result type {type(response_data)} for completion {request_id}.")
            return JSONResponse(status_code=500, content={"error": "Internal error: Unexpected result format."})

    class InitCommunicatorRequest(BaseModel):
        host: str
        port: int
        world_size: int

    @app.post("/init_communicator/")
    async def init_communicator(request: InitCommunicatorRequest):
        """
        Initializes the communicator for synchronizing model weights between a client and multiple server
        workers.

        Args:
            request (`InitCommunicatorRequest`):
                - `host` (`str`): Hostname or IP address of the master node.
                - `port` (`int`): Port number to be used for communication.
                - `world_size` (`int`): Total number of participating processes in the group.
        """
        logger.info(f"[INIT_COMMUNICATOR] Received request: host={request.host}, port={request.port}, world_size={request.world_size}")
        
        # Calculate actual world size based on vLLM configuration
        vllm_world_size = script_args.tensor_parallel_size * script_args.data_parallel_size
        expected_world_size = vllm_world_size + 1  # +1 for the client
        
        logger.info(f"[INIT_COMMUNICATOR] vLLM world size: {vllm_world_size} (TP={script_args.tensor_parallel_size} x DP={script_args.data_parallel_size})")
        logger.info(f"[INIT_COMMUNICATOR] Expected total world size: {expected_world_size}")
        
        if request.world_size != expected_world_size:
            logger.warning(f"[INIT_COMMUNICATOR] World size mismatch! Request: {request.world_size}, Expected: {expected_world_size}")

        # The function init_communicator is called this way: init_communicator(host, port, world_size)
        # So with collective_rpc we need to call it this way:
        # llm.collective_rpc(method="init_communicator", args=(host, port, world_size))
        kwargs = {"method": "init_communicator", "args": (request.host, request.port, expected_world_size)}
        
        # Send to all workers synchronously to ensure they're ready
        successful_workers = []
        failed_workers = []
        
        for i, connection in enumerate(connections):
            logger.debug(f"[INIT_COMMUNICATOR] Sending to worker {i}")
            try:
                connection.send({"type": "fire_and_forget", "method": "collective_rpc", "kwargs": kwargs})
                successful_workers.append(i)
            except Exception as e:
                logger.error(f"[INIT_COMMUNICATOR] Failed to notify worker {i}: {e}")
                failed_workers.append((i, str(e)))

        if failed_workers:
            error_msg = f"Failed to notify workers: {failed_workers}"
            logger.error(f"[INIT_COMMUNICATOR] {error_msg}")
            return JSONResponse(status_code=500, content={"error": error_msg})

        logger.info(f"[INIT_COMMUNICATOR] Successfully notified {len(successful_workers)} workers")
        return {"message": "Request received, initializing communicator", "workers_notified": len(successful_workers)}

    class UpdateWeightsRequest(BaseModel):
        name: str
        dtype: str
        shape: list[int]

    class BatchUpdateWeightsRequest(BaseModel):
        updates: list[UpdateWeightsRequest]

    @app.post("/update_named_param/")
    async def update_named_param(request: UpdateWeightsRequest):
        """
        Updates the model weights with the provided tensor.

        Once this endpoint is called, the client process should broadcast the updated weights to all server workers.

        Args:
            request (`UpdateWeightsRequest`):
                - `name` (`str`): Name of the weight tensor being updated.
                - `dtype` (`str`): Data type of the weight tensor (e.g., `"torch.float32"`).
                - `shape` (list of `int`): Shape of the weight

        """
        # Acquire semaphore to limit concurrent weight updates
        async with weight_update_semaphore:
            logger.debug(f"[UPDATE_PARAM] Received weight update for: {request.name}, dtype={request.dtype}, shape={request.shape}")
            
            # Wait for active generations to complete before updating weights
            # This prevents conflicts between generation and weight loading
            wait_start = time.time()
            if active_generation_count > 0:
                logger.info(f"[UPDATE_PARAM] Waiting for {active_generation_count} active generations to complete before weight update")
            while active_generation_count > 0:
                if time.time() - wait_start > 30.0:  # 30 second timeout
                    logger.warning(f"[UPDATE_PARAM] Timeout waiting for {active_generation_count} active generations to complete")
                    break
                await asyncio.sleep(0.1)
            if active_generation_count == 0:
                logger.debug(f"[UPDATE_PARAM] All generations complete, proceeding with weight update")
            
            # CRITICAL: Notify workers IMMEDIATELY so they're ready for NCCL broadcast
            # This must happen before returning the HTTP response to maintain synchronization with trainer
            dtype = getattr(torch, request.dtype.split(".")[-1])
            kwargs = {"method": "update_named_param", "args": (request.name, dtype, tuple(request.shape))}
            
            # Send to all workers synchronously to ensure they're ready
            # Using fire_and_forget since we don't need the result
            for i, connection in enumerate(connections):
                logger.debug(f"[UPDATE_PARAM] Notifying worker {i} about weight update")
                try:
                    connection.send({"type": "fire_and_forget", "method": "collective_rpc", "kwargs": kwargs})
                except Exception as e:
                    logger.error(f"[UPDATE_PARAM] Failed to notify worker {i}: {e}")
                    return JSONResponse(status_code=500, content={"error": f"Failed to notify worker {i}: {str(e)}"})
            
            logger.debug(f"[UPDATE_PARAM] All workers notified, trainer should now broadcast via NCCL")
            return {"message": "Weight update processed"}

    @app.post("/batch_update_named_params/")
    async def batch_update_named_params(request: BatchUpdateWeightsRequest):
        """
        Updates multiple model weights in a batch. Processes updates sequentially
        to ensure proper synchronization with NCCL broadcasts from the client.

        Args:
            request (`BatchUpdateWeightsRequest`):
                - `updates` (list of `UpdateWeightsRequest`): List of weight updates to process
        """
        logger.info(f"[BATCH_UPDATE] Received batch of {len(request.updates)} weight updates")
        
        # Process updates sequentially to maintain NCCL synchronization
        # The client will broadcast each parameter after we notify workers
        successful = []
        errors = []
        
        for update in request.updates:
            try:
                # Acquire semaphore to limit concurrent updates across different batches
                async with weight_update_semaphore:
                    logger.debug(f"[BATCH_UPDATE] Processing weight update for: {update.name}")
                    
                    # Wait for active generations if needed
                    wait_start = time.time()
                    while active_generation_count > 0:
                        if time.time() - wait_start > 30.0:
                            logger.warning(f"[BATCH_UPDATE] Timeout waiting for generations")
                            break
                        await asyncio.sleep(0.1)
                    
                    # Notify workers synchronously
                    dtype = getattr(torch, update.dtype.split(".")[-1])
                    kwargs = {"method": "update_named_param", "args": (update.name, dtype, tuple(update.shape))}
                    
                    for i, connection in enumerate(connections):
                        try:
                            connection.send({"type": "fire_and_forget", "method": "collective_rpc", "kwargs": kwargs})
                        except Exception as e:
                            logger.error(f"[BATCH_UPDATE] Failed to notify worker {i} for {update.name}: {e}")
                            raise Exception(f"Failed to notify worker {i}")
                    
                    successful.append(update.name)
                    logger.debug(f"[BATCH_UPDATE] Workers notified for {update.name}")
                    
            except Exception as e:
                errors.append({"param": update.name, "error": str(e)})
                logger.error(f"[BATCH_UPDATE] Error processing {update.name}: {e}")
        
        if errors:
            return JSONResponse(
                status_code=207,  # Multi-Status
                content={
                    "message": f"Batch update completed with {len(errors)} errors",
                    "successful": successful,
                    "errors": errors
                }
            )
        
        logger.info(f"[BATCH_UPDATE] Successfully processed {len(successful)} weight updates")
        return {"message": f"Successfully updated {len(successful)} parameters", "successful": successful}

    @app.post("/reset_prefix_cache/")
    async def reset_prefix_cache():
        """
        Resets the prefix cache for the model.
        """
        # Send requests and collect results synchronously
        all_outputs = []
        for connection in connections:
            try:
                connection.send({"type": "call", "method": "reset_prefix_cache"})
                output = connection.recv()
                all_outputs.append(output)
            except Exception as e:
                logger.error(f"Failed to reset prefix cache on worker: {e}")
                all_outputs.append(False)
        
        success = all(output for output in all_outputs)
        return {"message": "Request received, resetting prefix cache status: " + str(success)}

    @app.post("/close_communicator/")
    async def close_communicator():
        """
        Closes the weight update group and cleans up associated resources.
        """
        kwargs = {"method": "close_communicator"}
        
        # Send to all workers
        for connection in connections:
            try:
                connection.send({"type": "fire_and_forget", "method": "collective_rpc", "kwargs": kwargs})
            except Exception as e:
                logger.warning(f"Failed to send close_communicator to worker: {e}")
                # Don't fail the request if we can't notify a worker during shutdown
        
        return {"message": "Request received, closing communicator"}

    # Start the server
    # Always use 1 Uvicorn worker. vLLM handles its own worker processes and scheduling.
    num_uvicorn_workers = 1

    logger.info(f"Starting Uvicorn with {num_uvicorn_workers} worker(s). Data parallel size: {script_args.data_parallel_size}")
    uvicorn.run(
        app, 
        host=script_args.host, 
        port=script_args.port, 
        log_level=script_args.log_level,
        workers=num_uvicorn_workers
    )


def make_parser(subparsers: Optional[argparse._SubParsersAction] = None):
    if subparsers is not None:
        parser = subparsers.add_parser("vllm-serve", help="Run the vLLM serve script", dataclass_types=ScriptArguments)
    else:
        parser = TrlParser(ScriptArguments)
    return parser

if __name__ == "__main__":
    parser = make_parser()
    (script_args,) = parser.parse_args_and_config()
    main(script_args)