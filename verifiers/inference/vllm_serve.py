"""
OpenAI-compatible vLLM server with weight synchronization.

Usage:

```bash
uv run python vllm_serve.py --model <model_name> --port <port>
```

Supports:
- /v1/chat/completions
- /v1/completions

This script is adapted from trl/scripts/vllm_serve.py (huggingface/trl)
"""

import argparse
import logging
import os
import time # Added back time as it's used in lifespan
import asyncio # For run_in_executor
import threading # For Lock
from collections.abc import Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from itertools import chain
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from typing import Optional, Sequence

import torch

from trl import TrlParser

# FastAPI
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse 

# NEW – OpenAI-spec helpers
from uuid import uuid4
from datetime import datetime, timezone

from pydantic import BaseModel

import uvicorn

from vllm import LLM, SamplingParams
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.parallel_state import get_world_group
from vllm.distributed.utils import StatelessProcessGroup
from vllm.sampling_params import GuidedDecodingParams
from vllm.utils import get_open_port

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Ensure logger is defined

# We use CUDA with multiprocessing, so we must use the 'spawn' start method. Otherwise, we will get the following
# error: RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use
# the 'spawn' start method
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# At the global level, after imports and logger setup:
pipe_lock = threading.Lock()

def send_and_recv(conn: Connection, payload: dict):
    """Helper to send a payload and receive a response over a pipe, protected by a lock."""
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

        # Create a stateless process group to manage communication between training processes and vLLM workers.
        pg = StatelessProcessGroup.create(host=host, port=port, rank=rank, world_size=world_size)

        # Initialize the NCCL-based communicator for weight synchronization.
        self.pynccl_comm = PyNcclCommunicator(pg, device=self.device)

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
        if self.pynccl_comm is None:
            raise RuntimeError("Communicator not initialized. Call `init_communicator` first.")

        # Allocate memory for the incoming weight tensor on the correct device.
        weight = torch.empty(shape, dtype=dtype, device=self.device)

        # Use NCCL to broadcast the updated weights from the client (src) to all workers.
        self.pynccl_comm.broadcast(weight, src=self.client_rank)
        self.pynccl_comm.group.barrier()

        # Load the received weights into the model.
        self.model_runner.model.load_weights(weights=[(name, weight)]) # type: ignore

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

def llm_worker(
    script_args: ScriptArguments, data_parallel_rank: int, master_port: int, connection: Connection
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
        # Automatic Prefix Caching caches the KV cache of existing queries, so that a new query can
        # directly reuse the KV cache if it shares the same prefix with one of the existing queries.
        # This is particularly useful here because we generate completions from the same prompts.
        enable_prefix_caching=script_args.enable_prefix_caching,
        max_model_len=script_args.max_model_len,
        worker_extension_cls="verifiers.inference.vllm_serve.WeightSyncWorkerExtension",
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

def main(script_args: ScriptArguments):

    # Spawn dp workers, and setup pipes for communication
    master_port = get_open_port()
    connections: list[Connection] = [] # Added type hint for clarity
    processes = []
    for data_parallel_rank in range(script_args.data_parallel_size):
        parent_connection, child_connection = Pipe()
        process = Process(target=llm_worker, args=(script_args, data_parallel_rank, master_port, child_connection))
        process.start()
        connections.append(parent_connection)
        processes.append(process)

    @asynccontextmanager 
    async def lifespan(app: FastAPI):
        logger.info(f"Lifespan: Waiting for {script_args.data_parallel_size} LLM worker(s) to be ready...")
        ready_connections = set()
        
        # Timeout for waiting for workers to get ready (e.g., 5 minutes)
        timeout_seconds = 300 
        start_wait_time = time.time()

        while len(ready_connections) < script_args.data_parallel_size:
            if time.time() - start_wait_time > timeout_seconds:
                logger.error(f"Lifespan: Timeout waiting for all LLM workers. Expected {script_args.data_parallel_size}, got {len(ready_connections)} ready.")
                # Optionally, could raise an exception here to stop Uvicorn from starting in a bad state
                # For now, just logging and proceeding might lead to Uvicorn starting but being non-functional.
                # Consider raising RuntimeError("LLM workers failed to initialize in time")
                break # Exit loop to allow Uvicorn to start, though it might not work well

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
                    # else: # Optional: log if a connection is polled but has no data yet
                        # logger.debug(f"Lifespan: Polled worker {i}, no message yet.") 
            
            if len(ready_connections) < script_args.data_parallel_size:
                time.sleep(0.5) # Brief sleep to avoid busy-waiting if not all workers are ready yet

        if len(ready_connections) == script_args.data_parallel_size:
            logger.info(f"Lifespan: All {script_args.data_parallel_size} LLM worker(s) are ready. Proceeding to yield.")
        else:
            logger.error(f"Lifespan: Not all LLM workers became ready. Expected {script_args.data_parallel_size}, got {len(ready_connections)}. Uvicorn might not function correctly.")
        
        yield
        logger.info("Lifespan: Uvicorn server is shutting down. Cleaning up resources.")

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
    class OAChatMessage(BaseModel):
        role: str
        content: str

    class OAChatCompletionRequest(BaseModel):
        model: str
        messages: list[OAChatMessage]
        temperature: float | None = 1.0
        top_p:     float | None = 1.0
        max_tokens: int | None  = 16
        stream: bool = False
        # everything else is accepted but ignored for brevity
        extra_body: dict | None = None   # <- lets callers pass vLLM-specific params

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
    
    @app.post("/v1/chat/completions", response_model=OAChatCompletionResponse)
    async def openai_chat(req: OAChatCompletionRequest):
        vllm_specific_params = req.extra_body or {}
        guided_decoding = None
        if "guided_decoding_regex" in vllm_specific_params:
            guided_decoding = GuidedDecodingParams(backend="outlines", regex=vllm_specific_params.pop("guided_decoding_regex"))

        sampling_params = SamplingParams(
            n=1,
            temperature=req.temperature if req.temperature is not None else 1.0,
            top_p=req.top_p if req.top_p is not None else 1.0,
            max_tokens=req.max_tokens if req.max_tokens is not None else 16,
            guided_decoding=guided_decoding,
            **{k: v for k, v in vllm_specific_params.items() if k in SamplingParams.__fields__} # Pass only valid SamplingParams
        )

        chat_messages_for_vllm = [msg.model_dump() for msg in req.messages]

        if not connections:
            logger.error("No LLM workers available for openai_chat")
            return JSONResponse(status_code=503, content={"error": "No LLM workers available"})

        payload = {
            "type": "call", 
            "method": "chat",
            "kwargs": {"messages": chat_messages_for_vllm, "sampling_params": sampling_params}
        }

        loop = asyncio.get_running_loop()
        # Send request to the first LLM worker and get response
        # The send_and_recv helper handles locking for pipe access.
        first_worker_result_list = await loop.run_in_executor(
            None, send_and_recv, connections[0], payload
        )
        
        final_choices = []
        if first_worker_result_list:
            if isinstance(first_worker_result_list[0], object): # vLLM returns a list of RequestOutput
                request_output = first_worker_result_list[0]
                if hasattr(request_output, 'outputs') and request_output.outputs:
                    completion_output = request_output.outputs[0]
                    final_choices.append(OAChatChoice(
                        index=0,
                        message=OAChatMessage(role="assistant", content=completion_output.text),
                        finish_reason=completion_output.finish_reason
                    ))
                else:
                    logger.warning(f"openai_chat: RequestOutput from worker missing 'outputs'. req_id: {request_output.request_id if hasattr(request_output, 'request_id') else 'N/A'}")
            else:
                logger.warning("openai_chat: Worker returned None or non-object for the request output item.")
        else:
            logger.warning("openai_chat: No results received from worker or worker returned empty list.")

        if not final_choices:
            final_choices.append(OAChatChoice(
                index=0,
                message=OAChatMessage(role="assistant", content="Error processing LLM response."),
                finish_reason="error"
            ))

        out = OAChatCompletionResponse(
            id=f"chatcmpl-{uuid4().hex}",
            created=int(datetime.now(tz=timezone.utc).timestamp()),
            model=req.model,
            choices=final_choices
        )
        return JSONResponse(out.model_dump())

    @app.get("/v1/models")
    async def list_models():
        return {"data": [{"id": script_args.model, "object": "model", "owned_by": "vllm"}]}

    # -------- OpenAI-completions ---------- #
    class OACompletionRequest(BaseModel):
        model: str
        prompt: str | list[str] # OpenAI API can take a string or list of strings for prompt
        temperature: float | None = 1.0
        top_p:     float | None = 1.0
        max_tokens: int | None  = 16
        n: int | None = 1 # Number of completions to generate for each prompt
        # stream: bool = False # Deferring stream for now
        # everything else is accepted but ignored for brevity
        extra_body: dict | None = None   # <- lets callers pass vLLM-specific params

    class OACompletionChoice(BaseModel):
        index: int
        text: str
        logprobs: dict | None = None # Not currently supported by vLLM output, but part of OpenAI spec
        finish_reason: str | None = "length" # Assuming "length" if max_tokens reached, or "stop" if a stop token is hit (not explicitly handled here yet)

    class OACompletionResponse(BaseModel):
        id: str
        object: str = "text_completion"
        created: int
        model: str
        choices: list[OACompletionChoice]

    @app.post("/v1/completions", response_model=OACompletionResponse)
    async def openai_completions(req: OACompletionRequest):
        vllm_specific_params = req.extra_body or {}
        guided_decoding = None
        if "guided_decoding_regex" in vllm_specific_params:
            guided_decoding = GuidedDecodingParams(backend="outlines", regex=vllm_specific_params.pop("guided_decoding_regex"))

        sampling_params = SamplingParams(
            n=req.n or 1,
            temperature=req.temperature if req.temperature is not None else 1.0,
            top_p=req.top_p if req.top_p is not None else 1.0,
            max_tokens=req.max_tokens if req.max_tokens is not None else 16,
            guided_decoding=guided_decoding,
            **{k: v for k, v in vllm_specific_params.items() if k in SamplingParams.__fields__} # Pass only valid SamplingParams
        )

        prompts_for_vllm = req.prompt

        if not connections:
            logger.error("No LLM workers available for openai_completions")
            return JSONResponse(status_code=503, content={"error": "No LLM workers available"})

        loop = asyncio.get_running_loop()
        all_raw_outputs = []

        if script_args.data_parallel_size > 1 and isinstance(prompts_for_vllm, list):
            chunked_prompts = chunk_list(prompts_for_vllm, script_args.data_parallel_size)
            tasks = []
            for i, conn in enumerate(connections):
                current_prompts_for_worker = chunked_prompts[i] if i < len(chunked_prompts) else []
                if not current_prompts_for_worker: # Avoid sending empty lists if a worker gets no prompts
                    tasks.append(loop.run_in_executor(None, lambda: [])) # Return an empty list for this worker
                    continue
                payload = {
                    "type": "call",
                    "method": "generate",
                    "kwargs": {"prompts": current_prompts_for_worker, "sampling_params": sampling_params}
                }
                tasks.append(loop.run_in_executor(None, send_and_recv, conn, payload))
            all_raw_outputs = await asyncio.gather(*tasks)
        else:
            # If not data parallel or prompt is a single string, send to the first worker.
            # chunk_list handles single string to list of list for the worker.
            prompts_to_send = [prompts_for_vllm] if isinstance(prompts_for_vllm, str) else prompts_for_vllm
            payload = {
                "type": "call",
                "method": "generate",
                "kwargs": {"prompts": prompts_to_send, "sampling_params": sampling_params}
            }
            # Since send_and_recv expects a list of RequestOutput, and we get that from a single call for single/all prompts
            result_from_worker = await loop.run_in_executor(None, send_and_recv, connections[0], payload)
            all_raw_outputs = [result_from_worker] # Make it a list to match the structure expected by subsequent code

        final_choices = []
        overall_choice_idx = 0
        for worker_output_list in all_raw_outputs:
            if isinstance(worker_output_list, list):
                for request_output in worker_output_list:
                    if hasattr(request_output, 'outputs') and isinstance(request_output.outputs, list):
                        for completion_output in request_output.outputs:
                            final_choices.append(OACompletionChoice(
                                index=overall_choice_idx,
                                text=completion_output.text,
                                finish_reason=completion_output.finish_reason
                            ))
                            overall_choice_idx += 1
                    else:
                        logger.warning(f"openai_completions: RequestOutput missing 'outputs'. req_id: {request_output.request_id if hasattr(request_output, 'request_id') else 'N/A'}")
            else:
                logger.warning(f"openai_completions: Expected list from worker, got {type(worker_output_list)}")

        if not final_choices:
            num_prompts_req = len(prompts_for_vllm) if isinstance(prompts_for_vllm, list) else 1
            for i in range(num_prompts_req * (req.n or 1)):
                 final_choices.append(OACompletionChoice(index=i, text="Error: No output from LLM.", finish_reason="error"))

        out = OACompletionResponse(
            id=f"cmpl-{uuid4().hex}",
            created=int(datetime.now(tz=timezone.utc).timestamp()),
            model=req.model,
            choices=final_choices
        )
        return JSONResponse(out.model_dump())

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
        world_size = script_args.tensor_parallel_size * script_args.data_parallel_size + 1

        # The function init_communicator is called this way: init_communicator(host, port, world_size)
        # So with collective_rpc we need to call it this way:
        # llm.collective_rpc(method="init_communicator", args=(host, port, world_size))
        kwargs = {"method": "init_communicator", "args": (request.host, request.port, world_size)}
        for connection in connections:
            connection.send({"type": "fire_and_forget", "method": "collective_rpc", "kwargs": kwargs})

        return {"message": "Request received, initializing communicator"}

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


def make_parser(subparsers: argparse._SubParsersAction = None):
    if subparsers is not None:
        parser = subparsers.add_parser("vllm-serve", help="Run the vLLM serve script", dataclass_types=ScriptArguments)
    else:
        parser = TrlParser(ScriptArguments)
    return parser

if __name__ == "__main__":
    parser = make_parser()
    (script_args,) = parser.parse_args_and_config()
    main(script_args)
