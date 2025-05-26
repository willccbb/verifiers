import os
import signal
import argparse
import warnings

import torch

from trl import TrlParser


#if is_fastapi_available():
from fastapi import BackgroundTasks, FastAPI


#if is_pydantic_available():
from pydantic import BaseModel

#if is_uvloop_available():
import uvloop


#if is_vllm_available():
from vllm.logger import init_logger
from vllm.utils import FlexibleArgumentParser, set_ulimit, is_valid_ipv6_address
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.reasoning import ReasoningParserManager
from vllm.entrypoints.openai.tool_parsers import ToolParserManager
from vllm.entrypoints.openai.api_server import (
    build_app,
    build_async_engine_client,
    serve_http,
    init_app_state,
    create_server_socket,
    cli_env_setup,
    make_arg_parser,
)
from vllm.entrypoints.openai.cli_args import (
    make_arg_parser,
    validate_parsed_serve_args,
)
from vllm.version import __version__ as VLLM_VERSION

TIMEOUT_KEEP_ALIVE = 5  # seconds


logger = init_logger(__name__)

# We use CUDA with multiprocessing, so we must use the 'spawn' start method. Otherwise, we will get the following
# error: RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use
# the 'spawn' start method
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

def add_vllm_client_endpoints(app: FastAPI, llm: AsyncLLM):
    @app.get("/get_world_size/")
    async def get_world_size():
        """
        Retrieves the tensor parallel size from the LLM engine.

        Returns:
            `dict`:
                A dictionary containing the world size.

        Example response:
        ```json
        {"world_size": 8}
        ```
        """
        tp = llm.vllm_config.parallel_config.tensor_parallel_size
        dp = llm.vllm_config.parallel_config.data_parallel_size
        return {"world_size": tp * dp}

    class InitCommunicatorRequest(BaseModel):
        host: str
        port: int
        world_size: int

    @app.post("/init_communicator/")
    async def init_communicator(request: InitCommunicatorRequest, background_tasks: BackgroundTasks):
        """
        Initializes the communicator for synchronizing model weights between a client and multiple server
        workers.

        Args:
            request (`InitCommunicatorRequest`):
                - `host` (`str`): Hostname or IP address of the master node.
                - `port` (`int`): Port number to be used for communication.
                - `world_size` (`int`): Total number of participating processes in the group.
        """
        tp = llm.vllm_config.parallel_config.tensor_parallel_size
        dp = llm.vllm_config.parallel_config.data_parallel_size
        background_tasks.add_task(
            llm.engine_core.collective_rpc_async,
            "init_communicator",
            args=(request.host, request.port, tp * dp + 1)
        )
        return {"message": "Request received, initializing communicator"}

    class UpdateWeightsRequest(BaseModel):
        name: str
        dtype: str
        shape: list[int]

    @app.post("/update_named_param/")
    async def update_named_param(request: UpdateWeightsRequest, background_tasks: BackgroundTasks):
        """
        Updates the model weights with the provided tensor.

        Once this endpoint is called, the client process should broadcast the updated weights to all server workers.

        Args:
            request (`UpdateWeightsRequest`):
                - `name` (`str`): Name of the weight tensor being updated.
                - `dtype` (`str`): Data type of the weight tensor (e.g., `"torch.float32"`).
                - `shape` (list of `int`): Shape of the weight

        """
        # The function is called this way: update_named_param(name="name", dtype=torch.float32, shape=(10, 10))
        # So with collect_rpc we need to call it this way:
        # llm.collective_rpc("update_named_param", args=("name", torch.float32, (10, 10)))
        # And with background_tasks.add_task we need to call it this way:
        # background_tasks.add_task(llm.collective_rpc, "update_named_param", args=("name", torch.float32, (10, 10)))
        dtype = torch.__getattribute__(request.dtype.split(".")[-1])
        background_tasks.add_task(
            llm.engine_core.collective_rpc_async,
            "update_named_param",
            args=(request.name, dtype, request.shape)
        )
        return {"message": "Request received, updating named parameter"}

    @app.post("/reset_prefix_cache/")
    async def reset_prefix_cache(background_tasks: BackgroundTasks):
        """
        Resets the prefix cache for the model.
        """
        background_tasks.add_task(
            llm.engine_core.reset_prefix_cache_async
        )
        return {"message": "Request received, resetting prefix cache"}

    @app.post("/close_communicator/")
    async def close_communicator(background_tasks: BackgroundTasks):
        """
        Closes the weight update group and cleans up associated resources.
        """
        background_tasks.add_task(
            llm.engine_core.collective_rpc_async,
            "close_communicator"
        )
        return {"message": "Request received, closing communicator"}


async def run_server(args, **uvicorn_kwargs):
    if not is_fastapi_available():
        raise ImportError(
            "FastAPI is required to run the vLLM serve script. Please install it using `pip install fastapi`."
        )

    if not is_pydantic_available():
        raise ImportError(
            "Pydantic is required to run the vLLM serve script. Please install it using `pip install pydantic`."
        )
        
    if not is_uvicorn_available():
        raise ImportError(
            "Uvicorn is required to run the vLLM serve script. Please install it using `pip install uvicorn`."
        )
    
    if not is_uvloop_available():
        raise ImportError(
            "Uvloop is required to run the vLLM serve script. Please install it using `pip install uvloop`."
        )

    logger.info("vLLM API server version %s", VLLM_VERSION)
    logger.info("args: %s", args)

    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)

    valid_tool_parses = ToolParserManager.tool_parsers.keys()
    if args.enable_auto_tool_choice \
        and args.tool_call_parser not in valid_tool_parses:
        raise KeyError(f"invalid tool call parser: {args.tool_call_parser} "
                       f"(chose from {{ {','.join(valid_tool_parses)} }})")

    valid_reasoning_parses = ReasoningParserManager.reasoning_parsers.keys()
    if args.enable_reasoning \
        and args.reasoning_parser not in valid_reasoning_parses:
        raise KeyError(
            f"invalid reasoning parser: {args.reasoning_parser} "
            f"(chose from {{ {','.join(valid_reasoning_parses)} }})")

    # workaround to make sure that we bind the port before the engine is set up.
    # This avoids race conditions with ray.
    # see https://github.com/vllm-project/vllm/issues/8204
    sock_addr = (args.host or "", args.port)
    sock = create_server_socket(sock_addr)

    # workaround to avoid footguns where uvicorn drops requests with too
    # many concurrent requests active
    set_ulimit()

    def signal_handler(*_) -> None:
        # Interrupt server on sigterm while initializing
        raise KeyboardInterrupt("terminated")

    signal.signal(signal.SIGTERM, signal_handler)

    async with build_async_engine_client(args) as engine_client:
        app = build_app(args)

        add_vllm_client_endpoints(app, engine_client)  # This is essentially the only difference from vllm original code

        vllm_config = await engine_client.get_vllm_config()

        await init_app_state(engine_client, vllm_config, app.state, args)

        def _listen_addr(a: str) -> str:
            if is_valid_ipv6_address(a):
                return '[' + a + ']'
            return a or "0.0.0.0"

        is_ssl = args.ssl_keyfile and args.ssl_certfile
        logger.info("Starting vLLM API server on http%s://%s:%d",
                    "s" if is_ssl else "", _listen_addr(sock_addr[0]),
                    sock_addr[1])

        shutdown_task = await serve_http(
            app,
            sock=sock,
            enable_ssl_refresh=args.enable_ssl_refresh,
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            # NOTE: When the 'disable_uvicorn_access_log' value is True,
            # no access log will be output.
            access_log=not args.disable_uvicorn_access_log,
            timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            ssl_ca_certs=args.ssl_ca_certs,
            ssl_cert_reqs=args.ssl_cert_reqs,
            **uvicorn_kwargs,
        )

    # NB: Await server shutdown only after the backend context is exited
    try:
        await shutdown_task
    finally:
        sock.close()

def make_parser(subparsers: argparse._SubParsersAction = None):
    if subparsers is not None:
        parser = subparsers.add_parser("vllm-serve-async", add_help=False, help="Runs vLLM's OpenAI-compatible server with weight syncing.")
    else:
        parser = TrlParser()
    return parser

def main():
    cli_env_setup()
    
    parser = FlexibleArgumentParser(description="vLLM OpenAI-Compatible RESTful API server with weight syncing.")
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)

    # We can use the same worker extension class
    args.worker_extension_cls="verifiers.inference.vllm_serve.WeightSyncWorkerExtension"
    
    # We encounter the same problem as in vllm_serve.py
    if args.tensor_parallel_size == 1 and args.data_parallel_size > 1:
        warnings.warn(
            "Detected configuration: tensor_parallel_size=1 and data_parallel_size>1. This setup is known to "
            "cause a crash when using the `trl vllm-serve` CLI entry point. As a workaround, please run the "
            "server using the module path instead: `python -m trl.scripts.vllm_serve`",
            RuntimeWarning,
        )

    uvloop.run(run_server(args))

if __name__ == "__main__":
    main()

