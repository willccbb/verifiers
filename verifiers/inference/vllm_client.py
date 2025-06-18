# adapted from trl/extras/vllm_client.py (huggingface/trl)

import atexit
import logging
import time
from typing import Optional

import requests
from requests import ConnectionError
from requests.adapters import HTTPAdapter
from openai import OpenAI
import torch
from torch import nn
from trl.import_utils import is_requests_available, is_vllm_available

from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator # type: ignore
from vllm.distributed.utils import StatelessProcessGroup # type: ignore

logger = logging.getLogger(__name__)


class VLLMClient(OpenAI):
    """
    A client class to interact with a vLLM server.

    This class provides methods to generate completions, initialize and manage weight update groups, and update model
    weights in a distributed setting. Before using it, start the vLLM server with `trl vllm-serve`.

    Args:
        host (`str`, *optional*, defaults to `"0.0.0.0"`):
            IP address of the vLLM server.
        server_port (`int`, *optional*, defaults to `8000`):
            Port number of the vLLM server.
        group_port (`int`, *optional*, defaults to `51216`):
            Port number for the weight update group.
        connection_timeout (`float`, *optional*, defaults to `0.0`):
            Total timeout duration in seconds to wait for the server to be up. If the server is not up after the
            timeout, a `ConnectionError` is raised.

    Examples:
        Run the vLLM server with the model `Qwen/Qwen2.5-7B`:

        ```
        $ trl vllm-serve --model Qwen/Qwen2.5-7B
        ...
        INFO:     Application startup complete.
        INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
        ```

        Use the client to generate completions and update model weights:

        ```python
        >>> from trl.extras.vllm_client import VLLMClient
        >>> client = VLLMClient()
        >>> client.generate(["Hello, AI!", "Tell me a joke"])
        [[2980, 498, 1492, 752, 448, 264, 13027, 8645, 30, 358, 2776, 4460, 311, 3270, 264, 2025],
         [911, 7988, 1251, 382, 3838, 653, 498, 1618, 4325, 879, 2581, 20027, 264, 21428, 30, 362]]

        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", device_map="cuda")
        >>> client.update_model_params(model)
        ```
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        group_port: int = 51216, connection_timeout: float = 0.0
    ):
        if not is_requests_available():
            raise ImportError("requests is not installed. Please install it with `pip install requests`.")
        if not is_vllm_available():
            raise ImportError("vLLM is not installed. Please install it with `pip install vllm`.")

        super().__init__(base_url=f"http://{host}:{port}/v1", api_key="local")
        self.session = requests.Session()
        # Configure connection pooling to handle rapid requests better
        adapter = HTTPAdapter(
            pool_connections=10,
            pool_maxsize=10,
            max_retries=3,
            pool_block=False
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
        self.host = host
        self.server_port = port # Renamed from server_port to port to match super init
        self.group_port = group_port
        self.check_server(connection_timeout)  # check server and fail after timeout

    def check_server(self, total_timeout: float = 0.0, retry_interval: float = 2.0):
        """
        Check server availability with retries on failure, within a total timeout duration. If the server is not up
        after the total timeout duration, raise a `ConnectionError`.

        Args:
            retry_interval (`float`, *optional*, defaults to `2.0`):
                Interval in seconds between retries.
            total_timeout (`float`, *optional*, defaults to `0.0`):
                Total timeout duration in seconds.
        """
        url = f"http://{self.host}:{self.server_port}/health/"
        start_time = time.time()  # Record the start time

        while True:
            try: 
                response = requests.get(url) # type: ignore
            except requests.exceptions.RequestException as exc: # type: ignore
                # Check if the total timeout duration has passed
                elapsed_time = time.time() - start_time
                if elapsed_time >= total_timeout:
                    raise ConnectionError( # type: ignore
                        f"The vLLM server can't be reached at {self.host}:{self.server_port} after {total_timeout} "
                        "seconds. Make sure the server is running by running `trl vllm-serve`."
                    ) from exc
            else:
                if response.status_code == 200:
                    logger.info("Server is up!")
                    return None

            # Retry logic: wait before trying again
            logger.info(f"Server is not up yet. Retrying in {retry_interval} seconds...")
            time.sleep(retry_interval)

    def init_communicator(self):
        """
        Initializes the weight update group in a distributed setup for model synchronization.
        """
        logger.info(f"[VLLM_CLIENT] Starting init_communicator")
        
        # Get the world size from the server
        url = f"http://{self.host}:{self.server_port}/get_world_size/"
        logger.info(f"[VLLM_CLIENT] Getting world size from {url}")
        try:
            response = requests.get(url)
            logger.info(f"[VLLM_CLIENT] World size response: status={response.status_code}")
        except Exception as e:
            logger.error(f"[VLLM_CLIENT] Failed to get world size: {e}")
            raise
            
        if response.status_code == 200:
            vllm_world_size = response.json()["world_size"]
            logger.info(f"[VLLM_CLIENT] vLLM world size: {vllm_world_size}")
        else:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

        world_size = vllm_world_size + 1  # add the client to the world
        self.rank = vllm_world_size  # the client's rank is the last process
        logger.info(f"[VLLM_CLIENT] Client rank: {self.rank}, total world size: {world_size}")

        # Initialize weight update group
        url = f"http://{self.host}:{self.server_port}/init_communicator/"
        logger.info(f"[VLLM_CLIENT] Sending init_communicator request to {url}")
        # In the server side, the host is set to 0.0.0.0
        try:
            response = self.session.post(url, json={"host": "0.0.0.0", "port": self.group_port, "world_size": world_size})
            logger.info(f"[VLLM_CLIENT] Init communicator response: status={response.status_code}")
        except Exception as e:
            logger.error(f"[VLLM_CLIENT] Failed to init communicator: {e}")
            raise
            
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

        # Brief delay to allow server initialization. While not strictly required (client socket will retry on
        # connection failure), this prevents log warnings like:
        # [W416 23:24:57.460001114 socket.cpp:204] [c10d] The hostname of the client socket cannot be retrieved. err=-3
        time.sleep(0.1)

        # Set up the communication group for weight broadcasting
        pg = StatelessProcessGroup.create(host=self.host, port=self.group_port, rank=self.rank, world_size=world_size)
        # Use device 0 like the old code - this seems to work better for multi-GPU setups
        device = 0
        logger.info(f"[VLLM_CLIENT] Initializing PyNcclCommunicator on device {device}, rank {self.rank}, world_size {world_size}")
        self.pynccl_comm = PyNcclCommunicator(pg, device=device)

        # When the client object is deleted, close the weight update group
        atexit.register(self.close_communicator)

    def update_named_param(self, name: str, weights: torch.Tensor):
        """
        Updates a specific named parameter in the model and broadcasts it to other processes.

        Args:
            name (`str`):
                Name of the layer whose weights are being updated.
            weights (`torch.Tensor`):
                Tensor containing the updated weights.
        """
        dtype, shape = str(weights.dtype), tuple(weights.shape)
        url = f"http://{self.host}:{self.server_port}/update_named_param/"
        logger.debug(f"[VLLM_CLIENT] Sending weight update request for {name}")
        
        # Add timeout to prevent hanging on HTTP request
        try:
            response = self.session.post(url, json={"name": name, "dtype": dtype, "shape": shape}, timeout=300.0)
        except requests.exceptions.Timeout:
            logger.error(f"[VLLM_CLIENT] Timeout waiting for server response for {name} after 300s")
            raise Exception(f"Request timeout for {name} after 300s")
        except Exception as e:
            logger.error(f"[VLLM_CLIENT] Error sending request for {name}: {e}")
            raise
            
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")
        logger.debug(f"[VLLM_CLIENT] Server responded, starting NCCL broadcast for {name}")

        # Broadcast the weights to the other processes
        self.pynccl_comm.broadcast(weights, src=self.rank)
        logger.debug(f"[VLLM_CLIENT] NCCL broadcast complete, waiting at barrier for {name}")
        self.pynccl_comm.group.barrier()
        logger.debug(f"[VLLM_CLIENT] Barrier passed for {name}")

    def update_model_params(self, model: nn.Module):
        """
        Updates all parameters of the given model by calling `update_named_param` for each parameter in the model.

        Args:
            model (`nn.Module`):
                Model whose parameters (weights/biases) are to be updated.
        """
        for name, param in model.named_parameters():
            # Update each parameter individually
            self.update_named_param(name, param.data)

    def batch_update_model_params(self, model: nn.Module, batch_size: int = 50):
        """
        Updates all parameters of the given model in batches to reduce overhead and prevent overwhelming the server.
        
        This method coordinates with the server to ensure proper NCCL synchronization:
        1. Send batch of parameter metadata to server
        2. Server notifies workers for each parameter
        3. Client broadcasts each parameter via NCCL after server confirmation
        
        Args:
            model (`nn.Module`):
                Model whose parameters (weights/biases) are to be updated.
            batch_size (`int`, *optional*, defaults to 50):
                Number of parameters to update in each batch.
        """
        # Collect all parameters
        all_params = list(model.named_parameters())
        total_params = len(all_params)
        
        logger.info(f"[VLLM_CLIENT] Starting batch update of {total_params} parameters in batches of {batch_size}")
        
        # Process in batches
        for batch_idx, i in enumerate(range(0, total_params, batch_size)):
            batch_params = all_params[i:i + batch_size]
            
            # Prepare batch update request
            batch_updates = []
            for name, param in batch_params:
                batch_updates.append({
                    "name": name,
                    "dtype": str(param.data.dtype),
                    "shape": list(param.data.shape)
                })
            
            # Send batch update request
            url = f"http://{self.host}:{self.server_port}/batch_update_named_params/"
            logger.debug(f"[VLLM_CLIENT] Sending batch {batch_idx + 1} with {len(batch_updates)} parameters")
            
            try:
                response = self.session.post(url, json={"updates": batch_updates}, timeout=600.0)
                if response.status_code not in [200, 207]:  # 207 is Multi-Status
                    raise Exception(f"Batch request failed: {response.status_code}, {response.text}")
                    
                result = response.json()
                
                # Check for partial failures
                if response.status_code == 207:
                    logger.warning(f"[VLLM_CLIENT] Batch had errors: {result.get('errors', [])}")
                
                # Get list of successfully notified parameters
                successful_params = result.get('successful', [])
                if not successful_params:
                    logger.error(f"[VLLM_CLIENT] No successful parameters in batch response")
                    continue
                    
            except requests.exceptions.Timeout:
                logger.error(f"[VLLM_CLIENT] Timeout waiting for batch response after 600s")
                raise Exception(f"Batch request timeout after 600s")
            except Exception as e:
                logger.error(f"[VLLM_CLIENT] Error sending batch request: {e}")
                raise
            
            # Now broadcast weights for successfully notified parameters
            logger.debug(f"[VLLM_CLIENT] Broadcasting weights for {len(successful_params)} parameters in batch {batch_idx + 1}")
            
            for name, param in batch_params:
                if name in successful_params:
                    try:
                        # Broadcast this specific parameter
                        self.pynccl_comm.broadcast(param.data, src=self.rank)
                        self.pynccl_comm.group.barrier()
                        logger.debug(f"[VLLM_CLIENT] Broadcast complete for {name}")
                    except Exception as e:
                        logger.error(f"[VLLM_CLIENT] Failed to broadcast {name}: {e}")
                        raise
                else:
                    logger.warning(f"[VLLM_CLIENT] Skipping broadcast for {name} - not in successful list")
            
            logger.debug(f"[VLLM_CLIENT] Completed batch {batch_idx + 1}")
        
        logger.info(f"[VLLM_CLIENT] Batch update complete for {total_params} parameters")

    def reset_prefix_cache(self):
        """
        Resets the prefix cache for the model.
        """
        url = f"http://{self.host}:{self.server_port}/reset_prefix_cache/"
        response = self.session.post(url)
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")

    def close_communicator(self):
        """
        Closes the weight update group and cleans up the communication group.
        """
        url = f"http://{self.host}:{self.server_port}/close_communicator/"

        try:
            response = self.session.post(url)
        except ConnectionError:
            # The server might be already down, so we don't need to close the communicator
            pass
        else:
            if response.status_code != 200:
                raise Exception(f"Request failed: {response.status_code}, {response.text}")
