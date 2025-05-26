import os
import signal
import subprocess
import unittest

import torch
import psutil
import pytest
from transformers import AutoModelForCausalLM

from verifiers.inference.vllm_serve_async import VLLMClient


def require_torch_multi_gpu(test_case):
    return unittest.skipUnless(torch.cuda.device_count() > 1, "test requires multiple CUDA GPUs")(test_case)

def require_3_gpus(test_case):
    return unittest.skipUnless(torch.cuda.device_count() > 3, "test requires at least 3 GPUs")(test_case)


@pytest.mark.slow
@require_torch_multi_gpu
class TestVLLMClientServerAsync(unittest.TestCase):
    model_id = "Qwen/Qwen2.5-1.5B"

    @classmethod
    def setUpClass(cls):
        # We want the server to run on GPU 1, so we set CUDA_VISIBLE_DEVICES to "1"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "1"  # Restrict to GPU 1

        # Start the server process
        cls.server_process = subprocess.Popen(
            ["trl", "vllm-serve-async", "--model", cls.model_id], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
        )

        #Initialize the client
        cls.client = VLLMClient(connection_timeout=240)
        cls.client.init_communicator()

    def test_generate(self):
        prompt = "Hello, AI! Tell me a joke."
        response = self.client.session.post(
            url="http://localhost:8000/v1/completions",
            json={
                "model": self.model_id,
                "prompt": prompt,
                "max_tokens": 50
            }
        )
        response.raise_for_status()
        response_json = response.json()

        # Check basic response structure
        self.assertIn("choices", response_json)
        self.assertGreater(len(response_json["choices"]), 0)
        
        # Check that we got a non-empty text response
        first_choice = response_json["choices"][0]
        self.assertIn("text", first_choice)
        self.assertGreater(len(first_choice["text"]), 0)

    def test_update_model_params(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map="cuda")
        self.client.update_model_params(model)

    def test_funny(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map="cuda")
        for name, param in model.named_parameters():
            self.client.update_named_param(name, torch.randn_like(param.data))

    def test_reset_prefix_cache(self):
        # Test resetting the prefix cache
        self.client.reset_prefix_cache()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

        # Close the client
        cls.client.close_communicator()

        # vLLM x pytest (or Popen) seems not to handle process termination well. To avoid zombie processes, we need to
        # kill the server process and its children explicitly.
        parent = psutil.Process(cls.server_process.pid)
        children = parent.children(recursive=True)
        for child in children:
            child.send_signal(signal.SIGTERM)
        cls.server_process.terminate()
        cls.server_process.wait()


@pytest.mark.slow
@require_3_gpus
class TestVLLMClientAsyncServerTP(unittest.TestCase):
    model_id = "Qwen/Qwen2.5-1.5B"

    @classmethod
    def setUpClass(cls):
        # We want the server to run on GPU 1 and 2, so we set CUDA_VISIBLE_DEVICES to "1,2"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "1,2"  # Restrict to GPU 1 and 2

        # Start the server process
        cls.server_process = subprocess.Popen(
            ["trl", "vllm-serve-async", "--model", cls.model_id, "--tensor_parallel_size", "2"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

        # Initialize the client
        cls.client = VLLMClient(connection_timeout=240)
        cls.client.init_communicator()

    def test_generate(self):
        prompt = "Hello, AI! Tell me a joke."
        response = self.client.session.post(
            url="http://localhost:8000/v1/completions",
            json={
                "model": self.model_id,
                "prompt": prompt,
                "max_tokens": 50
            }
        )
        response.raise_for_status()
        response_json = response.json()

        # Check basic response structure
        self.assertIn("choices", response_json)
        self.assertGreater(len(response_json["choices"]), 0)
        
        # Check that we got a non-empty text response
        first_choice = response_json["choices"][0]
        self.assertIn("text", first_choice)
        self.assertGreater(len(first_choice["text"]), 0)

    def test_update_model_params(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map="cuda")
        self.client.update_model_params(model)

    def test_reset_prefix_cache(self):
        # Test resetting the prefix cache
        self.client.reset_prefix_cache()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

        # Close the client
        cls.client.close_communicator()

        # vLLM x pytest (or Popen) seems not to handle process termination well. To avoid zombie processes, we need to
        # kill the server process and its children explicitly.
        parent = psutil.Process(cls.server_process.pid)
        children = parent.children(recursive=True)
        for child in children:
            child.send_signal(signal.SIGTERM)
        cls.server_process.terminate()
        cls.server_process.wait()


@pytest.mark.slow
@require_3_gpus
class TestVLLMClientAsyncServerDP(unittest.TestCase):
    model_id = "Qwen/Qwen2.5-1.5B"

    @classmethod
    def setUpClass(cls):
        # We want the server to run on GPU 1 and 2, so we set CUDA_VISIBLE_DEVICES to "1,2"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "1,2"  # Restrict to GPU 1 and 2

        # Start the server process
        cls.server_process = subprocess.Popen(
            ["trl", "vllm-serve-async", "--model", cls.model_id, "--data_parallel_size", "2"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

        # Initialize the client
        cls.client = VLLMClient(connection_timeout=240)

    def test_generate(self):
        prompt = "Hello, AI! Tell me a joke."
        response = self.client.session.post(
            url="http://localhost:8000/v1/completions",
            json={
                "model": self.model_id,
                "prompt": prompt,
                "max_tokens": 50
            }
        )
        response.raise_for_status()
        response_json = response.json()

        # Check basic response structure
        self.assertIn("choices", response_json)
        self.assertGreater(len(response_json["choices"]), 0)
        
        # Check that we got a non-empty text response
        first_choice = response_json["choices"][0]
        self.assertIn("text", first_choice)
        self.assertGreater(len(first_choice["text"]), 0)

    def test_update_model_params(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map="cuda")
        self.client.update_model_params(model)

    def test_reset_prefix_cache(self):
        # Test resetting the prefix cache
        self.client.reset_prefix_cache()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

        # Close the client
        cls.client.close_communicator()

        # vLLM x pytest (or Popen) seems not to handle process termination well. To avoid zombie processes, we need to
        # kill the server process and its children explicitly.
        parent = psutil.Process(cls.server_process.pid)
        children = parent.children(recursive=True)
        for child in children:
            child.send_signal(signal.SIGTERM)
        cls.server_process.terminate()
        cls.server_process.wait()
