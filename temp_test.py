import os
import signal
import subprocess
import unittest

import torch
import psutil
import pytest
from transformers import AutoModelForCausalLM

from verifiers.inference.vllm_client import VLLMClient

# 2 gpus minimum
class TestVLLMClientServerAsync(unittest.TestCase):
    model_id = "Qwen/Qwen2.5-1.5B"

    @classmethod
    def setUpClass(cls):
        # We want the server to run on GPU 1, so we set CUDA_VISIBLE_DEVICES to "1"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "1"  # Restrict to GPU 1

        # Start the server process
        cls.server_process = subprocess.Popen(
            ["python", "-m", "verifiers.inference.vllm_serve_async", "--model", cls.model_id, 
             "--enable-auto-tool-choice", "--reasoning_parser", "deepseek_r1", "--tool-call-parser", "hermes"], 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
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

    def test_tool(self):
        """Test tool usage with python calculator tool"""
        # Define a simple Python calculator tool
        tool_definition = {
            "type": "function",
            "function": {
                "name": "python_calculator",
                "description": "Execute a Python expression for mathematical calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "A Python mathematical expression to evaluate"
                        }
                    },
                    "required": ["expression"]
                }
            }
        }

        # Test with a simple math problem that requires calculation
        prompt = "What is 25 * 13 + 47?"
        
        response = self.client.session.post(
            url="http://localhost:8000/v1/chat/completions",
            json={
                "model": self.model_id,
                "messages": [{"role": "user", "content": prompt}],
                "tools": [tool_definition],
                "tool_choice": "auto",
                "max_tokens": 200
            }
        )
        response.raise_for_status()
        response_json = response.json()

        # Check basic response structure
        self.assertIn("choices", response_json)
        self.assertGreater(len(response_json["choices"]), 0)
        
        # Check that we got a message response
        first_choice = response_json["choices"][0]
        self.assertIn("message", first_choice)
        
        # Check if tool calls were made
        message = first_choice["message"]
        if "tool_calls" in message and message["tool_calls"]:
            # Verify tool call structure
            tool_call = message["tool_calls"][0]
            self.assertIn("function", tool_call)
            self.assertEqual(tool_call["function"]["name"], "python_calculator")
            self.assertIn("arguments", tool_call["function"])

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

# 3 gpus minimum
class TestVLLMClientAsyncServerTP(unittest.TestCase):
    model_id = "Qwen/Qwen2.5-1.5B"

    @classmethod
    def setUpClass(cls):
        # We want the server to run on GPU 1 and 2, so we set CUDA_VISIBLE_DEVICES to "1,2"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "1,2"  # Restrict to GPU 1 and 2

        # Start the server process
        cls.server_process = subprocess.Popen(
            ["python", "-m", "verifiers.inference.vllm_serve_async", "--model", cls.model_id, "--tensor_parallel_size", "2",
             "--enable-auto-tool-choice", "--reasoning_parser", "deepseek_r1", "--tool-call-parser", "hermes"],
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

    def test_tool(self):
        """Test tool usage with python calculator tool"""
        # Define a simple Python calculator tool
        tool_definition = {
            "type": "function",
            "function": {
                "name": "python_calculator",
                "description": "Execute a Python expression for mathematical calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "A Python mathematical expression to evaluate"
                        }
                    },
                    "required": ["expression"]
                }
            }
        }

        # Test with a simple math problem that requires calculation
        prompt = "What is 25 * 13 + 47?"
        
        response = self.client.session.post(
            url="http://localhost:8000/v1/chat/completions",
            json={
                "model": self.model_id,
                "messages": [{"role": "user", "content": prompt}],
                "tools": [tool_definition],
                "tool_choice": "auto",
                "max_tokens": 200
            }
        )
        response.raise_for_status()
        response_json = response.json()

        # Check basic response structure
        self.assertIn("choices", response_json)
        self.assertGreater(len(response_json["choices"]), 0)
        
        # Check that we got a message response
        first_choice = response_json["choices"][0]
        self.assertIn("message", first_choice)
        
        # Check if tool calls were made
        message = first_choice["message"]
        if "tool_calls" in message and message["tool_calls"]:
            # Verify tool call structure
            tool_call = message["tool_calls"][0]
            self.assertIn("function", tool_call)
            self.assertEqual(tool_call["function"]["name"], "python_calculator")
            self.assertIn("arguments", tool_call["function"])

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


# data parallel is iffy, 
# # 3 gpus minimum
# class TestVLLMClientAsyncServerDP(unittest.TestCase):
#     model_id = "Qwen/Qwen2.5-1.5B"

#     @classmethod
#     def setUpClass(cls):
#         # We want the server to run on GPU 1 and 2, so we set CUDA_VISIBLE_DEVICES to "1,2"
#         env = os.environ.copy()
#         env["CUDA_VISIBLE_DEVICES"] = "1,2"  # Restrict to GPU 1 and 2

#         # Start the server process
#         cls.server_process = subprocess.Popen(
#             ["python", "-m", "verifiers.inference.vllm_serve_async", "--model", cls.model_id, "--data_parallel_size", "2",
#              "--enable-auto-tool-choice", "--reasoning_parser", "deepseek_r1", "--tool-call-parser", "hermes"],
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             env=env,
#         )

#         # Initialize the client
#         cls.client = VLLMClient(connection_timeout=240)

#     def test_generate(self):
#         prompt = "Hello, AI! Tell me a joke."
#         response = self.client.session.post(
#             url="http://localhost:8000/v1/completions",
#             json={
#                 "model": self.model_id,
#                 "prompt": prompt,
#                 "max_tokens": 50
#             }
#         )
#         response.raise_for_status()
#         response_json = response.json()

#         # Check basic response structure
#         self.assertIn("choices", response_json)
#         self.assertGreater(len(response_json["choices"]), 0)
        
#         # Check that we got a non-empty text response
#         first_choice = response_json["choices"][0]
#         self.assertIn("text", first_choice)
#         self.assertGreater(len(first_choice["text"]), 0)

#     def test_tool(self):
#         """Test tool usage with python calculator tool"""
#         # Define a simple Python calculator tool
#         tool_definition = {
#             "type": "function",
#             "function": {
#                 "name": "python_calculator",
#                 "description": "Execute a Python expression for mathematical calculations",
#                 "parameters": {
#                     "type": "object",
#                     "properties": {
#                         "expression": {
#                             "type": "string",
#                             "description": "A Python mathematical expression to evaluate"
#                         }
#                     },
#                     "required": ["expression"]
#                 }
#             }
#         }

#         # Test with a simple math problem that requires calculation
#         prompt = "What is 25 * 13 + 47?"
        
#         response = self.client.session.post(
#             url="http://localhost:8000/v1/chat/completions",
#             json={
#                 "model": self.model_id,
#                 "messages": [{"role": "user", "content": prompt}],
#                 "tools": [tool_definition],
#                 "tool_choice": "auto",
#                 "max_tokens": 200
#             }
#         )
#         response.raise_for_status()
#         response_json = response.json()

#         # Check basic response structure
#         self.assertIn("choices", response_json)
#         self.assertGreater(len(response_json["choices"]), 0)
        
#         # Check that we got a message response
#         first_choice = response_json["choices"][0]
#         self.assertIn("message", first_choice)
        
#         # Check if tool calls were made
#         message = first_choice["message"]
#         if "tool_calls" in message and message["tool_calls"]:
#             # Verify tool call structure
#             tool_call = message["tool_calls"][0]
#             self.assertIn("function", tool_call)
#             self.assertEqual(tool_call["function"]["name"], "python_calculator")
#             self.assertIn("arguments", tool_call["function"])

#     def test_update_model_params(self):
#         model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map="cuda")
#         self.client.update_model_params(model)

#     def test_reset_prefix_cache(self):
#         # Test resetting the prefix cache
#         self.client.reset_prefix_cache()

#     @classmethod
#     def tearDownClass(cls):
#         super().tearDownClass()

#         # Close the client
#         cls.client.close_communicator()

#         # vLLM x pytest (or Popen) seems not to handle process termination well. To avoid zombie processes, we need to
#         # kill the server process and its children explicitly.
#         parent = psutil.Process(cls.server_process.pid)
#         children = parent.children(recursive=True)
#         for child in children:
#             child.send_signal(signal.SIGTERM)
#         cls.server_process.terminate()
#         cls.server_process.wait()
