"""Tests for SandboxEnv and PythonEnv using a mocked sandbox client."""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any

import pytest
from datasets import Dataset
from unittest.mock import AsyncMock

from verifiers.parsers.parser import Parser
from verifiers.rubrics.rubric import Rubric


@pytest.fixture
def mock_prime_cli(monkeypatch):
    sandbox_module = types.ModuleType("prime_cli.api.sandbox")

    class FakeCreateSandboxRequest:
        def __init__(self, name: str, docker_image: str, start_command: str):
            self.name = name
            self.docker_image = docker_image
            self.start_command = start_command

    class FakeSandboxClient:
        def __init__(self):
            self.create = AsyncMock(return_value=types.SimpleNamespace(id="sandbox-123"))
            self.wait_for_creation = AsyncMock()
            self.execute_command = AsyncMock(
                return_value=types.SimpleNamespace(stdout="ok", stderr="")
            )
            self.delete = AsyncMock()

    sandbox_module.AsyncSandboxClient = FakeSandboxClient
    sandbox_module.CreateSandboxRequest = FakeCreateSandboxRequest

    api_module = types.ModuleType("prime_cli.api")
    api_module.sandbox = sandbox_module

    prime_module = types.ModuleType("prime_cli")
    prime_module.api = api_module

    monkeypatch.setitem(sys.modules, "prime_cli", prime_module)
    monkeypatch.setitem(sys.modules, "prime_cli.api", api_module)
    monkeypatch.setitem(sys.modules, "prime_cli.api.sandbox", sandbox_module)

    return sandbox_module


def _reload_env_module(module_name: str) -> Any:
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)
    return importlib.reload(module)


@pytest.fixture
def sandbox_env_module(mock_prime_cli):
    return _reload_env_module("verifiers.envs.sandbox_env")


@pytest.fixture
def python_env_module(mock_prime_cli):
    _reload_env_module("verifiers.envs.sandbox_env")
    return _reload_env_module("verifiers.envs.python_env")


def _make_dataset() -> Dataset:
    return Dataset.from_dict(
        {
            "prompt": [[{"role": "user", "content": "hi"}]],
            "answer": [""],
        }
    )


@pytest.mark.asyncio
async def test_sandbox_env_setup_and_cleanup(
    sandbox_env_module, mock_openai_client, sample_chat_dataset
):
    SandboxEnv = sandbox_env_module.SandboxEnv

    env = SandboxEnv(
        client=mock_openai_client,
        model="test-model",
        dataset=sample_chat_dataset,
        parser=Parser(),
        rubric=Rubric(),
        max_turns=1,
    )

    state: dict[str, Any] = {"turn": 0}
    state = await env.setup_state(state)
    assert state["sandbox_id"] == "sandbox-123"

    updated_args = env.update_tool_args("bash", {"command": "ls"}, [], state)
    assert updated_args["sandbox_id"] == "sandbox-123"

    env.sandbox_client.delete.assert_not_called()

    state["turn"] = 1
    messages = [{"role": "assistant", "content": "done"}]
    completed = await env.is_completed(messages, state)
    assert completed is True
    env.sandbox_client.delete.assert_awaited_once_with("sandbox-123")
    assert "sandbox_id" not in state


@pytest.mark.asyncio
async def test_python_env_execution_flow(
    python_env_module, mock_openai_client
):
    PythonEnv = python_env_module.PythonEnv

    env = PythonEnv(
        client=mock_openai_client,
        model="test-model",
        dataset=_make_dataset(),
        parser=Parser(),
        rubric=Rubric(),
        max_turns=2,
    )

    state: dict[str, Any] = {"turn": 0}
    state = await env.setup_state(state)
    sandbox_id = state["sandbox_id"]
    python_state = state["python_env"]
    assert python_state == {"ready": False, "execution_count": 0}

    env._wait_for_worker_ready = AsyncMock()
    env._send_worker_request = AsyncMock(
        return_value={
            "status": "ok",
            "stdout": "",
            "stderr": "",
            "result": "42",
            "execution_count": 1,
        }
    )

    result = await env.python("6 * 7", sandbox_id=sandbox_id, python_state=python_state)
    assert "Out[1]: 42" in result
    assert python_state["ready"] is True
    assert python_state["execution_count"] == 1
    env._wait_for_worker_ready.assert_awaited_once()
    env._send_worker_request.assert_awaited_once()

    env._send_worker_request.reset_mock()
    env._send_worker_request.return_value = {
        "status": "ok",
        "stdout": "hello\n",
        "stderr": "",
        "result": None,
    }

    result_again = await env.python(
        "print(\"hello\")", sandbox_id=sandbox_id, python_state=python_state
    )
    assert result_again.strip() == "hello"
    assert env._wait_for_worker_ready.await_count == 1

    args = env.update_tool_args("python", {"code": "1+1"}, [], state)
    assert args["sandbox_id"] == sandbox_id
    assert args["python_state"] is python_state

    state["turn"] = env.max_turns
    messages = [{"role": "assistant", "content": "done"}]
    completed = await env.is_completed(messages, state)
    assert completed is True
    env.sandbox_client.delete.assert_awaited()
    assert "python_env" not in state
