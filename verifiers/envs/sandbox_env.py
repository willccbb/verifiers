import time
from asyncio import Semaphore
from typing import Any

import verifiers as vf

try:
    from prime_cli.api.sandbox import (  # type: ignore[import-untyped]
        AdvancedConfigs,
        AsyncSandboxClient,
        CreateSandboxRequest,
    )
except ImportError:
    raise ImportError(
        "prime-cli is not installed. Please install it with `uv pip install prime`."
    )


class SandboxEnv(vf.StatefulToolEnv):
    def __init__(
        self,
        sandbox_name: str = "sandbox-env",
        docker_image: str = "python:3.11-slim",
        start_command: str = "tail -f /dev/null",
        cpu_cores: int = 1,
        memory_gb: int = 2,
        disk_size_gb: int = 5,
        gpu_count: int = 0,
        timeout_minutes: int = 60,
        environment_vars: dict[str, str] | None = None,
        team_id: str | None = None,
        advanced_configs: AdvancedConfigs | None = None,
        max_concurrent_sandboxes: int = 32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sandbox_client = AsyncSandboxClient()
        self.sandbox_request = CreateSandboxRequest(
            name=sandbox_name,
            docker_image=docker_image,
            start_command=start_command,
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            disk_size_gb=disk_size_gb,
            gpu_count=gpu_count,
            timeout_minutes=timeout_minutes,
            environment_vars=environment_vars,
            team_id=team_id,
            advanced_configs=advanced_configs,
        )
        self.logger.info(f"Using {max_concurrent_sandboxes} max concurrent sandboxes")
        self.sandbox_semaphore = Semaphore(max_concurrent_sandboxes)

        self.add_tool(self.bash, args_to_skip=["sandbox_id"])

    async def bash(self, command: str, sandbox_id: str) -> str:
        """Execute `command` inside persistent sandbox container."""
        # sandbox_id is passed via update_tool_args, not seen by model
        s = time.time()
        await self.sandbox_client.wait_for_creation(
            sandbox_id
        )  # wait for sandbox to be created
        self.logger.debug(f"Waited {time.time() - s:.1f}s for sandbox to be ready")
        s = time.time()
        self.logger.debug(f"Executing command {command} in sandbox {sandbox_id}")
        results = await self.sandbox_client.execute_command(sandbox_id, command)
        e = time.time()
        stdout = results.stdout.strip()
        stderr = (results.stderr or "").strip()
        combined = stdout
        if stderr:
            if combined:
                combined = f"{combined}\nstderr:\n{stderr}"
            else:
                combined = f"stderr:\n{stderr}"
        output = combined or "(no output)"
        self.logger.debug(
            f"Executed command in {time.time() - s:.1f}s. Got output: {output}"
        )
        return output

    async def _destroy_sandbox(self, sandbox_id: str | None) -> None:
        if sandbox_id is None:
            return
        try:
            await self.sandbox_client.delete(sandbox_id)
            self.sandbox_semaphore.release()
            self.logger.debug(f"Deleted sandbox {sandbox_id}")
        except Exception as e:
            self.logger.warning(f"Failed to delete sandbox {sandbox_id}: {e}")

    async def setup_state(self, state: vf.State, **kwargs) -> vf.State:
        """Create per-rollout sandbox"""
        await self.sandbox_semaphore.acquire()
        sandbox = await self.sandbox_client.create(self.sandbox_request)
        self.logger.debug(f"Created sandbox {sandbox.id}")
        state["sandbox_id"] = sandbox.id
        return await super().setup_state(state, **kwargs)

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        messages: vf.Messages,
        state: vf.State,
        **kwargs,
    ) -> dict[str, Any]:
        if tool_name == "bash":
            updated_args = dict(tool_args)
            updated_args["sandbox_id"] = state["sandbox_id"]
            return updated_args
        else:
            return tool_args

    async def is_completed(
        self, messages: vf.Messages, state: vf.State, **kwargs
    ) -> bool:
        """
        When overriding, if sandbox state is needed for reward functions,
        run computation here and cache the result in state.
        """
        completed = await super().is_completed(messages, state, **kwargs)
        if completed:
            await self._destroy_sandbox(state.pop("sandbox_id"))
        return completed
