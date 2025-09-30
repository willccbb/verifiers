from typing import Any

import verifiers as vf

try:
    from prime_cli.api.sandbox import (  # type: ignore[import-untyped]
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
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sandbox_name = sandbox_name
        self.docker_image = docker_image
        self.start_command = start_command
        self.sandbox_client = AsyncSandboxClient()
        self.sandbox_request = CreateSandboxRequest(
            name=sandbox_name,
            docker_image=docker_image,
            start_command=start_command,
        )

        self.add_tool(self.bash, args_to_skip=["sandbox_id"])

    async def bash(self, command: str, sandbox_id: str) -> str:
        """Execute `command` inside persistent sandbox container."""
        # sandbox_id is passed via update_tool_args, not seen by model
        await self.sandbox_client.wait_for_creation(
            sandbox_id
        )  # wait for sandbox to be created
        results = await self.sandbox_client.execute_command(sandbox_id, command)
        stdout = results.stdout.strip()
        stderr = (results.stderr or "").strip()
        combined = stdout
        if stderr:
            if combined:
                combined = f"{combined}\nstderr:\n{stderr}"
            else:
                combined = f"stderr:\n{stderr}"
        return combined or "(no output)"

    async def _destroy_sandbox(self, sandbox_id: str | None) -> None:
        if sandbox_id is None:
            return
        try:
            await self.sandbox_client.delete(sandbox_id)
        except Exception as e:
            self.logger.warning("Failed to delete sandbox %s: %s", sandbox_id, e)

    async def setup_state(self, state: vf.State, **kwargs) -> vf.State:
        """Create per-rollout sandbox"""
        request = CreateSandboxRequest(
            name=self.sandbox_name,
            docker_image=self.docker_image,
            start_command=self.start_command,
        )
        sandbox = await self.sandbox_client.create(request)
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
