import asyncio
import os
from typing import Callable, Dict, List

from datasets import Dataset
from dotenv import load_dotenv
from src.mcp_server_connection import MCPServerConnection
from src.mcp_tool_wrapper import MCPToolWrapper
from src.models import MCPServerConfig

import verifiers as vf
from verifiers.envs.tool_env import ToolEnv
from verifiers.types import Message, State

load_dotenv()

FETCH_MCP_SYSTEM_PROMPT = """You are a Web Search Agent with access to a fetch tool that can retrieve web page content and Exa a search tool. \

Do not respond to the user until after you have made the necessary tool call and got the results needed.
"""

EXA_FETCH_TOOLS = [
    {
        "name": "exa",
        "command": "npx",
        "args": [
            "-y",
            "mcp-remote",
            f"https://mcp.exa.ai/mcp?exaApiKey={os.getenv('EXA_API_KEY')}",
        ],
        "description": "Exa MCP server",
    },
    {
        "name": "fetch",
        "command": "uvx",
        "args": ["mcp-server-fetch"],
        "description": "Fetch MCP server",
    },
]


class MCPEnv(ToolEnv):
    """Environment for MCP-based tools using the official MCP SDK."""

    def __init__(
        self,
        mcp_servers: List[MCPServerConfig] = [],
        max_turns: int = 10,
        error_formatter: Callable[[Exception], str] = lambda e: f"Error: {str(e)}",
        ignore_connection_errors: bool = False,
        **kwargs,
    ):
        self.mcp_servers = []
        if mcp_servers:
            for server in mcp_servers:
                if isinstance(server, dict):
                    self.mcp_servers.append(MCPServerConfig(**server))
                else:
                    self.mcp_servers.append(server)

        self.server_connections: Dict[str, MCPServerConnection] = {}
        self.mcp_tools: Dict[str, MCPToolWrapper] = {}

        self.error_formatter = error_formatter
        self.ignore_connection_errors = ignore_connection_errors
        self._setup_complete = False
        self._init_kwargs = kwargs
        self._max_turns = max_turns

        super().__init__(
            tools=[], max_turns=max_turns, error_formatter=error_formatter, **kwargs
        )

    async def setup_state(self, state: State, **kwargs) -> State:
        if not self._setup_complete:
            await self._connect_servers()
            self._setup_complete = True

        if self.oai_tools:
            state["info"]["oai_tools"] = self.oai_tools

        return await super().setup_state(state, **kwargs)

    async def _connect_servers(self):
        wrapper_tools = []

        for server_config in self.mcp_servers:
            try:
                # Create connection
                connection = MCPServerConnection(server_config, self.logger)

                # Connect to server
                tools = await connection.connect()

                # Store connection
                self.server_connections[server_config.name] = connection

                # Create wrappers for each tool
                for tool_name, tool in tools.items():
                    wrapper = MCPToolWrapper(server_config.name, tool, connection)

                    wrapper_tools.append(wrapper)
                    self.mcp_tools[wrapper.__name__] = wrapper

                    self.logger.info(
                        f"Registered MCP tool: {wrapper.__name__} from server '{server_config.name}'"
                    )

            except Exception as e:
                self.logger.error(
                    f"Failed to connect to MCP server '{server_config.name}': {e}"
                )
                if not self.ignore_connection_errors:
                    raise

        self.tools = wrapper_tools
        self.oai_tools = [tool.to_oai_tool() for tool in wrapper_tools]
        self.tool_map = {tool.__name__: tool for tool in wrapper_tools}

    async def call_tool(
        self, tool_name: str, tool_args: dict, tool_call_id: str, **kwargs
    ) -> Message:
        """Call a tool and return the result as a Message."""
        try:
            if tool_name in self.tool_map:
                tool_wrapper = self.tool_map[tool_name]
                result = await tool_wrapper(**tool_args)
                return {
                    "role": "tool",
                    "content": str(result),
                    "tool_call_id": tool_call_id,
                }
            else:
                raise ValueError(f"Tool '{tool_name}' not found")

        except Exception as e:
            return {
                "role": "tool",
                "content": self.error_formatter(e),
                "tool_call_id": tool_call_id,
            }

    async def cleanup(self):
        for server_name, connection in self.server_connections.items():
            try:
                await connection.disconnect()
            except Exception as e:
                self.logger.error(f"Error cleaning up server '{server_name}': {e}")

        self.server_connections.clear()
        self.mcp_tools.clear()

    def __del__(self):
        if hasattr(self, "server_connections") and self.server_connections:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Schedule cleanup as a task
                    loop.create_task(self.cleanup())
                else:
                    # Run cleanup synchronously
                    loop.run_until_complete(self.cleanup())
            except RuntimeError:
                if hasattr(self, "logger"):
                    self.logger.warning(
                        "Unable to clean up MCP connections - no event loop available"
                    )
            except Exception as e:
                if hasattr(self, "logger"):
                    self.logger.error(f"Error during cleanup in destructor: {e}")


def load_environment(mcp_servers: list = [], dataset=None, **kwargs) -> vf.Environment:
    """Load an MCPEnv environment with fetch server for testing."""
    ds = dataset or Dataset.from_dict(
        {
            "question": [
                "Find out what Prime Intellect's newest announcement was from their website, give me the headline in 2 words. Their url is primeintellect.ai",
            ],
            "answer": ["ENVIRONMENTS HUB"],
        }
    )

    mcp_servers = mcp_servers or EXA_FETCH_TOOLS

    vf_env = MCPEnv(
        mcp_servers=mcp_servers,
        dataset=ds,
        system_prompt=FETCH_MCP_SYSTEM_PROMPT,
        ignore_connection_errors=kwargs.get("ignore_connection_errors", False),
        **kwargs,
    )

    return vf_env
