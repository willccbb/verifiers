import verifiers as vf

import os
import asyncio
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass
from datasets import Dataset
from dotenv import load_dotenv

from verifiers.envs.tool_env import ToolEnv
from verifiers.types import ChatCompletionMessageToolCall, Message, Messages, State
from verifiers.utils.tool_utils import convert_func_to_oai_tool

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool, CallToolResult

from src.mcp_server_connection import MCPServerConnection
from src.mcp_tool_wrapper import MCPToolWrapper
from src.models import MCPServerConfig

load_dotenv()


CHROMA_MCP_SYSTEM_PROMPT = """You are a ChromaDB DBA. \
You are to make tool calls based on user needs.  Don't respond with just text.  Keep calling tools until you have accomplished what is needed.
"""

BRAVE_MCP_SYSTEM_PROMPT = """You are a Web Search Agent. \
You are to make tool calls based on user needs.  Don't respond with just text.  Keep calling tools until you have accomplished what is needed.
"""

FETCH_MCP_SYSTEM_PROMPT = """You are a Web Search Agent with access to a fetch tool that can retrieve web page content. \

When the user asks about a website, use the fetch_fetch tool to retrieve the content. Then answer an user question if need.

You have access to the fetch_fetch tool which takes a 'url' parameter.

Do not respond to the user until after you have made the necessary tool call and got the results needed.
"""

BROWSERBASE_SYSTEM_PROMPT = """You are a Browser Agent that can navigate to urls, observe web pages, perform actions on web pages, extract information from web pages, and even take screenshots.

Use the tools you have available to help the user.  Do not reply with text simply keep making tool calls until you accomplish what the user needed.

"""

PEOPLEBASE_SYSTEM_PROMPT = """You are a Browser Agent that can navigate to urls, observe web pages, perform actions on web pages, extract information from web pages, and even take screenshots.

You also have tools to use a sqlite db including creating tables, writing data, and querying data from the sqlite db. Use this db to store any useful information you find relevant to the users needs.

Use the tools you have available to help the user.  Do not reply with text simply keep making tool calls until you accomplish what the user needed.

DO NOT use Google, they will block you.  If you need to search use Exa.

Prime Intellect website: https://www.primeintellect.ai/

"""

PEOPLEBASE_MCP_TOOLS = [
    {
        "name": "browserbase",
        "command": "npx",
        "args": [
            "-y",
            "@smithery/cli@latest",
            "run",
            "@browserbasehq/mcp-browserbase",
            "--key",
            os.getenv("BROWSERBASE_API_KEY"),
            "--profile",
            os.getenv("SMITHERY_PROFILE")
            
        ]
    },
    {
        "name": "sqlite",
        "command": "uvx",
        "args": [
            "mcp-server-sqlite",
            "--db-path",
            os.getenv("SQLITE_PATH")
        ]
    },
    {
        "name": "exa",
        "command": "npx",
        "args": [
            "-y",
            "@smithery/cli@latest",
            "run",
            "exa",
            "--key",
            os.getenv("EXA_API_KEY"),
            "--profile",
            os.getenv("SMITHERY_PROFILE")
        ]
    }
]

class MCPEnv(ToolEnv):
    """Environment for MCP-based tools using the official MCP SDK."""
    
    def __init__(
        self,
        mcp_servers: List[Union[MCPServerConfig, dict]] = None,
        max_turns: int = 10,
        error_formatter: Callable[[Exception], str] = lambda e: f"Error: {str(e)}",
        ignore_connection_errors: bool = False,
        **kwargs
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
            tools=[],
            max_turns=max_turns,
            error_formatter=error_formatter,
            **kwargs
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
                    wrapper = MCPToolWrapper(
                        server_config.name,
                        tool,
                        connection
                    )
                    
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


def load_environment(mcp_servers=None, **kwargs) -> vf.Environment:
    """Load an MCPEnv environment with fetch server for testing."""
    ds = Dataset.from_dict(
        {
            "question": [
                "find out who the founders of prime intellect are, try to think about what they like, then find some potential new office spaces for their startup in san fransisco they might like. you can store all information you find in the db",
                "check https://www.primeintellect.ai/ and see what the latest announcement was",
                "are there any collections?", 
                "what collections do we have", 
                "how many documents are in the collection"
            ],
            "answer": ["", "", "", "", ""]
        }
    )

    mcp_servers = mcp_servers or PEOPLEBASE_MCP_TOOLS
    
    
    vf_env = MCPEnv(
        mcp_servers=mcp_servers,
        dataset=ds,
        system_prompt=PEOPLEBASE_SYSTEM_PROMPT,
        ignore_connection_errors=kwargs.get("ignore_connection_errors", False),
        **kwargs
    )
    
    return vf_env


# Additional server configurations (commented out)
# {
#     "name": "chroma",
#     "command": "uvx",
#     "args": ["chroma-mcp"],
#     "description": "ChromaDB vector database"
# }
# {
#     "name": "brave-search",
#     "command": "npx",
#     "args": ["-y", "@brave/brave-search-mcp-server", "--transport", "stdio"],
#     "description": "Brave web search API"
# }
#{
#    "name": "fetch",
#    "command": "uvx",
#    "args": ["mcp-server-fetch"],
#    "description": "Fetch MCP server"
#}


#{
#    "name": "browserbase",
#    "command": "npx",
#    "args": [
#        "-y",
#        "@smithery/cli@latest",
#        "run",
#        "@browserbasehq/mcp-browserbase",
#        "--key",
#        os.getenv("BROWSERBASE_API_KEY"),
#        "--profile",
#        os.getenv("SMITHERY_PROFILE")
#        
#    ]
#},
#{
#    "name": "sqlite",
#    "command": "uvx",
#    "args": [
#        "mcp-server-sqlite",
#        "--db-path",
#        os.getenv("SQLITE_PATH")
#    ]
#},
#{
#    "name": "exa",
#    "command": "npx",
#    "args": [
#        "-y",
#        "@smithery/cli@latest",
#        "run",
#        "exa",
#        "--key",
#        os.getenv("EXA_API_KEY"),
#        "--profile",
#        os.getenv("SMITHERY_PROFILE")
#    ]
#}
