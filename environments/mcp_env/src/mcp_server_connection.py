import verifiers as vf

import asyncio
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass
from datasets import Dataset

from verifiers.envs.tool_env import ToolEnv
from verifiers.types import ChatCompletionMessageToolCall, Message, Messages, State
from verifiers.utils.tool_utils import convert_func_to_oai_tool

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool, CallToolResult

from .models import MCPServerConfig



class MCPServerConnection:
    def __init__(self, config: MCPServerConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.session: Optional[ClientSession] = None
        self.tools: Dict[str, Tool] = {}

        self._connection_task: Optional[asyncio.Task] = None
        self._ready = asyncio.Event()
        self._error: Optional[Exception] = None


    async def connect(self):
        self._connection_task = asyncio.create_task(self._get_connection())

        await self._ready.wait()

        if self._error:
            raise self._error

        return self.tools

    async def _get_connection(self):
        try:
            server_params = StdioServerParameters(
                command=self.config.command,
                args=self.config.args or [],
                env=self.config.env
            )

            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    self.session = session

                    await session.initialize()

                    tools_response = await session.list_tools()

                    for tool in tools_response.tools:
                        self.tools[tool.name] = tool

                    self._ready.set()

                    while True:
                        await asyncio.sleep(1)

        except asyncio.CancelledError:
            self.logger.info(f"Connection to '{self.config.name}' cancelled")
            raise
        except Exception as e:
            self._error = e
            self._ready.set() # so connection doesnt hang?
        finally:
            self.session = None
            self.tools = {}

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        if not self.session:
            raise RuntimeError(f"Server '{self.config.name}' not connected")

        try:
            result = await self.session.call_tool(tool_name, arguments=arguments)

            if result.content:
                text_parts = []
                for content_item in result.content:
                    if hasattr(content_item, 'text'):
                        text_parts.append(content_item.text)
                    elif hasattr(content_item, 'type') and content_item.type == 'text':
                        text_parts.append(getattr(content_item, 'text', str(content_item)))
                    else:
                        text_parts.append(str(content_item))

                return "\n".join(text_parts)

            return "No result returned from tool"

        except Exception as e:
            self.logger.error(f"Error calling tool {tool_name}: {e}")

    async def disconnect(self):
        if self._connection_task and not self._connection_task.done():
            self._connection_task.cancel()
            try:
                await self._connection_task
            except asyncio.CancelledError:
                pass

