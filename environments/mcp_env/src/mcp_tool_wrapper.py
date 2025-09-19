import verifiers as vf

import asyncio
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass
from datasets import Dataset
from contextlib import AsyncExitStack

from verifiers.envs.tool_env import ToolEnv
from verifiers.types import ChatCompletionMessageToolCall, Message, Messages, State
from verifiers.utils.tool_utils import convert_func_to_oai_tool

# Official MCP SDK imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool, CallToolResult

from .mcp_server_connection import MCPServerConnection
from .models import MCPServerConfig


class MCPToolWrapper:
    def __init__(
        self,
        server_name: str,
        tool: Tool,
        server_connection: MCPServerConnection
    ):
        self.server_name = server_name
        self.tool = tool
        self.server_connection = server_connection

        self.__name__ = f"{server_name}_{tool.name}"
        self.__doc__ = tool.description or ""

        self.__annotations__ = self._build_annotations()

    def _build_annotations(self) -> dict:
        annotations = {}

        if self.tool.inputSchema:
            properties = self.tool.inputSchema.get("properties", {})
        
            for param_name, param_spec in properties.items():
                param_type = param_spec.get("type", "string")
                if param_type == "string":
                    annotations[param_name] = str
                elif param_type == "integer":
                    annotations[param_name] = int
                elif param_type == "number":
                    annotations[param_name] = float
                elif param_type == "boolean":
                    annotations[param_name] = bool
                elif param_type == "array":
                    annotations[param_name] = list
                elif param_type == "object":
                    annotations[param_name] = dict
                else:
                    annotations[param_name] = Any
        
        annotations["return"] = str 
        return annotations

    async def __call__(self, **kwargs):
        return await self.server_connection.call_tool(self.tool.name, kwargs)

    def to_oai_tool(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.__name__,
                "description": self.__doc__ or "",
                "parameters": self.tool.inputSchema or {
                    "type": "object",
                    "properties": {}
                }
            }
        }


