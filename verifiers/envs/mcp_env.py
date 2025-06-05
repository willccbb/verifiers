import asyncio
import json
import os
import re
from typing import Dict, Any, List, Tuple

import requests
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

from verifiers import MultiTurnEnv
from verifiers.parsers import Parser, XMLParser

DEFAULT_MCP_PROMPT_TEMPLATE = """
You are a helpful assistant that can use the following tools to answer questions:
{tool_specs}

In each turn, think step-by-step inside <think>...</think> tags, then either call a tool inside <tool>...</tool> tags, or give your final answer inside <answer>...</answer> tags.

Tool calls should be formatted as JSON inside <tool> tags with:
- "name": the name of the tool to use
- "args": the arguments for the tool

You will then see the tool's output inside <result> tags as a new message.
"""


class MCPEnv(MultiTurnEnv):
    def __init__(self,
                 mcp_endpoints: Dict[str, str],
                 system_prompt: str = DEFAULT_MCP_PROMPT_TEMPLATE,
                 parser: Parser = XMLParser(fields=["think", ("tool", "answer")]),
                 env_parser: XMLParser = XMLParser(fields=["result"]),
                 **kwargs):
        self.mcp_endpoints = mcp_endpoints
        self.env_parser = env_parser
        super().__init__(
            system_prompt=system_prompt.format(tool_specs=self.load_tool_specs()),
            parser=parser,
            **kwargs
        )

    async def call_tool(self, 
                        tool_name: str,
                        tool_args: Dict[str, Any]) -> List[Any]:
        async with Client(StreamableHttpTransport(self.mcp_endpoints[tool_name])) as client:
            result = await client.call_tool(tool_name, tool_args)
            return result
        
    def load_tool_specs(self) -> str:
        specs = []
        for _, endpoint in self.mcp_endpoints.items():
            meta = requests.get(f"{endpoint}/metadata",
                                headers={"Accept": "text/event-stream"},
                                stream=True, timeout=3).json()
            specs.append(meta)
        return json.dumps(specs, indent=2)
        
    def is_completed(self, messages: List[Dict[str, Any]], state: Dict[str, Any], **kwargs: Any) -> bool:
        return self.parser.parse_answer(messages) is not None

    def env_response(self,
                     messages: List[Dict[str, str]], 
                     state: Dict[str, Any],
                     **kwargs: Any) -> Tuple[Dict[str, str], Dict[str, Any]]:
        import nest_asyncio
        nest_asyncio.apply()
        try:
            parsed = self.parser.parse(messages[-1]['content'])
            # Check if we got a valid tool field (not just None from failed parsing)
            if hasattr(parsed, 'tool') and parsed.tool is not None:
                result = asyncio.run(self.call_tool(parsed.tool)) # type: ignore
                print(result)
                if len(result.strip()) > 0:
                    return {'role': 'user', 'content': self.env_parser.format(result=str(result))}, {} 
                else:
                    return {'role': 'user', 'content': "Error: Tool execution returned empty output."}, {}
        except Exception:
            pass
        return {'role': 'user', 'content': "Error: Tool command not found or invalid XML format. Please ensure correct formatting."}, {}