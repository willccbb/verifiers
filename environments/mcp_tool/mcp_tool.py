import asyncio
import json
import logging
import concurrent.futures
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from datasets import Dataset
from prime_cli.api.sandbox import AsyncSandboxClient, CreateSandboxRequest

import verifiers as vf
from verifiers.utils.data_utils import load_example_dataset


class PrimeSandboxManager:
    """Manages Prime Intellect sandboxes using the Prime API client."""
    
    def __init__(self, 
                 image: str = "node:22-bullseye",
                 cpu_cores: int = 1,
                 memory_gb: int = 2,
                 disk_size_gb: int = 10,
                 api_key: Optional[str] = None):
        self.image = image
        self.cpu_cores = cpu_cores
        self.memory_gb = memory_gb
        self.disk_size_gb = disk_size_gb
        self.api_key = api_key or os.getenv('PRIME_API_KEY')
        self.sandbox_client = AsyncSandboxClient(api_key=self.api_key)
        
        self.sandbox_id: Optional[str] = None
        self.sandbox = None
        self.logger = logging.getLogger(f"verifiers.envs.MCPToolEnv.SandboxManager")
    
    async def _wait_for_sandbox_ready(self, max_wait_seconds: int = 60):
        """Wait for sandbox to be ready."""
        if not self.sandbox_id:
            return
        
        self.logger.info(f"Waiting for sandbox {self.sandbox_id} to be ready...")
        
        try:
            await self.sandbox_client.wait_for_creation(self.sandbox_id, max_attempts=max_wait_seconds)
            self.logger.info("Sandbox ready")
        except Exception as e:
            self.logger.warning(f"Sandbox may not be ready: {e}")
    
    async def create_sandbox(self) -> str:
        """Create a new Prime sandbox."""
        try:
            self.logger.info(f"Creating sandbox with {self.image}")
            
            request = CreateSandboxRequest(
                name=f"mcp-{int(asyncio.get_event_loop().time())}",
                docker_image=self.image,
                cpu_cores=self.cpu_cores,
                memory_gb=self.memory_gb,
                disk_size_gb=self.disk_size_gb,
                start_command="tail -f /dev/null"
            )
            
            self.sandbox = await self.sandbox_client.create(request)
            self.sandbox_id = self.sandbox.id
            self.logger.info(f"Created sandbox: {self.sandbox_id}")
            
            await self._wait_for_sandbox_ready()
            return self.sandbox_id
            
        except Exception as e:
            self.logger.error(f"Error creating sandbox: {e}")
            raise
    
    async def run_command(self, command: str) -> Tuple[int, str, str]:
        """Execute a command in the sandbox using API client."""
        if not self.sandbox_id:
            raise RuntimeError("No sandbox available")
        
        try:
            result = await self.sandbox_client.execute_command(
                sandbox_id=self.sandbox_id,
                command=command
            )
            
            return result.exit_code, result.stdout, result.stderr
            
        except Exception as e:
            self.logger.error(f"Error running command: {e}")
            raise
    
    async def cleanup(self):
        """Delete the sandbox using API client."""
        if not self.sandbox_id:
            return
        
        try:
            await self.sandbox_client.delete(self.sandbox_id)
            self.logger.info(f"Deleted sandbox: {self.sandbox_id}")
            self.sandbox_id = None
            self.sandbox = None
            
        except Exception as e:
            self.logger.error(f"Error deleting sandbox: {e}")


class MCPServerManager:
    """Manages MCP servers spawned from npx/uv commands."""
    
    def __init__(self, sandbox_manager: PrimeSandboxManager):
        self.sandbox_manager = sandbox_manager
        self.servers: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(f"verifiers.envs.MCPToolEnv.MCPServerManager")
    
    async def spawn_server(self, 
                          server_name: str,
                          launch_command: List[str],
                          setup_commands: Optional[List[str]] = None) -> Dict[str, Any]:
        """Spawn an MCP server using the given launch command."""
        try:
            # Run setup commands
            if setup_commands:
                for setup_cmd in setup_commands:
                    self.logger.info(f"Setup: {setup_cmd}")
                    returncode, stdout, stderr = await self.sandbox_manager.run_command(setup_cmd)
                    if returncode != 0:
                        self.logger.warning(f"Setup failed: {stderr}")
            
            # Install MCP dependencies
            install_commands = [
                "apt-get update && apt-get install -y python3 python3-pip",  # Install Python in Node image
                "pip3 install mcp2cli",
                "npm install -g @modelcontextprotocol/server-filesystem" if "filesystem" in server_name else None
            ]
            
            for install_cmd in install_commands:
                if install_cmd:
                    self.logger.info(f"Installing: {install_cmd.split('&&')[-1].strip()}")
                    await self.sandbox_manager.run_command(install_cmd)
            
            # Create MCP configuration
            mcp_config = {
                "mcpServers": {
                    server_name: {
                        "command": launch_command[0],
                        "args": launch_command[1:] if len(launch_command) > 1 else []
                    }
                }
            }
            
            # Write config and start server
            config_json = json.dumps(mcp_config, indent=2)
            write_config_cmd = f'echo \'{config_json}\' > mcp.json'
            await self.sandbox_manager.run_command(write_config_cmd)
            
            start_cmd = f"nohup {' '.join(launch_command)} > mcp_{server_name}.log 2>&1 &"
            await self.sandbox_manager.run_command(start_cmd)
            
            await asyncio.sleep(2)  # Give server time to start
            
            # Discover tools
            tools = await self._discover_tools(server_name)
            
            server_info = {
                "name": server_name,
                "command": launch_command,
                "tools": tools,
                "status": "running"
            }
            
            self.servers[server_name] = server_info
            self.logger.info(f"Started MCP server '{server_name}' with {len(tools)} tools")
            
            return server_info
            
        except Exception as e:
            self.logger.error(f"Error spawning MCP server '{server_name}': {e}")
            raise
    
    async def _discover_tools(self, server_name: str) -> List[Dict[str, Any]]:
        """Discover available tools from an MCP server."""
        try:
            # For filesystem server, return known tools
            if "filesystem" in server_name.lower():
                return [
                    {"name": "read_file", "description": "Read a file from the filesystem", "server": server_name},
                    {"name": "write_file", "description": "Write content to a file", "server": server_name},
                    {"name": "list_directory", "description": "List contents of a directory", "server": server_name}
                ]
            
            # Try to discover tools via mcp2cli
            returncode, stdout, stderr = await self.sandbox_manager.run_command("python3 -c 'import mcp2cli; print(\"mcp2cli available\")' 2>/dev/null || echo 'mcp2cli not available'")
            
            if "available" in stdout:
                # Return generic MCP tools
                return [
                    {"name": f"{server_name}_tool", "description": f"MCP tool from {server_name}", "server": server_name}
                ]
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error discovering tools: {e}")
            return []
    
    async def call_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Call an MCP tool with the given arguments."""
        try:
            # Handle filesystem tools directly for demo purposes
            if tool_name == "read_file" and "path" in args:
                cmd = f"cat {args['path']}"
            elif tool_name == "write_file" and "path" in args and "content" in args:
                cmd = f"echo '{args['content']}' > {args['path']}"
            elif tool_name == "list_directory" and "path" in args:
                cmd = f"ls -la {args['path']}"
            elif tool_name == "read_file" and "query" in args:
                # Handle generic query format
                cmd = f"cat /workspace/{args['query']}.txt 2>/dev/null || ls /workspace/"
            elif tool_name == "write_file" and "query" in args:
                cmd = f"echo 'MCP tool output: {args['query']}' > /workspace/mcp_output.txt"
            elif tool_name == "list_directory":
                cmd = "ls -la /workspace/"
            else:
                # Fallback to mcp2cli
                cmd_parts = ["python3", "-m", "mcp2cli", tool_name]
                for key, value in args.items():
                    cmd_parts.extend([f"--{key}", str(value)])
                cmd = " ".join(cmd_parts)
            
            returncode, stdout, stderr = await self.sandbox_manager.run_command(cmd)
            
            if returncode != 0:
                return f"Error calling tool {tool_name}: {stderr.strip()}"
            
            return stdout.strip()
            
        except Exception as e:
            self.logger.error(f"Error calling MCP tool '{tool_name}': {e}")
            return f"Error: {str(e)}"
    
    async def shutdown_servers(self):
        """Shutdown all MCP servers."""
        # Servers will be automatically terminated when sandbox is deleted
        self.logger.info(f"Shutting down {len(self.servers)} MCP servers")
        self.servers.clear()


def create_mcp_tool_wrapper(tool_name: str, tool_description: str, server_manager: MCPServerManager) -> Callable:
    """Create a Python function wrapper for an MCP tool."""
    
    def mcp_tool_wrapper(query: str = "") -> str:
        """MCP tool wrapper.
        
        Args:
            query: Input query or parameters for the MCP tool
            
        Returns:
            Result from the MCP tool execution
        """
        async def _async_call():
            return await server_manager.call_tool(tool_name, {"query": query})
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, _async_call())
                    return future.result()
            else:
                return loop.run_until_complete(_async_call())
        except RuntimeError:
            return asyncio.run(_async_call())
    
    mcp_tool_wrapper.__name__ = tool_name
    mcp_tool_wrapper.__doc__ = tool_description
    
    return mcp_tool_wrapper


class MCPToolEnv(vf.ToolEnv):
    """Environment that spawns MCP servers in Prime sandboxes and exposes them as tools."""
    
    def __init__(self,
                 mcp_launch_commands: List[Dict[str, Any]],
                 sandbox_config: Optional[Dict[str, Any]] = None,
                 prime_api_key: Optional[str] = None,
                 **kwargs):
        self.mcp_launch_commands = mcp_launch_commands
        self.sandbox_config = sandbox_config or {}
        self.prime_api_key = prime_api_key
        
        # Initialize managers
        sandbox_config_with_key = self.sandbox_config.copy()
        if self.prime_api_key:
            sandbox_config_with_key['api_key'] = self.prime_api_key
        
        self.sandbox_manager = PrimeSandboxManager(**sandbox_config_with_key)
        self.mcp_server_manager = MCPServerManager(self.sandbox_manager)
        
        # Initialize with empty tools - populated after server startup
        super().__init__(tools=[], **kwargs)
        
        self.logger = logging.getLogger(f"verifiers.envs.MCPToolEnv")
        self._initialized = False
    
    async def _initialize_servers(self):
        """Initialize sandbox and spawn MCP servers."""
        if self._initialized:
            return
        
        try:
            # Create sandbox
            await self.sandbox_manager.create_sandbox()
            
            # Spawn MCP servers and collect tools
            all_tools = []
            for server_config in self.mcp_launch_commands:
                server_name = server_config["name"]
                launch_command = server_config["command"]
                setup_commands = server_config.get("setup_commands")
                
                server_info = await self.mcp_server_manager.spawn_server(
                    server_name, launch_command, setup_commands
                )
                
                # Create tool functions
                for tool_info in server_info["tools"]:
                    tool_func = create_mcp_tool_wrapper(
                        tool_info["name"], 
                        tool_info["description"], 
                        self.mcp_server_manager
                    )
                    all_tools.append(tool_func)
            
            # Update tool environment
            self.tools = all_tools
            from verifiers.utils.tool_utils import convert_func_to_oai_tool
            self.oai_tools = [convert_func_to_oai_tool(tool) for tool in self.tools]
            self.tool_map = {tool.__name__: tool for tool in self.tools}
            
            self._initialized = True
            self.logger.info(f"Initialized MCPToolEnv with {len(all_tools)} tools")
            
        except Exception as e:
            self.logger.error(f"Error initializing MCP servers: {e}")
            await self.cleanup()
            raise
    
    async def rollout(self, *args, **kwargs) -> Tuple[vf.Messages, vf.State]:
        """Override rollout to ensure servers are initialized."""
        await self._initialize_servers()
        return await super().rollout(*args, **kwargs)
    
    async def cleanup(self):
        """Clean up MCP servers and sandbox."""
        try:
            if hasattr(self, 'mcp_server_manager'):
                await self.mcp_server_manager.shutdown_servers()
            
            if hasattr(self, 'sandbox_manager'):
                await self.sandbox_manager.cleanup()
                
            self.logger.info("MCPToolEnv cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


def load_environment(
    mcp_launch_commands: List[Dict[str, Any]],
    dataset_name: str = "gsm8k",
    split: str = "train",
    num_examples: int = 100,
    sandbox_config: Optional[Dict[str, Any]] = None,
    max_turns: int = 10,
    prime_api_key: Optional[str] = None,
    **kwargs
) -> MCPToolEnv:
    """
    Load MCPToolEnv with specified MCP server configurations.
    
    Args:
        mcp_launch_commands: List of MCP server configurations
        dataset_name: Name of the dataset to load
        split: Dataset split to use
        num_examples: Number of examples to use
        sandbox_config: Prime sandbox configuration
        max_turns: Maximum conversation turns
        prime_api_key: Prime Intellect API key
        **kwargs: Additional environment arguments
    
    Returns:
        Configured MCPToolEnv instance
    """
    
    # Load dataset
    dataset = load_example_dataset(dataset_name, split=split)
    
    if num_examples > 0:
        dataset = dataset.select(range(min(num_examples, len(dataset))))
    
    # Create parser and rubric
    parser = vf.Parser()
    rubric = vf.ToolRubric(tools=[])  # Tools populated after server initialization
    
    # Create environment
    env = MCPToolEnv(
        mcp_launch_commands=mcp_launch_commands,
        sandbox_config=sandbox_config,
        dataset=dataset,
        parser=parser,
        rubric=rubric,
        max_turns=max_turns,
        prime_api_key=prime_api_key,
        **kwargs
    )
    
    return env