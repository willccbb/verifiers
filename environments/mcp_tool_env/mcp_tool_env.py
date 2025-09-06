import asyncio
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from datasets import Dataset

import verifiers as vf
from verifiers.envs.tool_env import ToolEnv
from verifiers.types import Messages, State


class PrimeSandboxManager:
    """Manages Prime Intellect sandboxes for MCP server execution."""
    
    def __init__(self, 
                 image: str = "python:3.11-slim",
                 cpu_cores: int = 1,
                 memory_gb: int = 2,
                 disk_size_gb: int = 10,
                 api_key: Optional[str] = None):
        self.image = image
        self.cpu_cores = cpu_cores
        self.memory_gb = memory_gb
        self.disk_size_gb = disk_size_gb
        self.api_key = api_key
        self.sandbox_id: Optional[str] = None
        self.logger = logging.getLogger(f"verifiers.envs.MCPToolEnv.SandboxManager")
    
    def _get_env_with_auth(self) -> Dict[str, str]:
        """Get environment variables with Prime CLI authentication."""
        import os
        env = os.environ.copy()
        
        # If API key is provided, set it as environment variable
        if self.api_key:
            env['PRIME_API_KEY'] = self.api_key
        
        return env
    
    async def _wait_for_sandbox_ready(self, max_wait_seconds: int = 60):
        """Wait for sandbox to be in RUNNING state."""
        if not self.sandbox_id:
            return
        
        self.logger.info(f"Waiting for sandbox {self.sandbox_id} to be ready...")
        
        for attempt in range(max_wait_seconds):
            try:
                # Check sandbox status
                env = self._get_env_with_auth()
                result = await asyncio.create_subprocess_exec(
                    "prime", "sandbox", "get", self.sandbox_id,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env
                )
                stdout, stderr = await result.communicate()
                
                if result.returncode == 0:
                    output = stdout.decode().strip()
                    if "RUNNING" in output or "Running" in output:
                        self.logger.info(f"✅ Sandbox is ready after {attempt + 1} seconds")
                        return
                
                # Wait 1 second before next check
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.debug(f"Error checking sandbox status: {e}")
                await asyncio.sleep(1)
        
        self.logger.warning(f"⚠️  Sandbox may not be ready after {max_wait_seconds} seconds, proceeding anyway")
    
    async def create_sandbox(self) -> str:
        """Create a new Prime sandbox and return its ID."""
        try:
            # Follow the exact Prime CLI format from docs
            cmd = ["prime", "sandbox", "create", self.image, "--yes"]  # Auto-confirm creation
            
            # Add resource options if specified
            if self.cpu_cores > 1:
                cmd.extend(["--cpu-cores", str(self.cpu_cores)])
            if self.memory_gb > 1:
                cmd.extend(["--memory-gb", str(self.memory_gb)])
            if self.disk_size_gb > 10:
                cmd.extend(["--disk-size-gb", str(self.disk_size_gb)])
            
            self.logger.info(f"Creating sandbox with command: {' '.join(cmd)}")
            
            # Get environment with authentication
            env = self._get_env_with_auth()
            
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                raise RuntimeError(f"Failed to create sandbox: {stderr.decode()}")
            
            # Parse the output - look for "Successfully created sandbox <id>"
            output = stdout.decode().strip()
            self.logger.debug(f"Sandbox creation output: {output}")
            
            lines = [line.strip() for line in output.split('\n') if line.strip()]
            
            # Look for the success message with sandbox ID
            for line in lines:
                if "Successfully created sandbox" in line:
                    # Extract the sandbox ID from the success message
                    parts = line.split("Successfully created sandbox")
                    if len(parts) > 1:
                        self.sandbox_id = parts[1].strip()
                        break
            
            # Fallback: look for any line that looks like a sandbox ID
            if not self.sandbox_id:
                for line in lines:
                    # Sandbox IDs are typically long alphanumeric strings
                    if len(line) > 15 and line.replace('-', '').replace('_', '').isalnum():
                        self.sandbox_id = line
                        break
            
            if not self.sandbox_id:
                raise RuntimeError(f"Could not extract sandbox ID from output: {output}")
            
            self.logger.info(f"Created sandbox: {self.sandbox_id}")
            
            # Wait for sandbox to be in RUNNING state
            await self._wait_for_sandbox_ready()
            
            return self.sandbox_id
            
        except Exception as e:
            self.logger.error(f"Error creating sandbox: {e}")
            raise
    
    async def run_command(self, command: str) -> Tuple[int, str, str]:
        """Execute a command in the sandbox."""
        if not self.sandbox_id:
            raise RuntimeError("No sandbox available. Create one first.")
        
        try:
            cmd = ["prime", "sandbox", "run", self.sandbox_id, command]
            
            # Get environment with authentication
            env = self._get_env_with_auth()
            
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
            stdout, stderr = await result.communicate()
            
            return result.returncode, stdout.decode(), stderr.decode()
            
        except Exception as e:
            self.logger.error(f"Error running command '{command}': {e}")
            raise
    
    async def cleanup(self):
        """Delete the sandbox."""
        if not self.sandbox_id:
            return
        
        try:
            cmd = ["prime", "sandbox", "delete", self.sandbox_id, "--yes"]  # Auto-confirm deletion
            
            # Get environment with authentication
            env = self._get_env_with_auth()
            
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                self.logger.info(f"✅ Deleted sandbox: {self.sandbox_id}")
            else:
                self.logger.error(f"❌ Failed to delete sandbox: {stderr.decode()}")
            
            self.sandbox_id = None
            
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
            # Run setup commands if provided
            if setup_commands:
                for setup_cmd in setup_commands:
                    self.logger.info(f"Running setup command: {setup_cmd}")
                    returncode, stdout, stderr = await self.sandbox_manager.run_command(setup_cmd)
                    if returncode != 0:
                        self.logger.warning(f"Setup command failed: {stderr}")
            
            # Install required MCP tools
            install_commands = [
                "pip install mcp2cli",
                "npm install -g @modelcontextprotocol/server-filesystem" if "filesystem" in server_name else None
            ]
            
            for install_cmd in install_commands:
                if install_cmd:
                    self.logger.info(f"Installing MCP dependencies: {install_cmd}")
                    returncode, stdout, stderr = await self.sandbox_manager.run_command(install_cmd)
                    if returncode != 0:
                        self.logger.warning(f"Install command failed: {stderr}")
            
            # Create MCP configuration
            mcp_config = {
                "mcpServers": {
                    server_name: {
                        "command": launch_command[0],
                        "args": launch_command[1:] if len(launch_command) > 1 else []
                    }
                }
            }
            
            # Write config to sandbox
            config_json = json.dumps(mcp_config, indent=2)
            write_config_cmd = f'echo \'{config_json}\' > mcp.json'
            
            returncode, stdout, stderr = await self.sandbox_manager.run_command(write_config_cmd)
            if returncode != 0:
                raise RuntimeError(f"Failed to write MCP config: {stderr}")
            
            # Start the MCP server (in background)
            start_cmd = f"nohup {' '.join(launch_command)} > mcp_{server_name}.log 2>&1 &"
            returncode, stdout, stderr = await self.sandbox_manager.run_command(start_cmd)
            
            # Give the server time to start
            await asyncio.sleep(2)
            
            # Get available tools from the server
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
            # Use mcp2cli to list available tools
            list_cmd = "uvx mcp2cli"
            returncode, stdout, stderr = await self.sandbox_manager.run_command(list_cmd)
            
            if returncode != 0:
                self.logger.warning(f"Failed to list tools for {server_name}: {stderr}")
                return []
            
            # Parse tool list from mcp2cli output
            tools = []
            lines = stdout.strip().split('\n')
            
            for line in lines:
                if line.strip() and not line.startswith('Available'):
                    tool_name = line.strip().split()[0] if line.strip().split() else line.strip()
                    if tool_name:
                        tools.append({
                            "name": tool_name,
                            "description": f"MCP tool: {tool_name}",
                            "server": server_name
                        })
            
            return tools
            
        except Exception as e:
            self.logger.error(f"Error discovering tools for {server_name}: {e}")
            return []
    
    async def call_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Call an MCP tool with the given arguments."""
        try:
            # Build the mcp2cli command
            cmd_parts = ["uvx", "mcp2cli", tool_name]
            
            # Add arguments
            for key, value in args.items():
                cmd_parts.extend([f"--{key}", str(value)])
            
            cmd = " ".join(cmd_parts)
            
            returncode, stdout, stderr = await self.sandbox_manager.run_command(cmd)
            
            if returncode != 0:
                return f"Error calling tool {tool_name}: {stderr}"
            
            return stdout.strip()
            
        except Exception as e:
            self.logger.error(f"Error calling MCP tool '{tool_name}': {e}")
            return f"Error: {str(e)}"
    
    async def shutdown_servers(self):
        """Shutdown all MCP servers."""
        for server_name in self.servers:
            try:
                # Kill the server process
                kill_cmd = f"pkill -f '{' '.join(self.servers[server_name]['command'])}'"
                await self.sandbox_manager.run_command(kill_cmd)
                self.logger.info(f"Shutdown MCP server: {server_name}")
            except Exception as e:
                self.logger.error(f"Error shutting down server {server_name}: {e}")
        
        self.servers.clear()


class MCPToolEnv(ToolEnv):
    """Environment that spawns MCP servers in Prime sandboxes and exposes them as tools."""
    
    def __init__(self,
                 mcp_launch_commands: List[Dict[str, Any]],
                 sandbox_config: Optional[Dict[str, Any]] = None,
                 dataset: Optional[Dataset] = None,
                 eval_dataset: Optional[Dataset] = None,
                 max_turns: int = 10,
                 prime_api_key: Optional[str] = None,
                 **kwargs):
        """
        Initialize MCPToolEnv.
        
        Args:
            mcp_launch_commands: List of MCP server configurations, each containing:
                - name: Server name
                - command: Launch command as list of strings
                - setup_commands: Optional setup commands to run before launching
            sandbox_config: Configuration for Prime sandbox (cpu_cores, memory_gb, etc.)
            dataset: Training dataset
            eval_dataset: Evaluation dataset
            max_turns: Maximum number of conversation turns
            **kwargs: Additional arguments passed to ToolEnv
        """
        self.mcp_launch_commands = mcp_launch_commands
        self.sandbox_config = sandbox_config or {}
        self.prime_api_key = prime_api_key
        
        # Initialize sandbox and MCP managers
        sandbox_config_with_key = self.sandbox_config.copy()
        if self.prime_api_key:
            sandbox_config_with_key['api_key'] = self.prime_api_key
        
        self.sandbox_manager = PrimeSandboxManager(**sandbox_config_with_key)
        self.mcp_server_manager = MCPServerManager(self.sandbox_manager)
        
        # Initialize with empty tools list - will be populated after server startup
        super().__init__(
            tools=[],
            dataset=dataset,
            eval_dataset=eval_dataset,
            max_turns=max_turns,
            **kwargs
        )
        
        self.logger = logging.getLogger(f"verifiers.envs.MCPToolEnv")
        self._initialized = False
    
    async def _initialize_servers(self):
        """Initialize sandbox and spawn MCP servers."""
        if self._initialized:
            return
        
        try:
            # Create sandbox
            await self.sandbox_manager.create_sandbox()
            
            # Spawn MCP servers
            all_tools = []
            for server_config in self.mcp_launch_commands:
                server_name = server_config["name"]
                launch_command = server_config["command"]
                setup_commands = server_config.get("setup_commands")
                
                server_info = await self.mcp_server_manager.spawn_server(
                    server_name, launch_command, setup_commands
                )
                
                # Create tool functions for each MCP tool
                for tool_info in server_info["tools"]:
                    tool_func = self._create_tool_function(tool_info)
                    all_tools.append(tool_func)
            
            # Update the tool environment with discovered tools
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
    
    def _create_tool_function(self, tool_info: Dict[str, Any]) -> Callable:
        """Create a Python function wrapper for an MCP tool."""
        tool_name = tool_info["name"]
        tool_description = tool_info["description"]
        
        def mcp_tool_wrapper(query: str = "") -> str:
            """Dynamically generated MCP tool wrapper.
            
            Args:
                query: Input query or parameters for the MCP tool
                
            Returns:
                Result from the MCP tool execution
            """
            # Create async wrapper for sync function signature
            async def _async_call():
                return await self.mcp_server_manager.call_tool(tool_name, {"query": query})
            
            # Run in event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're in an async context, create a task
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, _async_call())
                        return future.result()
                else:
                    return loop.run_until_complete(_async_call())
            except RuntimeError:
                return asyncio.run(_async_call())
        
        # Set function metadata
        mcp_tool_wrapper.__name__ = tool_name
        mcp_tool_wrapper.__doc__ = tool_description
        
        return mcp_tool_wrapper
    
    
    async def rollout(self, *args, **kwargs) -> Tuple[Messages, State]:
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
        **kwargs: Additional environment arguments
    
    Returns:
        Configured MCPToolEnv instance
    
    Example:
        mcp_commands = [
            {
                "name": "filesystem",
                "command": ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                "setup_commands": ["mkdir -p /tmp/test"]
            },
            {
                "name": "python_tools", 
                "command": ["uv", "run", "python", "-m", "mcp_this"],
                "setup_commands": ["pip install mcp-this"]
            }
        ]
        
        env = load_environment(
            mcp_launch_commands=mcp_commands,
            dataset_name="math",
            split="train"
        )
    """
    
    # Load dataset
    from verifiers.utils.data_utils import load_example_dataset
    dataset = load_example_dataset(dataset_name, split=split)
    
    if num_examples > 0:
        dataset = dataset.select(range(min(num_examples, len(dataset))))
    
    # Create parser and rubric
    parser = vf.Parser()
    rubric = vf.ToolRubric(tools=[])  # Tools will be populated after server initialization
    
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