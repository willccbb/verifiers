import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datasets import Dataset

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'environments', 'mcp_tool_env'))

from mcp_tool_env import MCPToolEnv, PrimeSandboxManager, MCPServerManager, load_environment


class TestPrimeSandboxManager:
    """Test suite for PrimeSandboxManager."""
    
    @pytest.fixture
    def sandbox_manager(self):
        return PrimeSandboxManager(
            image="python:3.11-slim",
            cpu_cores=1,
            memory_gb=2,
            disk_size_gb=10
        )
    
    @pytest.mark.asyncio
    async def test_create_sandbox_success(self, sandbox_manager):
        """Test successful sandbox creation."""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (
            b"Successfully created sandbox sandbox-12345\n", 
            b""
        )
        
        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            sandbox_id = await sandbox_manager.create_sandbox()
            
            assert sandbox_id == "sandbox-12345"
            assert sandbox_manager.sandbox_id == "sandbox-12345"
    
    @pytest.mark.asyncio
    async def test_create_sandbox_failure(self, sandbox_manager):
        """Test sandbox creation failure."""
        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate.return_value = (
            b"", 
            b"Error: Failed to create sandbox"
        )
        
        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            with pytest.raises(RuntimeError, match="Failed to create sandbox"):
                await sandbox_manager.create_sandbox()
    
    @pytest.mark.asyncio
    async def test_run_command_success(self, sandbox_manager):
        """Test successful command execution."""
        sandbox_manager.sandbox_id = "sandbox-12345"
        
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (
            b"Command output\n", 
            b""
        )
        
        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            returncode, stdout, stderr = await sandbox_manager.run_command("echo hello")
            
            assert returncode == 0
            assert stdout == "Command output\n"
            assert stderr == ""
    
    @pytest.mark.asyncio
    async def test_run_command_no_sandbox(self, sandbox_manager):
        """Test command execution without sandbox."""
        with pytest.raises(RuntimeError, match="No sandbox available"):
            await sandbox_manager.run_command("echo hello")
    
    @pytest.mark.asyncio
    async def test_cleanup(self, sandbox_manager):
        """Test sandbox cleanup."""
        sandbox_manager.sandbox_id = "sandbox-12345"
        
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"", b"")
        
        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            await sandbox_manager.cleanup()
            
            assert sandbox_manager.sandbox_id is None


class TestMCPServerManager:
    """Test suite for MCPServerManager."""
    
    @pytest.fixture
    def sandbox_manager(self):
        manager = MagicMock(spec=PrimeSandboxManager)
        manager.run_command = AsyncMock()
        return manager
    
    @pytest.fixture
    def server_manager(self, sandbox_manager):
        return MCPServerManager(sandbox_manager)
    
    @pytest.mark.asyncio
    async def test_spawn_server_success(self, server_manager, sandbox_manager):
        """Test successful MCP server spawning."""
        # Mock successful command execution
        sandbox_manager.run_command.return_value = (0, "Success", "")
        
        # Mock tool discovery
        with patch.object(server_manager, '_discover_tools', return_value=[
            {"name": "test_tool", "description": "Test tool", "server": "test_server"}
        ]):
            server_info = await server_manager.spawn_server(
                "test_server",
                ["python", "-m", "test_mcp_server"]
            )
            
            assert server_info["name"] == "test_server"
            assert server_info["status"] == "running"
            assert len(server_info["tools"]) == 1
            assert "test_server" in server_manager.servers
    
    @pytest.mark.asyncio
    async def test_discover_tools(self, server_manager, sandbox_manager):
        """Test MCP tool discovery."""
        sandbox_manager.run_command.return_value = (
            0, 
            "Available tools:\ntool1\ntool2\ntool3", 
            ""
        )
        
        tools = await server_manager._discover_tools("test_server")
        
        assert len(tools) == 3
        assert tools[0]["name"] == "tool1"
        assert tools[1]["name"] == "tool2" 
        assert tools[2]["name"] == "tool3"
    
    @pytest.mark.asyncio
    async def test_call_tool(self, server_manager, sandbox_manager):
        """Test MCP tool execution."""
        sandbox_manager.run_command.return_value = (0, "Tool result", "")
        
        result = await server_manager.call_tool("test_tool", {"arg1": "value1"})
        
        assert result == "Tool result"
        sandbox_manager.run_command.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_call_tool_error(self, server_manager, sandbox_manager):
        """Test MCP tool execution with error."""
        sandbox_manager.run_command.return_value = (1, "", "Tool error")
        
        result = await server_manager.call_tool("test_tool", {"arg1": "value1"})
        
        assert "Error calling tool test_tool: Tool error" in result
    
    @pytest.mark.asyncio
    async def test_shutdown_servers(self, server_manager, sandbox_manager):
        """Test MCP server shutdown."""
        server_manager.servers = {
            "test_server": {
                "command": ["python", "-m", "test_server"]
            }
        }
        
        sandbox_manager.run_command.return_value = (0, "", "")
        
        await server_manager.shutdown_servers()
        
        assert len(server_manager.servers) == 0
        sandbox_manager.run_command.assert_called_once()


class TestMCPToolEnv:
    """Test suite for MCPToolEnv."""
    
    @pytest.fixture
    def sample_dataset(self):
        return Dataset.from_dict({
            "prompt": [
                [{"role": "user", "content": "Test question 1"}],
                [{"role": "user", "content": "Test question 2"}]
            ],
            "answer": ["Answer 1", "Answer 2"]
        })
    
    @pytest.fixture
    def mcp_commands(self):
        return [
            {
                "name": "filesystem",
                "command": ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                "setup_commands": ["mkdir -p /tmp/test"]
            }
        ]
    
    def test_init(self, sample_dataset, mcp_commands):
        """Test MCPToolEnv initialization."""
        env = MCPToolEnv(
            mcp_launch_commands=mcp_commands,
            dataset=sample_dataset,
            max_turns=5
        )
        
        assert env.mcp_launch_commands == mcp_commands
        assert env.max_turns == 5
        assert not env._initialized
        assert isinstance(env.sandbox_manager, PrimeSandboxManager)
        assert isinstance(env.mcp_server_manager, MCPServerManager)
    
    @pytest.mark.asyncio
    async def test_initialize_servers(self, sample_dataset, mcp_commands):
        """Test MCP server initialization."""
        env = MCPToolEnv(
            mcp_launch_commands=mcp_commands,
            dataset=sample_dataset
        )
        
        # Mock the managers
        env.sandbox_manager.create_sandbox = AsyncMock(return_value="sandbox-12345")
        env.mcp_server_manager.spawn_server = AsyncMock(return_value={
            "name": "filesystem",
            "tools": [
                {"name": "read_file", "description": "Read file"},
                {"name": "write_file", "description": "Write file"}
            ],
            "status": "running"
        })
        
        await env._initialize_servers()
        
        assert env._initialized
        assert len(env.tools) == 2
        assert len(env.oai_tools) == 2
        assert "read_file" in env.tool_map
        assert "write_file" in env.tool_map
    
    def test_create_tool_function(self, sample_dataset, mcp_commands):
        """Test MCP tool function creation."""
        env = MCPToolEnv(
            mcp_launch_commands=mcp_commands,
            dataset=sample_dataset
        )
        
        tool_info = {
            "name": "test_tool",
            "description": "Test MCP tool"
        }
        
        tool_func = env._create_tool_function(tool_info)
        
        assert tool_func.__name__ == "test_tool"
        assert tool_func.__doc__ == "Test MCP tool"
        assert callable(tool_func)
    
    @pytest.mark.asyncio
    async def test_cleanup(self, sample_dataset, mcp_commands):
        """Test environment cleanup."""
        env = MCPToolEnv(
            mcp_launch_commands=mcp_commands,
            dataset=sample_dataset
        )
        
        # Mock the cleanup methods
        env.mcp_server_manager.shutdown_servers = AsyncMock()
        env.sandbox_manager.cleanup = AsyncMock()
        
        await env.cleanup()
        
        env.mcp_server_manager.shutdown_servers.assert_called_once()
        env.sandbox_manager.cleanup.assert_called_once()


class TestLoadEnvironment:
    """Test suite for load_environment function."""
    
    @pytest.fixture
    def mcp_commands(self):
        return [
            {
                "name": "test_server",
                "command": ["python", "-m", "test_server"],
                "setup_commands": ["pip install test-server"]
            }
        ]
    
    def test_load_environment_basic(self, mcp_commands):
        """Test basic environment loading."""
        with patch('verifiers.utils.data_utils.load_example_dataset') as mock_load_dataset:
            mock_dataset = Dataset.from_dict({
                "prompt": [[{"role": "user", "content": "Test"}]],
                "answer": ["Test answer"]
            })
            mock_load_dataset.return_value = mock_dataset
            
            env = load_environment(
                mcp_launch_commands=mcp_commands,
                dataset_name="test_dataset",
                split="train",
                num_examples=10
            )
            
            assert isinstance(env, MCPToolEnv)
            assert env.mcp_launch_commands == mcp_commands
            assert env.max_turns == 10
            mock_load_dataset.assert_called_once_with("test_dataset", split="train")
    
    def test_load_environment_with_config(self, mcp_commands):
        """Test environment loading with sandbox config."""
        sandbox_config = {
            "cpu_cores": 2,
            "memory_gb": 4,
            "disk_size_gb": 20
        }
        
        with patch('verifiers.utils.data_utils.load_example_dataset') as mock_load_dataset:
            mock_dataset = Dataset.from_dict({
                "prompt": [[{"role": "user", "content": "Test"}]],
                "answer": ["Test answer"]
            })
            mock_load_dataset.return_value = mock_dataset
            
            env = load_environment(
                mcp_launch_commands=mcp_commands,
                sandbox_config=sandbox_config,
                max_turns=15
            )
            
            assert env.sandbox_config == sandbox_config
            assert env.max_turns == 15


# Integration tests (these would require actual Prime API access)
class TestIntegration:
    """Integration tests for MCPToolEnv (requires Prime API access)."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test full MCP workflow with real Prime sandbox."""
        # This test requires actual Prime API credentials and would be expensive
        # It's marked with @pytest.mark.integration so it can be skipped in CI
        pytest.skip("Integration test requires Prime API access")
        
        mcp_commands = [
            {
                "name": "filesystem",
                "command": ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                "setup_commands": ["mkdir -p /tmp/test"]
            }
        ]
        
        sample_dataset = Dataset.from_dict({
            "prompt": [[{"role": "user", "content": "List files in /tmp"}]],
            "answer": ["File listing"]
        })
        
        env = MCPToolEnv(
            mcp_launch_commands=mcp_commands,
            dataset=sample_dataset,
            max_turns=3
        )
        
        try:
            # Initialize servers
            await env._initialize_servers()
            
            # Verify tools are available
            assert len(env.tools) > 0
            assert env._initialized
            
        finally:
            # Always cleanup
            await env.cleanup()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
