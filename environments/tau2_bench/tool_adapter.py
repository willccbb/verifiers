"""
Tool adapter for converting tau2-bench tools to verifiers format.
Handles both agent and user tools.
"""

import json
from typing import Dict, Any, List, Optional, Callable, Tuple
from pydantic import BaseModel, Field, create_model

import verifiers as vf
from verifiers import BaseTool


class Tau2ToolAdapter(BaseTool):
    """
    Adapter to convert tau2-bench tools to verifiers BaseTool format.
    """
    
    def __init__(self, 
                 tau2_tool: Any,
                 tool_name: str,
                 is_user_tool: bool = False):
        """
        Initialize tool adapter.
        
        Args:
            tau2_tool: The tau2 tool instance or function
            tool_name: Name of the tool
            is_user_tool: Whether this is a user tool (for tracking)
        """
        self.tau2_tool = tau2_tool
        self.name = tool_name
        self.is_user_tool = is_user_tool
        
        # Extract description from tau2 tool if available
        if hasattr(tau2_tool, '__doc__') and tau2_tool.__doc__:
            self.description = tau2_tool.__doc__.strip()
        else:
            self.description = f"Tool: {tool_name}"
            
        # Build input schema from tau2 tool signature
        self._build_input_schema()
        
    def _build_input_schema(self):
        """Build Pydantic input schema from tau2 tool."""
        # Try to extract parameters from tau2 tool
        if hasattr(self.tau2_tool, '__annotations__'):
            # Create fields from annotations
            fields = {}
            for param_name, param_type in self.tau2_tool.__annotations__.items():
                if param_name != 'return':
                    # Create field with type and description
                    fields[param_name] = (param_type, Field(description=f"Parameter {param_name}"))
                    
            # Create dynamic Pydantic model
            self.args_schema = create_model(
                f"{self.name}Schema",
                **fields
            )
        else:
            # Fallback to generic schema
            self.args_schema = create_model(
                f"{self.name}Schema",
                **{"args": (Dict[str, Any], Field(description="Tool arguments"))}
            )
            
    def _run(self, **kwargs) -> str:
        """
        Execute the tau2 tool and return result as string.
        
        Args:
            **kwargs: Tool arguments
            
        Returns:
            String representation of tool result
        """
        try:
            # Call the tau2 tool
            if callable(self.tau2_tool):
                result = self.tau2_tool(**kwargs)
            else:
                # If it's a method, we might need to handle it differently
                result = self.tau2_tool(**kwargs)
                
            # Convert result to string format expected by verifiers
            if isinstance(result, str):
                return result
            elif isinstance(result, dict):
                return json.dumps(result, indent=2)
            elif isinstance(result, list):
                return json.dumps(result, indent=2)
            else:
                return str(result)
                
        except Exception as e:
            return f"Error executing tool {self.name}: {str(e)}"


def create_tools_from_tau2(tau2_tools_dict: Dict[str, Any], 
                          is_user_tools: bool = False) -> List[BaseTool]:
    """
    Convert a dictionary of tau2 tools to verifiers BaseTool format.
    
    Args:
        tau2_tools_dict: Dictionary mapping tool names to tool functions
        is_user_tools: Whether these are user tools
        
    Returns:
        List of adapted tools
    """
    tools = []
    
    for tool_name, tool_func in tau2_tools_dict.items():
        # Skip private methods
        if tool_name.startswith('_'):
            continue
            
        # Create adapter for this tool
        adapted_tool = Tau2ToolAdapter(
            tau2_tool=tool_func,
            tool_name=tool_name,
            is_user_tool=is_user_tools
        )
        
        tools.append(adapted_tool)
        
    return tools


def extract_tool_definitions_from_tau2(tau2_env) -> Tuple[List[BaseTool], List[BaseTool]]:
    """
    Extract both agent and user tools from tau2 environment.
    
    Args:
        tau2_env: The tau2 environment instance
        
    Returns:
        Tuple of (agent_tools, user_tools)
    """
    agent_tools = []
    user_tools = []
    
    # Extract agent tools
    if hasattr(tau2_env, 'tools') and tau2_env.tools:
        if hasattr(tau2_env.tools, 'get_tools'):
            # Tools are in a toolkit
            agent_tools_dict = tau2_env.tools.get_tools()
            agent_tools = create_tools_from_tau2(agent_tools_dict, is_user_tools=False)
        else:
            # Tools might be direct methods
            tool_methods = [m for m in dir(tau2_env.tools) 
                          if not m.startswith('_') and callable(getattr(tau2_env.tools, m))]
            agent_tools_dict = {
                method: getattr(tau2_env.tools, method) 
                for method in tool_methods
            }
            agent_tools = create_tools_from_tau2(agent_tools_dict, is_user_tools=False)
            
    # Extract user tools (mainly for telecom domain)
    if hasattr(tau2_env, 'user_tools') and tau2_env.user_tools:
        if hasattr(tau2_env.user_tools, 'get_tools'):
            # Tools are in a toolkit
            user_tools_dict = tau2_env.user_tools.get_tools()
            user_tools = create_tools_from_tau2(user_tools_dict, is_user_tools=True)
        else:
            # Tools might be direct methods
            tool_methods = [m for m in dir(tau2_env.user_tools) 
                          if not m.startswith('_') and callable(getattr(tau2_env.user_tools, m))]
            user_tools_dict = {
                method: getattr(tau2_env.user_tools, method) 
                for method in tool_methods
            }
            user_tools = create_tools_from_tau2(user_tools_dict, is_user_tools=True)
            
    return agent_tools, user_tools


class ToolExecutor:
    """
    Handles tool execution for both agent and user tools.
    Maintains proper state synchronization with tau2 environment.
    """
    
    def __init__(self, tau2_env, domain: str):
        """
        Initialize tool executor.
        
        Args:
            tau2_env: The tau2 environment instance
            domain: Domain name (retail, airline, telecom)
        """
        self.tau2_env = tau2_env
        self.domain = domain
        
    def execute_agent_tool(self, tool_name: str, tool_args: Dict[str, Any], 
                          state: vf.State) -> Dict[str, Any]:
        """Execute an agent tool and update state."""
        try:
            # Get the tool function
            if hasattr(self.tau2_env.tools, tool_name):
                tool_func = getattr(self.tau2_env.tools, tool_name)
            else:
                return {"error": f"Tool {tool_name} not found"}
                
            # Execute tool
            result = tool_func(**tool_args)
            
            # Sync state after tool execution
            self._sync_agent_state(state)
            
            return {"result": result, "success": True}
            
        except Exception as e:
            return {"error": str(e), "success": False}
            
    def execute_user_tool(self, tool_name: str, tool_args: Dict[str, Any],
                         state: vf.State) -> Dict[str, Any]:
        """Execute a user tool and update state."""
        if self.domain != "telecom" or not hasattr(self.tau2_env, 'user_tools'):
            return {"error": "User tools not available in this domain"}
            
        try:
            # Get the tool function
            if hasattr(self.tau2_env.user_tools, tool_name):
                tool_func = getattr(self.tau2_env.user_tools, tool_name)
            else:
                return {"error": f"User tool {tool_name} not found"}
                
            # Execute tool
            result = tool_func(**tool_args)
            
            # Sync state after tool execution
            self._sync_user_state(state)
            
            return {"result": result, "success": True}
            
        except Exception as e:
            return {"error": str(e), "success": False}
            
    def _sync_agent_state(self, state: vf.State):
        """Sync agent-related environment state."""
        # Update database state if available
        if hasattr(self.tau2_env.tools, 'db'):
            state["env_db"]["agent_db"] = self.tau2_env.tools.db.model_dump()
            
        # Sync any environment changes
        if hasattr(self.tau2_env, 'sync_tools'):
            self.tau2_env.sync_tools()
            
    def _sync_user_state(self, state: vf.State):
        """Sync user-related environment state."""
        # Update user database state if available
        if hasattr(self.tau2_env.user_tools, 'db'):
            state["user_state"]["user_db"] = self.tau2_env.user_tools.db.model_dump()
            
        # Sync any environment changes
        if hasattr(self.tau2_env, 'sync_tools'):
            self.tau2_env.sync_tools()