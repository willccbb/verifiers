"""
Orchestrator implementation for tau2-bench.
Manages the complex interaction between agent, user, and environment.
"""

import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from copy import deepcopy

import verifiers as vf
from verifiers.types import Messages, State

# Import tau2 components
try:
    from tau2.data_model.message import (
        AssistantMessage, UserMessage, ToolMessage,
        MultiToolMessage, Message as Tau2Message
    )
    from tau2.user.user_simulator import UserSimulator
    from tau2.user.base import STOP, TRANSFER, OUT_OF_SCOPE
except ImportError:
    # Define fallbacks if tau2 not available
    STOP = "STOP"
    TRANSFER = "TRANSFER"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


class Tau2Orchestrator:
    """
    Orchestrates the interaction between agent, user, and environment.
    Implements the full tau2-bench orchestration logic within verifiers.
    """
    
    def __init__(self, 
                 tau2_env,
                 domain: str,
                 task: Dict[str, Any],
                 user_llm: str = "gpt-4",
                 max_turns: int = 30,
                 max_errors: int = 10):
        """
        Initialize orchestrator.
        
        Args:
            tau2_env: The tau2 environment instance
            domain: Domain name (retail, airline, telecom)
            task: Current task dictionary
            user_llm: LLM to use for user simulation
            max_turns: Maximum number of turns
            max_errors: Maximum number of errors before stopping
        """
        self.tau2_env = tau2_env
        self.domain = domain
        self.task = task
        self.user_llm = user_llm
        self.max_turns = max_turns
        self.max_errors = max_errors
        
        # Initialize user simulator
        self._init_user_simulator()
        
    def _init_user_simulator(self):
        """Initialize the user simulator based on domain."""
        user_instructions = self.task.get("user_instructions", {})
        
        if self.domain == "telecom" and hasattr(self.tau2_env, 'user_tools'):
            # User with tools for telecom
            user_tools = list(self.tau2_env.user_tools.get_tools().values())
            self.user_sim = UserSimulator(
                tools=user_tools,
                instructions=user_instructions,
                llm=self.user_llm
            )
        else:
            # User without tools for other domains
            self.user_sim = UserSimulator(
                tools=None,
                instructions=user_instructions,
                llm=self.user_llm
            )
            
    def process_agent_message(self, 
                            messages: Messages, 
                            state: State) -> Tuple[Messages, State]:
        """
        Process an agent message and generate appropriate responses.
        
        Args:
            messages: Conversation history
            state: Current state
            
        Returns:
            Tuple of (new_messages, updated_state)
        """
        if not messages:
            return [], state
            
        last_msg = messages[-1]
        if last_msg["role"] != "assistant":
            return [], state
            
        response_messages = []
        
        # Handle tool calls if present
        if "tool_calls" in last_msg and last_msg["tool_calls"]:
            tool_results = self._execute_agent_tools(last_msg["tool_calls"], state)
            response_messages.extend(tool_results)
            
        # Generate user response after processing tools
        user_response = self._generate_user_response(messages + response_messages, state)
        if user_response:
            response_messages.append(user_response)
            
            # Check if user response contains tool calls (telecom only)
            if self.domain == "telecom" and "tool_calls" in user_response:
                user_tool_results = self._execute_user_tools(
                    user_response["tool_calls"], 
                    state
                )
                response_messages.extend(user_tool_results)
                
        return response_messages, state
        
    def process_user_message(self,
                           messages: Messages,
                           state: State) -> Tuple[Messages, State]:
        """
        Process a user message (mainly for handling user tool calls).
        
        Args:
            messages: Conversation history
            state: Current state
            
        Returns:
            Tuple of (new_messages, updated_state)
        """
        if not messages:
            return [], state
            
        last_msg = messages[-1]
        if last_msg["role"] != "user":
            return [], state
            
        response_messages = []
        
        # In telecom domain, user messages might contain tool calls
        if self.domain == "telecom" and "tool_calls" in last_msg and last_msg["tool_calls"]:
            tool_results = self._execute_user_tools(last_msg["tool_calls"], state)
            response_messages.extend(tool_results)
            
        return response_messages, state
        
    def _execute_agent_tools(self, 
                           tool_calls: List[Dict], 
                           state: State) -> List[Dict]:
        """Execute agent tool calls."""
        tool_messages = []
        
        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            tool_args = json.loads(tool_call["function"]["arguments"])
            
            # Track execution
            exec_record = {
                "role": "agent",
                "tool": tool_name,
                "args": tool_args,
                "timestamp": datetime.now().isoformat()
            }
            
            try:
                # Execute tool through tau2 environment
                if hasattr(self.tau2_env.tools, tool_name):
                    tool_func = getattr(self.tau2_env.tools, tool_name)
                    result = tool_func(**tool_args)
                    
                    exec_record["result"] = result
                    exec_record["success"] = True
                    
                    # Sync environment state
                    self._sync_environment_state(state, "agent")
                    
                    tool_messages.append({
                        "role": "tool",
                        "content": json.dumps(result) if isinstance(result, (dict, list)) else str(result),
                        "tool_call_id": tool_call["id"]
                    })
                else:
                    exec_record["error"] = f"Tool {tool_name} not found"
                    exec_record["success"] = False
                    
                    tool_messages.append({
                        "role": "tool",
                        "content": f"Error: Tool {tool_name} not found",
                        "tool_call_id": tool_call["id"]
                    })
                    
            except Exception as e:
                exec_record["error"] = str(e)
                exec_record["success"] = False
                
                tool_messages.append({
                    "role": "tool",
                    "content": f"Error executing {tool_name}: {str(e)}",
                    "tool_call_id": tool_call["id"]
                })
                
            # Track execution
            state["tool_executions"].append(exec_record)
            
        return tool_messages
        
    def _execute_user_tools(self,
                          tool_calls: List[Dict],
                          state: State) -> List[Dict]:
        """Execute user tool calls (telecom only)."""
        if self.domain != "telecom":
            return []
            
        tool_messages = []
        
        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            tool_args = json.loads(tool_call["function"]["arguments"])
            
            # Track execution
            exec_record = {
                "role": "user",
                "tool": tool_name,
                "args": tool_args,
                "timestamp": datetime.now().isoformat()
            }
            
            try:
                # Execute tool through tau2 environment
                if hasattr(self.tau2_env.user_tools, tool_name):
                    tool_func = getattr(self.tau2_env.user_tools, tool_name)
                    result = tool_func(**tool_args)
                    
                    exec_record["result"] = result
                    exec_record["success"] = True
                    
                    # Sync environment state
                    self._sync_environment_state(state, "user")
                    
                    tool_messages.append({
                        "role": "tool",
                        "content": json.dumps(result) if isinstance(result, (dict, list)) else str(result),
                        "tool_call_id": tool_call["id"],
                        "name": f"user_{tool_name}"
                    })
                else:
                    exec_record["error"] = f"User tool {tool_name} not found"
                    exec_record["success"] = False
                    
                    tool_messages.append({
                        "role": "tool",
                        "content": f"Error: User tool {tool_name} not found",
                        "tool_call_id": tool_call["id"]
                    })
                    
            except Exception as e:
                exec_record["error"] = str(e)
                exec_record["success"] = False
                
                tool_messages.append({
                    "role": "tool",
                    "content": f"Error executing user tool {tool_name}: {str(e)}",
                    "tool_call_id": tool_call["id"]
                })
                
            # Track execution
            state["tool_executions"].append(exec_record)
            
        return tool_messages
        
    def _generate_user_response(self,
                              messages: Messages,
                              state: State) -> Optional[Dict]:
        """Generate user response using tau2 user simulator."""
        # Convert messages to tau2 format
        tau2_messages = self._convert_to_tau2_messages(messages)
        
        try:
            # Generate response
            user_msg, new_user_state = self.user_sim.generate_next_message(
                message_history=tau2_messages,
                user_state=state.get("user_state", {})
            )
            
            # Update user state
            if new_user_state:
                state["user_state"].update(new_user_state)
                
            # Handle special responses
            if user_msg.content == STOP:
                state["termination_reason"] = "user_stop"
                return {
                    "role": "user",
                    "content": "I'd like to stop here. Thank you."
                }
            elif user_msg.content == TRANSFER:
                state["termination_reason"] = "user_transfer"
                return {
                    "role": "user",
                    "content": "I'd like to speak to a human agent please."
                }
            elif user_msg.content == OUT_OF_SCOPE:
                return {
                    "role": "user",
                    "content": "I'm not sure that's related to what I need help with."
                }
                
            # Convert to verifiers format
            return self._convert_tau2_message_to_verifiers(user_msg)
            
        except Exception as e:
            # Fallback response
            state["user_errors"] = state.get("user_errors", 0) + 1
            return {
                "role": "user",
                "content": "I'm having trouble understanding. Could you please help me?"
            }
            
    def _sync_environment_state(self, state: State, actor: str):
        """Sync environment state after tool execution."""
        # Sync tau2 environment internal state
        if hasattr(self.tau2_env, 'sync_tools'):
            self.tau2_env.sync_tools()
            
        # Update state based on actor
        if actor == "agent":
            if hasattr(self.tau2_env.tools, 'db'):
                state["env_db"]["agent_db"] = self.tau2_env.tools.db.model_dump()
        elif actor == "user":
            if hasattr(self.tau2_env.user_tools, 'db'):
                state["env_db"]["user_db"] = self.tau2_env.user_tools.db.model_dump()
                
    def _convert_to_tau2_messages(self, messages: Messages) -> List[Tau2Message]:
        """Convert verifiers messages to tau2 format."""
        tau2_messages = []
        
        for msg in messages:
            if msg["role"] == "assistant":
                tau2_msg = AssistantMessage(
                    role="assistant",
                    content=msg.get("content", ""),
                    tool_calls=msg.get("tool_calls", []),
                    cost=0.0  # tau2 tracks costs
                )
            elif msg["role"] == "user":
                tau2_msg = UserMessage(
                    role="user",
                    content=msg.get("content", ""),
                    tool_calls=msg.get("tool_calls", [])
                )
            elif msg["role"] == "tool":
                tau2_msg = ToolMessage(
                    role="tool",
                    content=msg.get("content", ""),
                    tool_call_id=msg.get("tool_call_id", "")
                )
            else:
                continue
                
            tau2_messages.append(tau2_msg)
            
        return tau2_messages
        
    def _convert_tau2_message_to_verifiers(self, tau2_msg: Tau2Message) -> Dict:
        """Convert tau2 message to verifiers format."""
        msg = {
            "role": tau2_msg.role,
            "content": tau2_msg.content if tau2_msg.content else ""
        }
        
        # Handle tool calls
        if hasattr(tau2_msg, 'tool_calls') and tau2_msg.tool_calls:
            msg["tool_calls"] = []
            for tc in tau2_msg.tool_calls:
                msg["tool_calls"].append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                })
                
        return msg
        
    def check_task_completion(self, state: State) -> bool:
        """Check if the task is completed based on expected state."""
        expected_state = self.task.get("expected_state", {})
        if not expected_state:
            return False
            
        # Check goals
        goals = expected_state.get("goals", [])
        if not goals:
            # No explicit goals, check general completion
            return state.get("termination_reason") == "goal_achieved"
            
        # Check each goal
        achieved_goals = 0
        for goal in goals:
            if self._check_single_goal(goal, state):
                achieved_goals += 1
                
        # All goals must be achieved
        return achieved_goals == len(goals)
        
    def _check_single_goal(self, goal: Dict, state: State) -> bool:
        """Check if a single goal is achieved."""
        goal_type = goal.get("type", "")
        
        if goal_type == "db_state":
            # Check database state
            return self._check_db_state_goal(goal, state)
        elif goal_type == "tool_called":
            # Check if specific tool was called
            return self._check_tool_called_goal(goal, state)
        elif goal_type == "conversation":
            # Check conversation content
            return self._check_conversation_goal(goal, state)
        else:
            # Unknown goal type
            return False
            
    def _check_db_state_goal(self, goal: Dict, state: State) -> bool:
        """Check if database state matches expected."""
        # This would need domain-specific implementation
        # For now, simplified check
        expected_db = goal.get("expected_db", {})
        current_db = state.get("env_db", {})
        
        # Check key fields match
        for key, expected_value in expected_db.items():
            if current_db.get(key) != expected_value:
                return False
                
        return True
        
    def _check_tool_called_goal(self, goal: Dict, state: State) -> bool:
        """Check if required tool was called."""
        required_tool = goal.get("tool_name", "")
        tool_execs = state.get("tool_executions", [])
        
        for exec in tool_execs:
            if exec["tool"] == required_tool and exec.get("success", False):
                return True
                
        return False
        
    def _check_conversation_goal(self, goal: Dict, state: State) -> bool:
        """Check if conversation contains required elements."""
        # This would check for specific conversation patterns
        # Simplified for now
        return True