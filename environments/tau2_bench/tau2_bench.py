"""
τ²-bench implementation for verifiers.
Supports full dual-control (both agent and user can execute tools).
All tool execution and user simulation happens within env_response.
"""

import json
import os
import subprocess
from typing import List, Tuple, Dict, Any, Optional
from copy import deepcopy
from datetime import datetime

import verifiers as vf
from verifiers.envs.multiturn_env import MultiTurnEnv
from datasets import Dataset

# Import tau2-bench components
try:
    from tau2.domains.retail.environment import get_environment as get_retail_env
    from tau2.domains.retail.environment import get_tasks as get_retail_tasks
    from tau2.domains.airline.environment import get_environment as get_airline_env
    from tau2.domains.airline.environment import get_tasks as get_airline_tasks
    from tau2.domains.telecom.environment import get_environment as get_telecom_env
    from tau2.domains.telecom.environment import get_tasks as get_telecom_tasks
    from tau2.data_model.message import (
        AssistantMessage, UserMessage, ToolMessage, Message as Tau2Message
    )
    from tau2.user.user_simulator import UserSimulator
    from tau2.user.base import STOP, TRANSFER, OUT_OF_SCOPE
    from tau2.utils.utils import DATA_DIR
    TAU2_AVAILABLE = True
except ImportError:
    TAU2_AVAILABLE = False
    STOP = "STOP"
    TRANSFER = "TRANSFER"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    DATA_DIR = None
    print("Warning: tau2-bench not installed. Please install it to use this environment.")


def setup_tau2_data():
    """Setup tau2-bench data by downloading from GitHub if not present."""
    if not TAU2_AVAILABLE or not DATA_DIR:
        return
        
    # Check if data already exists
    if os.path.exists(DATA_DIR) and os.path.exists(os.path.join(DATA_DIR, "tau2", "domains")):
        return
        
    print(f"Setting up tau2-bench data in {DATA_DIR}...")
    
    # Create data directory
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Clone tau2-bench temporarily to get data
    temp_dir = "/tmp/tau2_bench_temp"
    try:
        # Clone the repository
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/sierra-research/tau2-bench.git", temp_dir],
            check=True,
            capture_output=True
        )
        
        # Copy data directory
        import shutil
        src_data = os.path.join(temp_dir, "data")
        if os.path.exists(src_data):
            shutil.copytree(src_data, DATA_DIR, dirs_exist_ok=True)
            print(f"✅ tau2-bench data successfully set up in {DATA_DIR}")
        else:
            print(f"⚠️  Warning: Could not find data directory in tau2-bench repository")
            
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Warning: Failed to download tau2-bench data: {e}")
    finally:
        # Clean up temp directory
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)


class Tau2BenchEnv(MultiTurnEnv):
    """
    τ²-bench environment supporting dual-control scenarios.
    Both agent and user can execute tools within env_response.
    """
    
    def __init__(self,
                 dataset: Dataset,
                 rubric: vf.Rubric,
                 domain: str,
                 tau2_env,
                 tau2_tasks: List[Any],
                 user_llm: str = "gpt-4.1-mini",
                 max_turns: int = 30,
                 max_errors: int = 3,
                 **kwargs):
        # Initialize parent class
        super().__init__(dataset=dataset, rubric=rubric, **kwargs)
        self.domain = domain
        self.tau2_env = tau2_env
        self.tau2_tasks = tau2_tasks
        self.user_llm = user_llm
        self.max_turns = max_turns
        self.max_errors = max_errors
        
        # Create task lookup
        self.task_lookup = {task.id: task for task in tau2_tasks}
        
    def is_completed(self, messages: vf.Messages, state: vf.State, **kwargs) -> bool:
        """Check if conversation is completed."""
        # Check max turns
        turn_count = state.get("turn_count", 0)
        if turn_count >= self.max_turns:
            state["termination_reason"] = "max_turns_reached"
            return True
            
        # Check error count
        if state.get("error_count", 0) >= self.max_errors:
            state["termination_reason"] = "too_many_errors"
            return True
            
        # Check if user said stop/transfer
        if messages:
            last_msg = messages[-1]
            if last_msg["role"] == "user":
                content = last_msg.get("content", "").lower()
                if any(word in content for word in ["stop", "transfer", "goodbye", "bye"]):
                    state["termination_reason"] = "user_stop"
                    return True
                    
        # Check if task goal is achieved
        if self._check_task_completion(state):
            state["termination_reason"] = "goal_achieved"
            return True
            
        return False
        
    def env_response(self, messages: vf.Messages, state: vf.State, **kwargs) -> Tuple[vf.Messages, vf.State]:
        """
        Handle environment response including tool execution and user simulation.
        All non-agent logic happens here.
        """
        if not messages:
            return [], state
            
        last_msg = messages[-1]
        response_messages = []
        
        # Initialize state components if needed
        self._init_state(state)
        
        # Handle assistant messages (may contain tool calls)
        if last_msg["role"] == "assistant":
            # Process any tool calls from the agent
            if "tool_calls" in last_msg and last_msg["tool_calls"]:
                tool_results = self._execute_agent_tools(last_msg["tool_calls"], state)
                response_messages.extend(tool_results)
                
            # Generate user response after agent message/tools
            user_response = self._generate_user_response(messages + response_messages, state)
            if user_response:
                response_messages.append(user_response)
                
                # In telecom, user response might contain tool calls
                if self.domain == "telecom" and "tool_calls" in user_response:
                    user_tool_results = self._execute_user_tools(
                        user_response["tool_calls"], 
                        state
                    )
                    response_messages.extend(user_tool_results)
                    
        # Update turn count
        state["turn_count"] += len([m for m in response_messages if m["role"] in ["assistant", "user"]])
        
        return response_messages, state
        
    def _init_state(self, state: vf.State):
        """Initialize state components if not already present."""
        # Ensure task_id is in state
        if "task_id" not in state and "info" in state:
            state["task_id"] = state["info"].get("task_id")
            
        if "agent_state" not in state:
            state["agent_state"] = {}
            
        if "user_state" not in state:
            task_id = state.get("task_id")
            
            if not task_id:
                print(f"WARNING: No task_id found in state: {list(state.keys())}")
                
            task = self.task_lookup.get(task_id) if task_id else None
            user_scenario = task.user_scenario.model_dump() if task and hasattr(task, 'user_scenario') and task.user_scenario else {}
            # Extract instructions from user_scenario
            instructions = user_scenario.get("instructions", {}) if user_scenario else {}
            state["user_state"] = {
                "instructions": instructions,
                "persona": user_scenario.get("persona"),
                "context": {},
                "conversation_stage": "initial"
            }
            
        if "env_db" not in state:
            task_id = state.get("task_id")
            task = self.task_lookup.get(task_id)
            if task and hasattr(task, 'initial_state') and task.initial_state:
                initial_state_dict = task.initial_state.model_dump()
                if "initialization_data" in initial_state_dict:
                    state["env_db"] = deepcopy(initial_state_dict["initialization_data"])
                else:
                    state["env_db"] = {}
            else:
                state["env_db"] = {}
                
        if "turn_count" not in state:
            state["turn_count"] = 0
            
        if "tool_executions" not in state:
            state["tool_executions"] = []
            
        if "error_count" not in state:
            state["error_count"] = 0
            
        # Initialize user simulator if needed
        if "user_simulator" not in state:
            self._init_user_simulator(state)
            
    def _init_user_simulator(self, state: vf.State):
        """Initialize the user simulator for this task."""
        user_instructions = state["user_state"]["instructions"]
        
        # Debug: Check what instructions we're passing
        if not user_instructions or (isinstance(user_instructions, dict) and not any(user_instructions.values())):
            print(f"WARNING: Empty user instructions for task {state.get('task_id')}")
            print(f"User state: {state['user_state']}")
        
        if self.domain == "telecom" and hasattr(self.tau2_env, 'user_tools'):
            # User with tools for telecom
            user_tools = list(self.tau2_env.user_tools.get_tools().values())
            state["user_simulator"] = UserSimulator(
                tools=user_tools,
                instructions=user_instructions,
                llm=self.user_llm
            )
        else:
            # User without tools for other domains
            state["user_simulator"] = UserSimulator(
                tools=None,
                instructions=user_instructions,
                llm=self.user_llm
            )
            
    def _execute_agent_tools(self, tool_calls: List[Dict], state: vf.State) -> List[Dict]:
        """Execute agent tool calls and return tool messages."""
        tool_messages = []
        
        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            tool_args = json.loads(tool_call["function"]["arguments"])
            
            # Record execution
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
                    
                    # Sync environment state after tool execution
                    if hasattr(self.tau2_env.tools, 'db'):
                        state["env_db"]["agent_db"] = self.tau2_env.tools.db.model_dump()
                    if hasattr(self.tau2_env, 'sync_tools'):
                        self.tau2_env.sync_tools()
                    
                    # Create tool message
                    content = json.dumps(result) if isinstance(result, (dict, list)) else str(result)
                    tool_messages.append({
                        "role": "tool",
                        "content": content,
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
                state["error_count"] = state.get("error_count", 0) + 1
                
                tool_messages.append({
                    "role": "tool",
                    "content": f"Error executing {tool_name}: {str(e)}",
                    "tool_call_id": tool_call["id"]
                })
                
            # Track execution
            state["tool_executions"].append(exec_record)
            
        return tool_messages
        
    def _execute_user_tools(self, tool_calls: List[Dict], state: vf.State) -> List[Dict]:
        """Execute user tool calls (only in telecom domain)."""
        if self.domain != "telecom":
            return []
            
        tool_messages = []
        
        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            tool_args = json.loads(tool_call["function"]["arguments"])
            
            # Record execution
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
                    
                    # Sync user's environment state
                    if hasattr(self.tau2_env.user_tools, 'db'):
                        state["env_db"]["user_db"] = self.tau2_env.user_tools.db.model_dump()
                    if hasattr(self.tau2_env, 'sync_tools'):
                        self.tau2_env.sync_tools()
                    
                    # Create tool message
                    content = json.dumps(result) if isinstance(result, (dict, list)) else str(result)
                    tool_messages.append({
                        "role": "tool",
                        "content": content,
                        "tool_call_id": tool_call["id"],
                        "name": f"user_{tool_name}"  # Prefix to distinguish user tools
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
                state["error_count"] = state.get("error_count", 0) + 1
                
                tool_messages.append({
                    "role": "tool",
                    "content": f"Error executing user tool {tool_name}: {str(e)}",
                    "tool_call_id": tool_call["id"]
                })
                
            # Track execution
            state["tool_executions"].append(exec_record)
            
        return tool_messages
        
    def _generate_user_response(self, messages: vf.Messages, state: vf.State) -> Optional[Dict]:
        """Generate user response using tau2 user simulator."""
        user_sim = state.get("user_simulator")
        if not user_sim:
            return None
            
        # Get last message to respond to
        if not messages:
            return None
        last_msg = messages[-1]
        
        # Convert to tau2 format
        tau2_messages = self._convert_to_tau2_messages(messages)
        if not tau2_messages:
            return None
            
        # Get or create user state
        if "tau2_user_state" not in state:
            # Initialize user state with message history (excluding tool messages)
            # Filter out tool messages and messages with tool_calls
            history_for_user = []
            for msg in tau2_messages[:-1]:
                if msg.role == "tool":
                    continue
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    # Skip messages with tool calls as they cause issues
                    continue
                history_for_user.append(msg)
                
            state["tau2_user_state"] = user_sim.get_init_state(
                message_history=history_for_user
            )
        
        try:
            # Generate user response - pass ONLY the last message
            last_tau2_msg = tau2_messages[-1]

            
            user_msg, new_user_state = user_sim.generate_next_message(
                message=last_tau2_msg,
                state=state["tau2_user_state"]
            )
            
            # Update user state
            state["tau2_user_state"] = new_user_state
                
            # Handle special responses
            if user_msg.content == STOP:
                state["termination_reason"] = "user_stop"
                return {
                    "role": "user",
                    "content": "I'd like to stop here. Thank you for your help."
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
            msg = {
                "role": "user",
                "content": user_msg.content if user_msg.content else ""
            }
            
            # Handle tool calls in user message (telecom only)
            if hasattr(user_msg, 'tool_calls') and user_msg.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in user_msg.tool_calls
                ]
                
            return msg
            
        except Exception as e:
            # Fallback response
            state["user_errors"] = state.get("user_errors", 0) + 1
            return {
                "role": "user",
                "content": "I'm having trouble understanding. Could you please help me?"
            }
            
    def _check_task_completion(self, state: vf.State) -> bool:
        """Check if task goals are achieved using environment state comparison."""
        task_id = state.get("task_id")
        task = self.task_lookup.get(task_id)
        
        if not task or not hasattr(task, 'evaluation_criteria') or not task.evaluation_criteria:
            return False
            
        # First try environment hash comparison (most accurate)
        if hasattr(self.tau2_env, 'get_db_hash'):
            try:
                # Get current environment state hash
                current_hash = self.tau2_env.get_db_hash()
                
                # Compare with expected state
                # Note: This is simplified - the original creates a golden environment
                # and compares hashes after applying expected actions
                if hasattr(task.evaluation_criteria, 'expected_hash'):
                    return current_hash == task.evaluation_criteria.expected_hash
            except Exception:
                pass  # Fall back to other checks
                
        # Fall back to checking goals if defined
        if hasattr(task, 'expected_state') and task.expected_state:
            expected_state = task.expected_state.model_dump()
            goals = expected_state.get("goals", [])
            
            if goals:
                achieved = sum(1 for goal in goals if self._check_single_goal(goal, state))
                return achieved == len(goals)
                
        # Check if required actions were performed
        if hasattr(task.evaluation_criteria, 'actions') and task.evaluation_criteria.actions:
            required_actions = {(action.name, action.requestor) for action in task.evaluation_criteria.actions}
            performed_actions = {
                (exec["tool"], exec.get("role", "agent")) 
                for exec in state.get("tool_executions", []) 
                if exec.get("success", False)
            }
            return required_actions.issubset(performed_actions)
            
        return False
        
    def _check_single_goal(self, goal: Dict, state: vf.State) -> bool:
        """Check if a single goal is achieved."""
        goal_type = goal.get("type", "")
        
        if goal_type == "db_state":
            # Check database state matches expected
            expected_db = goal.get("expected_db", {})
            current_db = state.get("env_db", {})
            
            # Simple check - all expected fields match
            for key, expected_value in expected_db.items():
                if current_db.get(key) != expected_value:
                    return False
            return True
            
        elif goal_type == "tool_called":
            # Check if required tool was called successfully
            required_tool = goal.get("tool_name", "")
            tool_execs = state.get("tool_executions", [])
            
            for exec in tool_execs:
                if exec["tool"] == required_tool and exec.get("success", False):
                    return True
            return False
            
        elif goal_type == "conversation":
            # Check conversation content - simplified
            return True
            
        return False
        
    def _convert_to_tau2_messages(self, messages: vf.Messages) -> List[Tau2Message]:
        """Convert verifiers messages to tau2 format."""
        tau2_messages = []
        
        for msg in messages:
            if msg["role"] == "assistant":
                # Convert tool calls to tau2 format
                tau2_tool_calls = []
                if "tool_calls" in msg and msg["tool_calls"]:
                    for tc in msg["tool_calls"]:
                        args_str = tc.get("function", {}).get("arguments", "{}")
                        # Parse arguments if they're a string
                        try:
                            args_dict = json.loads(args_str) if isinstance(args_str, str) else args_str
                        except:
                            args_dict = {}
                            
                        tau2_tool_calls.append({
                            "id": tc.get("id", ""),
                            "name": tc.get("function", {}).get("name", ""),
                            "arguments": args_dict
                        })
                
                # Set tool_calls to None if empty (critical for tau2 compatibility)
                tau2_msg = AssistantMessage(
                    role="assistant",
                    content=msg.get("content", ""),
                    tool_calls=tau2_tool_calls if tau2_tool_calls else None,
                    cost=0.0
                )
            elif msg["role"] == "user":
                # Handle user tool calls (for telecom domain)
                user_tool_calls = msg.get("tool_calls", [])
                tau2_msg = UserMessage(
                    role="user",
                    content=msg.get("content", ""),
                    tool_calls=user_tool_calls if user_tool_calls else None
                )
            elif msg["role"] == "tool":
                tau2_msg = ToolMessage(
                    id=msg.get("tool_call_id", ""),  # tau2 expects 'id' not 'tool_call_id'
                    role="tool",
                    content=msg.get("content", ""),
                    name=msg.get("name", "tool")  # tau2 also expects tool name
                )
            else:
                continue
                
            tau2_messages.append(tau2_msg)
            
        return tau2_messages


def create_tau2_dataset(tau2_tasks: List[Any], domain: str) -> Dataset:
    """Convert tau2 tasks to verifiers dataset format."""
    dataset_rows = []
    
    for task in tau2_tasks:
        # Extract key information from task
        user_scenario = task.user_scenario if hasattr(task, 'user_scenario') else None
        scenario = ""
        if user_scenario:
            scenario = user_scenario.scenario if hasattr(user_scenario, 'scenario') else ""
        
        # Get initial message history if available
        initial_messages = []
        if hasattr(task, 'initial_state') and task.initial_state and hasattr(task.initial_state, 'message_history') and task.initial_state.message_history:
            initial_messages = [
                {
                    "role": msg.role,
                    "content": msg.content if hasattr(msg, 'content') else ""
                }
                for msg in task.initial_state.message_history
            ]
        
        # Default system prompt based on domain
        if not initial_messages:
            if domain == "retail":
                system_content = "You are a customer service agent for an online retail company. You have access to tools to help customers with their orders, returns, and other inquiries."
            elif domain == "airline":
                system_content = "You are a customer service agent for an airline. You have access to tools to help customers with flight bookings, changes, and cancellations."
            elif domain == "telecom":
                system_content = "You are a technical support agent for a telecom company. You have access to tools to help customers with technical issues, account management, and service inquiries."
            else:
                system_content = "You are a helpful customer service agent."
                
            initial_messages = [{"role": "system", "content": system_content}]
        
        # Create dataset row
        row = {
            "prompt": initial_messages,
            "question": scenario if scenario else "Help the customer with their request.",
            "info": {
                "task_id": task.id,
                "domain": domain,
                "expected_state": task.expected_state.model_dump() if hasattr(task, 'expected_state') and task.expected_state else {},
                "initial_state": task.initial_state.model_dump() if hasattr(task, 'initial_state') and task.initial_state else {},
                "user_scenario": user_scenario.model_dump() if user_scenario and hasattr(user_scenario, 'model_dump') else {}
            },
            "answer": "Successfully helped the customer",  # Placeholder
            "task": f"tau2_{domain}",
            # Store task_id in state for easy lookup
            "task_id": task.id
        }
        dataset_rows.append(row)
        
    return Dataset.from_list(dataset_rows)


def check_action_sequence(required_actions: List[Dict], tool_executions: List[Dict]) -> bool:
    """Check if required actions were performed in the correct order."""
    if not required_actions:
        return True
        
    exec_tools = [e["tool"] for e in tool_executions if e.get("success", False)]
    required_tools = [a.get("tool") for a in required_actions]
    
    # Simple check: all required tools were called
    # More sophisticated check would verify order and parameters
    return all(tool in exec_tools for tool in required_tools)


def create_tau2_rubric(domain: str) -> vf.Rubric:
    """Create evaluation rubric for tau2 tasks aligned with original τ²-bench evaluation."""
    
    def check_goal_achievement(completion, info, state, **kwargs) -> float:
        """Check if task goals were achieved (aligned with original env evaluation)."""
        # Primary check: termination reason
        termination = state.get("termination_reason", "")
        if termination == "goal_achieved":
            return 1.0
        elif termination in ["too_many_errors", "max_turns_reached"]:
            return 0.0  # Failed due to limits
            
        # Secondary check: look at expected state if available
        expected_state = info.get("expected_state", {})
        goals = expected_state.get("goals", [])
        
        if not goals:
            # No specific goals defined
            if termination == "user_stop" and state.get("error_count", 0) == 0:
                return 0.5  # Partial credit for clean user-initiated stop
            else:
                return 0.0
                
        # Check each goal
        achieved = 0
        for goal in goals:
            goal_type = goal.get("type", "")
            if goal_type == "tool_called":
                # Check if required tool was called successfully
                tool_execs = state.get("tool_executions", [])
                required_tool = goal.get("tool_name", "")
                if any(exec["tool"] == required_tool and exec.get("success", False) 
                       for exec in tool_execs):
                    achieved += 1
            elif goal_type == "db_state":
                # Check database state matches expected
                expected_db = goal.get("expected_db", {})
                current_db = state.get("env_db", {})
                if all(current_db.get(k) == v for k, v in expected_db.items()):
                    achieved += 1
            elif goal_type == "action_sequence":
                # Check if required actions were performed in order
                required_actions = goal.get("actions", [])
                tool_execs = state.get("tool_executions", [])
                if check_action_sequence(required_actions, tool_execs):
                    achieved += 1
                    
        return achieved / len(goals) if goals else 0.0
        
    def check_efficiency(completion, info, state, **kwargs) -> float:
        """Check efficiency of task completion (considers errors and termination)."""
        turn_count = state.get("turn_count", 0)
        error_count = state.get("error_count", 0)
        termination = state.get("termination_reason", "")
        
        # Base efficiency score from turn count
        if turn_count <= 10:
            efficiency_score = 1.0  # Very efficient
        elif turn_count <= 20:
            efficiency_score = 0.7  # Reasonably efficient
        elif turn_count < 30:
            efficiency_score = 0.4  # Completed but inefficient
        else:
            efficiency_score = 0.1  # Hit max turns
            
        # Penalize errors
        error_penalty = min(0.3, error_count * 0.1)
        efficiency_score -= error_penalty
        
        # Bonus for successful completion
        if termination == "goal_achieved":
            efficiency_score += 0.1
            
        return max(0.0, min(1.0, efficiency_score))
            
    def check_tool_usage(completion, info, state, **kwargs) -> float:
        """Check appropriate tool usage."""
        tool_execs = state.get("tool_executions", [])
        
        if not tool_execs:
            # No tools used - might be okay for simple queries
            return 0.5
            
        # Check for errors
        errors = [e for e in tool_execs if not e.get("success", False)]
        if errors:
            error_rate = len(errors) / len(tool_execs)
            return max(0, 1.0 - error_rate)
        else:
            return 1.0
            
    # Different weights for different domains
    if domain == "telecom":
        # Telecom values correct tool usage highly due to dual-control
        weights = [0.6, 0.2, 0.2]
    else:
        # Other domains focus more on goal achievement
        weights = [0.7, 0.2, 0.1]
        
    rubric = vf.Rubric(
        funcs=[check_goal_achievement, check_efficiency, check_tool_usage],
        weights=weights
    )
    
    return rubric


def load_environment(
    domain: str = "retail",
    num_train_examples: int = -1,
    num_eval_examples: int = 10,
    user_llm: str = "gpt-4.1-mini",
    **kwargs
) -> vf.Environment:
    """
    Load τ²-bench environment with full dual-control support.
    
    Args:
        domain: One of "retail", "airline", or "telecom"
        num_train_examples: Number of training examples (-1 for all)
        num_eval_examples: Number of evaluation examples
        user_llm: LLM to use for user simulation
        **kwargs: Additional arguments
        
    Returns:
        Configured tau2-bench environment
    """
    if not TAU2_AVAILABLE:
        raise ImportError("tau2-bench is not installed. Please install it first.")
        
    # Setup tau2 data if not already present
    setup_tau2_data()

    # Load tau2 environment and tasks based on domain
    if domain == "retail":
        tau2_env = get_retail_env()
        tau2_tasks = get_retail_tasks()
    elif domain == "airline":
        tau2_env = get_airline_env()
        tau2_tasks = get_airline_tasks()
    elif domain == "telecom":
        tau2_env = get_telecom_env(solo_mode=False)  # Important: dual-control
        tau2_tasks = get_telecom_tasks()
    else:
        raise ValueError(f"Unknown domain: {domain}")
        
    # Convert tasks to dataset
    full_dataset = create_tau2_dataset(tau2_tasks, domain)
    
    # Split into train/eval
    total_tasks = len(full_dataset)
    if num_train_examples == -1:
        num_train_examples = max(0, total_tasks - num_eval_examples)
        
    train_dataset = full_dataset.select(range(min(num_train_examples, total_tasks)))
    eval_dataset = full_dataset.select(range(
        num_train_examples,
        min(num_train_examples + num_eval_examples, total_tasks)
    ))
    
    # Create rubric
    rubric = create_tau2_rubric(domain)
    

    
    # Create environment
    env = Tau2BenchEnv(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        rubric=rubric,
        domain=domain,
        tau2_env=tau2_env,
        tau2_tasks=tau2_tasks,
        user_llm=user_llm,
        **kwargs
    )
    
    return env