"""
τ²-bench implementation for verifiers.
Supports full dual-control (both agent and user can execute tools).
All tool execution and user simulation happens within env_response.
"""

import json
import os
import subprocess
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime

import verifiers as vf
from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.types import ChatCompletionMessageToolCall
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
        AssistantMessage, UserMessage, ToolMessage, Message as Tau2Message, ToolCall
    )
    from tau2.user.user_simulator import UserSimulator
    from tau2.utils.utils import DATA_DIR
    # Import the evaluators
    from tau2.evaluator.evaluator import evaluate_simulation, EvaluationType
    from tau2.data_model.simulation import SimulationRun, TerminationReason
    TAU2_AVAILABLE = True
except ImportError as e:
    print(f"DEBUG: Import error: {e}")
    import traceback
    traceback.print_exc()
    TAU2_AVAILABLE = False
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
            print("⚠️  Warning: Could not find data directory in tau2-bench repository")
            
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
                 tau2_tasks: List[Any],
                 user_llm: str = "gpt-4.1",
                 max_steps: int = 200,  # tau2's default
                 max_errors: int = 10,  # tau2's default  
                 solo_mode: bool = False,
                 **kwargs):
        super().__init__(dataset=dataset, rubric=rubric, **kwargs)
        self.solo_mode = solo_mode
        self.domain = domain
        self.tau2_tasks = tau2_tasks
        self.user_llm = user_llm
        self.max_steps = max_steps
        self.max_errors = max_errors
        
        # Create task lookup
        self.task_lookup = {task.id: task for task in tau2_tasks}
        
        # Store domain and configuration needed to create fresh environments
        self.env_config = {
            "domain": domain,
            "solo_mode": solo_mode
        }

    def _create_fresh_env(self, initial_db_state: dict = None):
        """Create a fresh tau2 environment instance with its own isolated database."""
        # Create fresh environment with fresh database for each instance
        if self.domain == "retail":
            # This creates a fresh RetailDB instance
            env = get_retail_env()
        elif self.domain == "airline":
            env = get_airline_env()
        elif self.domain == "telecom":
            env = get_telecom_env(solo_mode=self.solo_mode)
        else:
            raise ValueError(f"Unknown domain: {self.domain}")
        
        # Set initial database state if provided
        if initial_db_state:
            env.set_state(initial_db_state)
            
        return env

    def is_completed(self, messages: vf.Messages, state: vf.State, **kwargs) -> bool:
        """Check if conversation should end based on tau2's termination criteria."""
        # Initialize state if needed (only once)
        if "tau2_env" not in state:
            self._init_state(state)
            
        # Don't terminate if there are pending tool calls
        if messages:
            last_msg = messages[-1]
            if last_msg.get("role") == "assistant" and last_msg.get("tool_calls"):
                # Assistant made tool calls - need to execute them first
                return False
                
        # Check max steps
        step_count = state.get("step_count", 0)
        if step_count >= self.max_steps:
            state["termination_reason"] = "max_steps"
            return True
        
        # Check error count
        error_count = state.get("error_count", 0)
        if error_count >= self.max_errors:
            state["termination_reason"] = "max_errors"
            return True
        
        # Get tau2 environment and simulators
        tau2_env = state.get("tau2_env")
        user_simulator = state.get("user_simulator")
        if not tau2_env or not user_simulator:
            return False
            
        # Check user simulator stop conditions
        tau2_user_state = state.get("tau2_user_state", {})
        if tau2_user_state:
            try:
                if user_simulator.is_stop(tau2_user_state):
                    state["termination_reason"] = "user_stop"
                    return True
            except Exception:
                pass
                
        # Check agent stop conditions
        agent_state = state.get("agent_state")
        if agent_state:
            try:
                # Import LLMAgent for is_stop check
                from tau2.agent.llm_agent import LLMAgent
                if LLMAgent.is_stop(agent_state):
                    state["termination_reason"] = "agent_stop"
                    return True
            except Exception:
                pass
        
        return False
        
    def env_response(self, messages: vf.Messages, state: vf.State, **kwargs) -> Tuple[vf.Messages, vf.State]:
        """Generate environment response based on tau2 logic."""
        response_messages = []
        
        # Get the last message to determine response
        if not messages:
            return response_messages, state
            
        last_msg = messages[-1]
        
        # Handle assistant messages
        if last_msg["role"] == "assistant":
            # Process any tool calls from the agent
            if "tool_calls" in last_msg and last_msg["tool_calls"]:
                tool_results = self._execute_agent_tools(last_msg["tool_calls"], state)
                response_messages.extend(tool_results)
                # After tool results, the verifiers framework will call the model again
                # and then we'll generate user response in the next env_response call
            else:
                # No tool calls - generate user response
                user_response = self._generate_user_response(messages, state)
                if user_response:
                    response_messages.append(user_response)
                    
                    # In telecom, user response might contain tool calls
                    if self.domain == "telecom" and "tool_calls" in user_response:
                        user_tool_results = self._execute_user_tools(
                            user_response["tool_calls"], 
                            state
                        )
                        response_messages.extend(user_tool_results)
                    
        # Update step count - count EVERY message like tau2 does
        state["step_count"] = state.get("step_count", 0) + len(response_messages)
        
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
            
        if "step_count" not in state:
            state["step_count"] = 0
            
        if "tau2_env" not in state:
            # Create fresh environment
            state["tau2_env"] = self._create_fresh_env()
            
            # Use tau2's set_state method to initialize properly
            task_id = state.get("task_id")
            task = self.task_lookup.get(task_id) if task_id else None
            
            if task and hasattr(task, 'initial_state') and task.initial_state:
                # Initialize the environment with data and actions only
                # Do NOT replay message history - tau2's evaluator will do that
                initialization_data = task.initial_state.initialization_data if hasattr(task.initial_state, 'initialization_data') else None
                initialization_actions = task.initial_state.initialization_actions if hasattr(task.initial_state, 'initialization_actions') else None
                
                # Use set_state but with empty message history
                state["tau2_env"].set_state(
                    initialization_data=initialization_data,
                    initialization_actions=initialization_actions,
                    message_history=[]  # Empty - don't replay past messages
                )
                
                # Store initial database hashes after initialization
                state["initial_db_hash"] = state["tau2_env"].get_db_hash()
                state["initial_user_db_hash"] = state["tau2_env"].get_user_db_hash()
            else:
                # No initial state - store empty hashes
                state["initial_db_hash"] = state["tau2_env"].get_db_hash()
                state["initial_user_db_hash"] = state["tau2_env"].get_user_db_hash()
                
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
        
        tau2_env = state.get("tau2_env")
        if self.domain == "telecom" and tau2_env and hasattr(tau2_env, 'user_tools'):
            # User with tools for telecom
            user_tools = list(tau2_env.user_tools.get_tools().values())
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
            
    def _execute_agent_tools(self, tool_calls: List[Any], state: vf.State) -> List[Dict]:
        """Execute agent tool calls and return tool messages."""
        tool_messages = []
        
        for tool_call in tool_calls:
            # Handle both dict and object formats
            if isinstance(tool_call, ChatCompletionMessageToolCall):
                # This is the OpenAI object format
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                tool_id = tool_call.id
            elif hasattr(tool_call, 'function'):
                # Generic object format
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                tool_id = getattr(tool_call, 'id', f"tool_call_{tool_name}_{datetime.now().timestamp()}")
            else:
                # Dictionary format
                tool_name = tool_call["function"]["name"]
                tool_args = json.loads(tool_call["function"]["arguments"])
                tool_id = tool_call.get("id", f"tool_call_{tool_name}_{datetime.now().timestamp()}")
            
            # Create tau2 ToolCall object
            tau2_tool_call = ToolCall(
                id=tool_id,
                name=tool_name,
                arguments=tool_args,
                requestor="assistant"
            )
            
            # Get database hash before execution
            tau2_env = state["tau2_env"]
            db_hash_before = tau2_env.get_db_hash()
            
            # Debug: Check order status if it's an order-related tool
            if tool_name in ["return_delivered_order_items", "exchange_delivered_order_items", "modify_pending_order_items"]:
                order_id = tool_args.get("order_id")
                if order_id and hasattr(tau2_env, 'tools') and hasattr(tau2_env.tools, 'db'):
                    try:
                        order = tau2_env.tools.db.orders.get(order_id)
                        if order:
                            pass
                    except Exception:
                        pass
            
            # Use tau2's get_response method directly
            tool_response = tau2_env.get_response(tau2_tool_call)
            
            # Debug: Check order status after
            if tool_name in ["return_delivered_order_items", "exchange_delivered_order_items", "modify_pending_order_items"]:
                order_id = tool_args.get("order_id") 
                if order_id and hasattr(tau2_env, 'tools') and hasattr(tau2_env.tools, 'db'):
                    try:
                        order = tau2_env.tools.db.orders.get(order_id)
                        if order:
                            pass
                    except Exception:
                        pass
            
            # Debug: Log the response
            if tool_name == "modify_pending_order_items":
                pass
            
            if tool_response.error:
                pass
            
            # Debug: Check order status after execution
            if tool_name in ["return_delivered_order_items", "exchange_delivered_order_items"] and not tool_response.error:
                order_id = tool_args.get("order_id")
                if order_id == "#W2378156":
                    try:
                        order = tau2_env.tools.db.orders.get(order_id)
                        if order:
                            pass
                    except Exception:
                        pass
            
            # Get database hashes after execution
            db_hash_after = tau2_env.get_db_hash()
            
            # Track execution for evaluation
            exec_record = {
                "role": "assistant",
                "tool": tool_name,
                "arguments": tool_args,
                "timestamp": datetime.now().isoformat(),
                "requestor": "assistant",
                "result": tool_response.content,
                "error": tool_response.error,
                "db_hash_before": db_hash_before,
                "db_hash_after": db_hash_after,
                "db_changed": db_hash_before != db_hash_after
            }
            state["tool_executions"].append(exec_record)
            
            # Update error count if needed
            if tool_response.error:
                state["error_count"] = state.get("error_count", 0) + 1
            
            # Store current database hash
            state["current_db_hash"] = db_hash_after
            
            # Convert tau2 response to verifiers format
            tool_messages.append({
                "role": "tool",
                "content": tool_response.content,
                "tool_call_id": tool_id,
                "name": tool_name,  # Keep tool name for debugging
                "error": tool_response.error  # Preserve error flag
            })
            
        return tool_messages
        
    def _execute_user_tools(self, tool_calls: List[Any], state: vf.State) -> List[Dict]:
        """Execute user tool calls (only in telecom domain)."""
        if self.domain != "telecom":
            return []
            
        tool_messages = []
        
        for tool_call in tool_calls:
            if isinstance(tool_call, ChatCompletionMessageToolCall):
                # This is the OpenAI object format
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                tool_id = tool_call.id
            elif hasattr(tool_call, 'function'):
                # Generic object format
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                tool_id = getattr(tool_call, 'id', f"tool_call_{tool_name}_{datetime.now().timestamp()}")
            else:
                # Dictionary format
                tool_name = tool_call.get("function", {}).get("name", "")
                tool_args = json.loads(tool_call.get("function", {}).get("arguments", "{}"))
                tool_id = tool_call.get("id", f"tool_call_{tool_name}_{datetime.now().timestamp()}")
                
            # Create tau2 ToolCall object
            tau2_tool_call = ToolCall(
                id=tool_id,
                name=tool_name,
                arguments=tool_args,
                requestor="user"
            )
            
            # Get database hashes before execution
            tau2_env = state["tau2_env"]
            user_db_hash_before = tau2_env.get_user_db_hash()
            
            # Use tau2's get_response method directly
            tool_response = tau2_env.get_response(tau2_tool_call)
            
            # Get database hashes after execution
            user_db_hash_after = tau2_env.get_user_db_hash()
            
            # Track execution for evaluation
            exec_record = {
                "role": "user",
                "tool": tool_name,
                "arguments": tool_args,
                "timestamp": datetime.now().isoformat(),
                "requestor": "user",
                "result": tool_response.content,
                "error": tool_response.error,
                "user_db_hash_before": user_db_hash_before,
                "user_db_hash_after": user_db_hash_after,
                "user_db_changed": user_db_hash_before != user_db_hash_after
            }
            state["tool_executions"].append(exec_record)
            
            # Update error count if needed
            if tool_response.error:
                state["error_count"] = state.get("error_count", 0) + 1
            
            # Store current user database hash
            state["current_user_db_hash"] = user_db_hash_after
            
            # Convert tau2 response to verifiers format
            tool_messages.append({
                "role": "tool",
                "content": tool_response.content,
                "tool_call_id": tool_id,
                "name": f"user_{tool_name}",  # Prefix to distinguish user tools
                "error": tool_response.error  # Preserve error flag
            })
            
        return tool_messages
        
    def _generate_user_response(self, messages: vf.Messages, state: vf.State) -> Optional[Dict]:
        """Generate user response using tau2 user simulator."""
        user_sim = state.get("user_simulator")
        if not user_sim:
            return None
            
        # Get last message to respond to
        if not messages:
            return None
        
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
                
            # Convert to verifiers format - no special handling needed
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
            
        except Exception:
            # Fallback response
            state["user_errors"] = state.get("user_errors", 0) + 1
            return {
                "role": "user",
                "content": "I'm having trouble understanding. Could you please help me?"
            }
            
    def _check_task_completion(self, state: vf.State) -> bool:
        """Check if task goals are achieved using tau2's environment state comparison."""
        task_id = state.get("task_id")
        task = self.task_lookup.get(task_id)
        
        if not task or not hasattr(task, 'evaluation_criteria') or not task.evaluation_criteria:
            return False
            
        # Let the evaluation handle the final assessment to prevent premature termination
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
                        # tc should be a tau2 ToolCall or similar object
                        # But in dataset creation, these come from tau2 message history
                        # so they might be in a different format
                        if hasattr(tc, 'name') and hasattr(tc, 'arguments'):
                            # Direct tau2 ToolCall format
                            tau2_tool_calls.append({
                                "id": getattr(tc, 'id', ""),
                                "name": tc.name,
                                "arguments": tc.arguments if isinstance(tc.arguments, dict) else {}
                            })
                        elif hasattr(tc, 'function'):
                            # OpenAI format
                            args_str = tc.function.arguments
                            tc_id = tc.id
                            tc_name = tc.function.name
                            
                            # Parse arguments if they're a string
                            try:
                                args_dict = json.loads(args_str) if isinstance(args_str, str) else args_str
                            except (json.JSONDecodeError, TypeError):
                                args_dict = {}
                                
                            tau2_tool_calls.append({
                                "id": tc_id,
                                "name": tc_name,
                                "arguments": args_dict
                            })
                        else:
                            # Skip unknown format
                            print(f"WARNING: Unknown tool call format: {type(tc)}")
                            continue
                
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


def create_tau2_dataset(domain: str = "retail") -> Dataset:
    """Create a dataset from tau2 tasks using tau2's native functions."""
    from tau2.domains.retail.environment import get_tasks as get_retail_tasks
    from tau2.domains.airline.environment import get_tasks as get_airline_tasks
    from tau2.domains.telecom.environment import get_tasks as get_telecom_tasks
    from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT
    
    # Get tasks using tau2's native functions
    if domain == "retail":
        tau2_tasks = get_retail_tasks()
        tau2_env = get_retail_env()
    elif domain == "airline":
        tau2_tasks = get_airline_tasks()
        tau2_env = get_airline_env()
    elif domain == "telecom":
        tau2_tasks = get_telecom_tasks()
        tau2_env = get_telecom_env(solo_mode=False)
    else:
        raise ValueError(f"Unknown domain: {domain}")
    
    # Get tools using tau2's environment method
    tools = tau2_env.get_tools()
    
    # Get policy from environment
    policy = tau2_env.policy
    
    # Build the system prompt exactly as tau2 does
    system_prompt = SYSTEM_PROMPT.format(
        agent_instruction=AGENT_INSTRUCTION,
        domain_policy=policy
    )
    
    # Convert to OpenAI format - store as JSON string to avoid HF Dataset schema inference issues
    oai_tools = [tool.openai_schema for tool in tools] if tools else []
    
    dataset_rows = []
    for task in tau2_tasks:
        # Get initial messages from task
        initial_messages = []
        if hasattr(task, 'initial_state') and task.initial_state:
            if hasattr(task.initial_state, 'message_history') and task.initial_state.message_history:
                # Convert tau2 messages to verifiers format - preserve full structure
                for msg in task.initial_state.message_history:
                    if hasattr(msg, 'role') and hasattr(msg, 'content'):
                        verifiers_msg = {
                            "role": msg.role,
                            "content": msg.content
                        }
                        
                        # Preserve tool_calls if present (for assistant messages)
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            verifiers_msg["tool_calls"] = []
                            for tc in msg.tool_calls:
                                verifiers_msg["tool_calls"].append({
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.name,
                                        "arguments": json.dumps(tc.arguments)
                                    }
                                })
                        
                        # Preserve tool_call_id if present (for tool messages)
                        if msg.role == "tool" and hasattr(msg, 'id'):
                            verifiers_msg["tool_call_id"] = msg.id
                            if hasattr(msg, 'name'):
                                verifiers_msg["name"] = msg.name
                            
                        initial_messages.append(verifiers_msg)
        
        # Always start with the exact system prompt from original
        initial_messages = [{"role": "system", "content": system_prompt}] + initial_messages
        
        # Get scenario description
        scenario = ""
        if hasattr(task, 'user_scenario') and task.user_scenario:
            # Use the full string representation of user_scenario
            # This includes both persona and instructions
            scenario = str(task.user_scenario)
        
        # Check if we have a user message in initial_messages
        has_user_message = any(msg.get("role") == "user" for msg in initial_messages[1:])  # Skip system message
        
        # If no user message but we have a scenario, add it
        # This ensures the agent sees the user's request with order IDs etc.
        if not has_user_message and scenario:
            print(f"DEBUG: Adding user scenario to prompt for task {task.id}: {scenario[:100]}...")
            initial_messages.append({
                "role": "user",
                "content": scenario
            })
        else:
            print(f"DEBUG: Task {task.id} - has_user_message: {has_user_message}, has scenario: {bool(scenario)}")
        
        # Create dataset row
        row = {
            "prompt": initial_messages,
            "question": scenario if scenario else "Help the customer with their request.",
            "info": {
                "task_id": task.id,
                "domain": domain,
                "expected_state": task.expected_state.model_dump() if hasattr(task, 'expected_state') and task.expected_state else {},
                "initial_state": task.initial_state.model_dump() if hasattr(task, 'initial_state') and task.initial_state else {},
                "user_scenario": task.user_scenario.model_dump() if hasattr(task, 'user_scenario') and task.user_scenario else {},
                "evaluation_criteria": task.evaluation_criteria.model_dump() if hasattr(task, 'evaluation_criteria') and task.evaluation_criteria else {},
                "oai_tools": json.dumps(oai_tools)  # Store as JSON string
            },
            "answer": "Successfully helped the customer",  # Placeholder
            # Store task_id in state for easy lookup  
            "task_id": task.id
        }
        
        # Debug: Show what the agent will see
        if task.id in [5, 13]:  # Debug specific failing tasks
            print(f"\n{'='*80}")
            print(f"DEBUG: Task {task.id} initial messages:")
            for j, msg in enumerate(initial_messages):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                print(f"  {j}. {role}: {content[:200]}...")
                # Also show tool calls if present
                if "tool_calls" in msg and msg["tool_calls"]:
                    for tc in msg["tool_calls"]:
                        if isinstance(tc, dict):
                            func_name = tc.get("function", {}).get("name", "unknown")
                            print(f"     -> tool call: {func_name}")
            print(f"Scenario: {scenario[:200] if scenario else 'None'}...")
            print(f"{'='*80}\n")
        
        dataset_rows.append(row)
    
    return Dataset.from_list(dataset_rows)





def create_tau2_rubric(domain: str) -> vf.Rubric:
    """Create evaluation rubric that uses tau2-bench's official evaluation logic."""
    
    def evaluate_tau2_task(prompt, completion, info, state, **kwargs) -> float:
        """
        Evaluate task using tau2-bench's official evaluation logic.
        Returns 1.0 for pass, 0.0 for fail (no partial credit).
        """
        print("\n!!! EVALUATE_TAU2_TASK CALLED !!!")
        print(f"State keys: {list(state.keys()) if state else 'No state'}")
        print(f"Info keys: {list(info.keys()) if info else 'No info'}")
        
        # Get task info
        task_id = state.get("task_id") or info.get("task_id")
        if not task_id:
            print("DEBUG: No task_id found, returning 0.0")
            return 0.0
            
        print(f"DEBUG: Task ID = {task_id}")
        print(f"DEBUG: Domain = {domain}")
        
        # Ensure we have the current rollout's messages
        if isinstance(completion, list):
            print(f"DEBUG: Completion has {len(completion)} messages")
            actual_step_count = len(completion)
            state_step_count = state.get("step_count", 0)
            print(f"DEBUG: State step_count = {state_step_count}, Actual completion length = {actual_step_count}")
            
        # Get the original task from tau2
        if domain == "retail":
            tasks = get_retail_tasks()
        elif domain == "airline":
            tasks = get_airline_tasks()
        elif domain == "telecom":
            tasks = get_telecom_tasks()
        else:
            print(f"DEBUG: Unknown domain {domain}, returning 0.0")
            return 0.0
            
        print(f"DEBUG: Found {len(tasks)} tasks for domain {domain}")
            
        task = next((t for t in tasks if t.id == task_id), None)
        if not task:
            print(f"DEBUG: Task {task_id} not found in tasks, returning 0.0")
            return 0.0
            
        print(f"DEBUG: Found task {task_id}")
        
        try:
            # Create a SimulationRun object from our state and messages
            termination_reason = state.get("termination_reason", "")
            if termination_reason == "too_many_errors":
                term_reason = TerminationReason.TOO_MANY_ERRORS
            elif termination_reason == "max_steps":
                term_reason = TerminationReason.MAX_STEPS
            elif termination_reason == "user_stop":
                term_reason = TerminationReason.USER_STOP
            elif termination_reason == "agent_stop":
                term_reason = TerminationReason.AGENT_STOP
            else:
                term_reason = TerminationReason.AGENT_STOP
                
            print(f"DEBUG: Termination reason = {term_reason}")
                
            # Build list of all messages for simulation
            tau2_messages = []
            
            # Debug: log what we're building
            print("!!! Building simulation messages !!!")
            print(f"Prompt type: {type(prompt)}, length: {len(prompt) if isinstance(prompt, list) else 'N/A'}")
            print(f"Completion type: {type(completion)}, length: {len(completion) if isinstance(completion, list) else 'N/A'}")
            
            # NOTE: We do NOT include prompt messages in the simulation!
            # The initial message history is already embedded in the task and will be
            # replayed by tau2's evaluator when it calls set_state.
            # We only include the NEW messages from our rollout (completion).
            
            # Include all messages from the completion (the rollout)
            if isinstance(completion, list):
                print(f"\n!!! Processing {len(completion)} completion messages !!!")
                for i, msg in enumerate(completion):
                    # Skip system messages
                    if msg.get("role") == "system":
                        print(f"  Skipping system message {i}")
                        continue
                        
                    # Ensure message has a role
                    if not msg.get("role"):
                        print(f"  WARNING: Message {i} has no role! Skipping: {msg}")
                        continue
                        
                    # Convert each message to tau2 format
                    if msg.get("role") == "assistant":
                        print(f"  Message {i}: assistant, has_tool_calls={bool(msg.get('tool_calls'))}")
                        tau2_msg = AssistantMessage(
                            role="assistant",
                            content=msg.get("content", "")
                        )
                        # Handle tool calls
                        if msg.get("tool_calls"):
                            tool_calls = []
                            for tc in msg["tool_calls"]:
                                tool_calls.append(ToolCall(
                                    id=tc.id,
                                    name=tc.function.name,
                                    arguments=json.loads(tc.function.arguments),
                                    requestor="assistant"
                                ))
                            tau2_msg.tool_calls = tool_calls
                            print(f"    Added {len(tool_calls)} tool calls")
                        tau2_messages.append(tau2_msg)
                        
                    elif msg.get("role") == "user":
                        print(f"  Message {i}: user")
                        tau2_msg = UserMessage(
                            role="user",
                            content=msg.get("content", "")
                        )
                        # Handle tool calls for user messages (telecom domain)
                        if msg.get("tool_calls"):
                            tool_calls = []
                            for tc in msg["tool_calls"]:
                                tool_calls.append(ToolCall(
                                    id=tc.id,
                                    name=tc.function.name,
                                    arguments=json.loads(tc.function.arguments),
                                    requestor="user"
                                ))
                            tau2_msg.tool_calls = tool_calls
                            print(f"    Added {len(tool_calls)} user tool calls")
                        tau2_messages.append(tau2_msg)
                        
                    elif msg.get("role") == "tool":
                        print(f"  Message {i}: tool, name={msg.get('name')}")
                        # Determine requestor based on context - look at previous message
                        requestor = "assistant"  # default
                        if i > 0 and tau2_messages and hasattr(tau2_messages[-1], 'role'):
                            prev_msg_role = tau2_messages[-1].role
                            if prev_msg_role == "user":
                                requestor = "user"
                        
                        tau2_msg = ToolMessage(
                            role="tool",
                            id=msg.get("tool_call_id"),
                            content=msg.get("content", ""),
                            requestor=requestor,
                            error=msg.get("error", False)  # Use actual error flag from execution
                        )
                        tau2_messages.append(tau2_msg)
            
            print(f"\n!!! Total tau2_messages: {len(tau2_messages)} !!!")
            
            # Validate tool call/message pairing
            tool_call_count = 0
            tool_msg_count = 0
            print("\n!!! Message sequence !!!")
            for j, msg in enumerate(tau2_messages):
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    tool_call_count += len(msg.tool_calls)
                    print(f"  {j}: {msg.role} with {len(msg.tool_calls)} tool calls")
                    for tc in msg.tool_calls:
                        print(f"      -> {tc.name}(id={tc.id})")
                elif isinstance(msg, ToolMessage):
                    tool_msg_count += 1
                    print(f"  {j}: tool response (id={msg.id}, error={msg.error})")
                else:
                    print(f"  {j}: {msg.role}")
            
            if tool_call_count != tool_msg_count:
                print(f"WARNING: Tool call/message mismatch! {tool_call_count} calls vs {tool_msg_count} messages")
                
                # Find orphaned tool calls
                tool_call_ids = []
                tool_msg_ids = []
                for msg in tau2_messages:
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        for tc in msg.tool_calls:
                            tool_call_ids.append(tc.id)
                    elif isinstance(msg, ToolMessage):
                        tool_msg_ids.append(msg.id)
                
                missing_responses = set(tool_call_ids) - set(tool_msg_ids)
                if missing_responses:
                    print(f"ERROR: Tool calls without responses: {missing_responses}")
                    # This will cause tau2 evaluation to fail
                    return 0.0
            
            for i, msg in enumerate(tau2_messages[:5]):  # First 5 messages
                print(f"  {i}: {msg.role} - {msg.content[:50] if msg.content else 'No content'}...")
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"    Tool call: {tc.name}({tc.arguments})")
            if len(tau2_messages) > 5:
                print(f"  ... and {len(tau2_messages) - 5} more messages")
            
            # Build simulation run
            task_id = info.get("task_id", "unknown")
            
            # Debug: Print expected vs actual for this task
            if hasattr(task, 'expected_state') and task.expected_state:
                if hasattr(task.expected_state, 'actions') and task.expected_state.actions:
                    print(f"\n{'='*80}")
                    print(f"EVALUATION DEBUG for task {task_id}")
                    print(f"{'='*80}")
                    
                    print(f"\nEXPECTED ACTIONS ({len(task.expected_state.actions)}):")
                    for i, action in enumerate(task.expected_state.actions):
                        print(f"  {i+1}. {action.requestor}: {action.name}({action.arguments})")
                    
                    # Extract actual tool calls from tau2_messages
                    actual_calls = []
                    for msg in tau2_messages:
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            for tc in msg.tool_calls:
                                actual_calls.append((tc.requestor, tc.name, tc.arguments))
                    
                    print("\nACTUAL TOOL CALLS:")
                    for i, (requestor, name, args) in enumerate(actual_calls):
                        print(f"  {i+1}. {requestor}: {name}({args})")
                    
                    # Detailed comparison
                    print("\nDETAILED ACTION MATCHING:")
                    matched_indices = set()
                    for exp_action in task.expected_state.actions:
                        print(f"\nLooking for: {exp_action.name} by {exp_action.requestor}")
                        print(f"  Expected args: {exp_action.arguments}")
                        
                        found = False
                        for j, (act_req, act_name, act_args) in enumerate(actual_calls):
                            if j not in matched_indices and act_name == exp_action.name and act_req == exp_action.requestor:
                                if act_args == exp_action.arguments:
                                    print(f"  ✓ MATCHED with {act_name} call")
                                    matched_indices.add(j)
                                    found = True
                                    break
                                else:
                                    print(f"  ✗ Args mismatch with {act_name}: expected {exp_action.arguments}, got {act_args}")
                        
                        if not found:
                            print("  ✗ NOT FOUND")
            
            print("\nRunning tau2 evaluation...")
            simulation = SimulationRun(
                id=f"verifiers_eval_{task_id}_{datetime.now().isoformat()}",
                agent_id="verifiers_agent",
                task_id=task_id,
                messages=tau2_messages,
                termination_reason=term_reason,
                task_completed=state.get("termination_reason") == "agent_stop", # Use agent_stop for completion
                errors=state.get("error_count", 0),
                num_steps=state.get("step_count", 0),
                cost=0.0,  # We don't track cost in verifiers
                timestamp=datetime.now().isoformat(),
                start_time=datetime.now().isoformat(),
                end_time=datetime.now().isoformat(),
                duration=0.0,  # We don't track duration in verifiers
                metadata={}
            )
            
            # Check actions
            expected_actions = []
            if task.evaluation_criteria and task.evaluation_criteria.actions:
                expected_actions = task.evaluation_criteria.actions
                
            # Print detailed comparison
            print("\n================================================================================")
            print(f"EVALUATION DEBUG for task {task_id}")
            print("================================================================================")
            
            print(f"\nEXPECTED ACTIONS ({len(expected_actions)}):")
            for i, action in enumerate(expected_actions, 1):
                print(f"  {i}. {action.requestor}: {action.name}({action.arguments})")
                if action.compare_args:
                    print(f"     Compare only: {action.compare_args}")
                    
            print("\nACTUAL TOOL CALLS:")
            actual_count = 0
            for exec_record in state.get("tool_executions", []):
                actual_count += 1
                print(f"  {actual_count}. {exec_record.get('requestor', 'unknown')}: {exec_record['tool']}({exec_record['arguments']})")
                
            # Try to match each expected action
            print("\nDETAILED ACTION MATCHING:")
            for action in expected_actions:
                print(f"\nLooking for: {action.name} by {action.requestor}")
                print(f"  Expected args: {action.arguments}")
                if action.compare_args:
                    print(f"  Compare only: {action.compare_args}")
                    
                found = False
                for exec_record in state.get("tool_executions", []):
                    if exec_record['tool'] == action.name and exec_record.get('requestor', 'assistant') == action.requestor:
                        # Check arguments
                        if action.compare_args:
                            # Only compare specified args
                            expected_subset = {k: v for k, v in action.arguments.items() if k in action.compare_args}
                            actual_subset = {k: v for k, v in exec_record['arguments'].items() if k in action.compare_args}
                            if expected_subset == actual_subset:
                                found = True
                                print(f"  ✓ MATCHED with {exec_record['tool']} call")
                                break
                            else:
                                print(f"  ✗ Args mismatch with {exec_record['tool']}: expected {expected_subset}, got {actual_subset}")
                        else:
                            # Compare all args
                            if exec_record['arguments'] == action.arguments:
                                found = True
                                print(f"  ✓ MATCHED with {exec_record['tool']} call")
                                break
                            else:
                                print(f"  ✗ Args mismatch with {exec_record['tool']}: expected {action.arguments}, got {exec_record['arguments']}")
                                
                if not found:
                    print("  ✗ NOT FOUND")
                    
            # Also run the tau2 evaluation
            print("\nRunning tau2 evaluation...")
            
            # Use tau2-bench's official evaluation
            reward_info = evaluate_simulation(
                simulation=simulation,
                task=task,
                evaluation_type=EvaluationType.ALL,
                solo_mode=False,  # All domains use False for solo_mode
                domain=domain
            )
            
            # Log evaluation results
            print("\nEVALUATION RESULTS:")
            print(f"  Final reward: {reward_info.reward}")
            if hasattr(reward_info, 'reward_breakdown') and reward_info.reward_breakdown:
                print(f"  Reward breakdown: {reward_info.reward_breakdown}")
            
            # Log specific check results
            if hasattr(reward_info, 'action_checks') and reward_info.action_checks:
                print("\n  Action checks:")
                for check in reward_info.action_checks:
                    status = "✓" if check.action_match else "✗"
                    print(f"    {status} {check.action.name} - Match: {check.action_match}")
                    if not check.action_match:
                        print(f"       Expected args: {check.action.arguments}")
                        if check.action.compare_args:
                            print(f"       Compare only: {check.action.compare_args}")
                        # Try to find closest match
                        print(f"       Looking for tool call with name: {check.action.name}")
                        found_with_name = False
                        for msg in tau2_messages:
                            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                for tc in msg.tool_calls:
                                    if tc.name == check.action.name:
                                        found_with_name = True
                                        print(f"       Found call with args: {tc.arguments}")
                                        # Show which args differ
                                        if check.action.compare_args:
                                            for arg in check.action.compare_args:
                                                expected = check.action.arguments.get(arg)
                                                actual = tc.arguments.get(arg)
                                                if expected != actual:
                                                    print(f"         - {arg}: expected '{expected}', got '{actual}'")
                        if not found_with_name:
                            print(f"       No tool call found with name '{check.action.name}'")
            
            if hasattr(reward_info, 'db_check') and reward_info.db_check:
                print(f"\n  DB check: {reward_info.db_check}")
                
            if hasattr(reward_info, 'nl_assertions') and reward_info.nl_assertions:
                print(f"\n  NL assertions: {len(reward_info.nl_assertions)} checks")
                
            if hasattr(reward_info, 'communicate_checks') and reward_info.communicate_checks:
                print(f"\n  Communicate checks: {len(reward_info.communicate_checks)} checks")
            
            # Log termination reason
            print(f"\n  Termination: {term_reason.value}")
            # Additional info if available
            if hasattr(simulation, 'errors'):
                print(f"  Errors: {simulation.errors}")
            if hasattr(simulation, 'num_steps'):
                print(f"  Steps: {simulation.num_steps}")
            
            # Log info if present
            if hasattr(reward_info, 'info') and reward_info.info:
                print(f"\n  Additional info: {reward_info.info}")
            
            print(f"{'='*80}\n")
            
            return reward_info.reward
        except Exception as e:
            import traceback
            print(f"ERROR during evaluation: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return 0.0
    
    # Create rubric with the exact evaluation function
    return vf.Rubric(
        funcs=[evaluate_tau2_task],
        weights=[1.0]
    )


def load_environment(
    dataset_name: str = "tau2-bench",
    dataset_config: str = "retail",
    dataset_split: str = "train", 
    subset_size: Optional[int] = None,
    seed: int = 42,
    domain: str = "retail",
    use_cache: bool = True,
    solo_mode: bool = False,
    **kwargs
) -> vf.MultiTurnEnv:
    """Load tau2-bench environment using tau2's native functions."""
    if not TAU2_AVAILABLE:
        raise ImportError("tau2-bench is not installed. Please install it first.")
    
    # Ensure data is set up
    setup_tau2_data()
    
    # Use domain from dataset_config if not explicitly provided
    if dataset_config and domain == "retail":
        domain = dataset_config
    
    # Create dataset using tau2's native functions
    full_dataset = create_tau2_dataset(domain)
    
    # Get tasks using tau2's native functions
    if domain == "retail":
        tau2_tasks = get_retail_tasks()
    elif domain == "airline":
        tau2_tasks = get_airline_tasks()
    elif domain == "telecom":
        tau2_tasks = get_telecom_tasks()
    else:
        raise ValueError(f"Unknown domain: {domain}")
    
    # Handle subset if requested
    if subset_size is not None and subset_size < len(full_dataset):
        indices = list(range(len(full_dataset)))
        import random
        random.seed(seed)
        random.shuffle(indices)
        full_dataset = full_dataset.select(indices[:subset_size])
    
    # Create rubric using tau2's evaluation
    rubric = create_tau2_rubric(domain)
    
    # Create environment instance
    env = Tau2BenchEnv(
        dataset=full_dataset,
        rubric=rubric,
        domain=domain,
        tau2_tasks=tau2_tasks,
        user_llm=kwargs.get("user_llm", "gpt-4.1-mini"),
        max_steps=kwargs.get("max_steps", 200),
        max_errors=kwargs.get("max_errors", 10),
        solo_mode=solo_mode,
        **kwargs
    )
    
    return env