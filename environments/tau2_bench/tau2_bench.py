"""
Main τ²-bench environment implementation for verifiers.
Supports full dual-control (both agent and user can execute tools).
"""

import json
from typing import List, Tuple, Dict, Any, Optional, Union
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import verifiers as vf
from verifiers.envs import MultiTurnEnv
from datasets import Dataset

# Import local modules
from .orchestrator import Tau2Orchestrator
from .tool_adapter import extract_tool_definitions_from_tau2, ToolExecutor

# Import tau2-bench components
try:
    from tau2.domains.retail.environment import get_environment as get_retail_env
    from tau2.domains.retail.environment import get_tasks as get_retail_tasks
    from tau2.domains.airline.environment import get_environment as get_airline_env
    from tau2.domains.airline.environment import get_tasks as get_airline_tasks
    from tau2.domains.telecom.environment import get_environment as get_telecom_env
    from tau2.domains.telecom.environment import get_tasks as get_telecom_tasks
    from tau2.data_model.message import (
        AssistantMessage, UserMessage, ToolMessage, 
        MultiToolMessage, ToolCall, Message as Tau2Message
    )
    from tau2.user.user_simulator import UserSimulator
    from tau2.user.dummy_user import DummyUser
    TAU2_AVAILABLE = True
except ImportError:
    TAU2_AVAILABLE = False
    print("Warning: tau2-bench not installed. Please install it to use this environment.")


class Tau2BenchEnv(MultiTurnEnv):
    """
    τ²-bench environment supporting dual-control scenarios.
    Both agent and user can execute tools.
    """
    
    def __init__(self,
                 dataset: Dataset,
                 rubric: vf.Rubric,
                 domain: str,
                 tau2_env,
                 tau2_tasks: List[Dict],
                 user_llm: str = "gpt-4",
                 max_turns: int = 30,
                 **kwargs):
        super().__init__(dataset, rubric, **kwargs)
        self.domain = domain
        self.tau2_env = tau2_env
        self.tau2_tasks = tau2_tasks
        self.user_llm = user_llm
        self.max_turns = max_turns
        
        # Create task lookup
        self.task_lookup = {task['id']: task for task in tau2_tasks}
        
        # Extract tool definitions
        self.agent_tools, self.user_tools = extract_tool_definitions_from_tau2(tau2_env)
        
    def is_completed(self, messages: vf.Messages, state: vf.State, **kwargs) -> bool:
        """Check if conversation is completed."""
        # Check max turns
        turn_count = state.get("turn_count", 0)
        if turn_count >= self.max_turns:
            state["termination_reason"] = "max_turns_reached"
            return True
            
        # Check if user said stop/transfer
        if messages:
            last_msg = messages[-1]
            if last_msg["role"] == "user":
                content = last_msg.get("content", "").lower()
                if "stop" in content or "transfer" in content:
                    state["termination_reason"] = "user_stop"
                    return True
                    
        # Check if task goal is achieved using orchestrator
        if "orchestrator" in state:
            orchestrator = state["orchestrator"]
            if orchestrator.check_task_completion(state):
                state["termination_reason"] = "goal_achieved"
                return True
            
        return False
        
    def env_response(self, messages: vf.Messages, state: vf.State, **kwargs) -> Tuple[vf.Messages, vf.State]:
        """
        Handle environment response including user simulation and tool execution.
        This is the core of the dual-control logic.
        """
        if not messages:
            return [], state
            
        last_msg = messages[-1]
        response_messages = []
        
        # Initialize state namespaces if needed
        if "agent_state" not in state:
            state["agent_state"] = {}
        if "user_state" not in state:
            state["user_state"] = self._init_user_state(state)
        if "env_db" not in state:
            state["env_db"] = self._init_env_db(state)
        if "turn_count" not in state:
            state["turn_count"] = 0
        if "tool_executions" not in state:
            state["tool_executions"] = []
        
        # Initialize orchestrator for this task if not already done
        if "orchestrator" not in state:
            task_id = state.get("task_id")
            task = self.task_lookup.get(task_id, {})
            state["orchestrator"] = Tau2Orchestrator(
                tau2_env=self.tau2_env,
                domain=self.domain,
                task=task,
                user_llm=self.user_llm,
                max_turns=self.max_turns
            )
            
        # Use orchestrator to handle the message
        orchestrator = state["orchestrator"]
        
        if last_msg["role"] == "assistant":
            # Process agent message through orchestrator
            new_messages, state = orchestrator.process_agent_message(messages, state)
            response_messages.extend(new_messages)
                    
        elif last_msg["role"] == "user":
            # Process user message through orchestrator (mainly for user tools in telecom)
            new_messages, state = orchestrator.process_user_message(messages, state)
            response_messages.extend(new_messages)
                
        # Update turn count
        state["turn_count"] += len([m for m in response_messages if m["role"] in ["assistant", "user"]])
        
        return response_messages, state
        
    def _init_user_state(self, state: vf.State) -> Dict[str, Any]:
        """Initialize user simulator state."""
        task_id = state.get("task_id")
        task = self.task_lookup.get(task_id, {})
        
        user_state = {
            "instructions": task.get("user_instructions", {}),
            "context": {},
            "conversation_stage": "initial"
        }
        
        return user_state
        
    def _init_env_db(self, state: vf.State) -> Dict[str, Any]:
        """Initialize environment database state."""
        task_id = state.get("task_id")
        task = self.task_lookup.get(task_id, {})
        
        # Get initial state from task
        initial_state = task.get("initial_state", {})
        if initial_state and "initialization_data" in initial_state:
            # Deep copy the initialization data
            return deepcopy(initial_state["initialization_data"])
        
        # Default empty DB
        return {}


def create_tau2_dataset(tau2_tasks: List[Dict], domain: str) -> Dataset:
    """Convert tau2 tasks to verifiers dataset format."""
    dataset_rows = []
    
    for task in tau2_tasks:
        # Extract key information
        user_instructions = task.get("user_instructions", {})
        scenario = user_instructions.get("scenario", "")
        
        # Get initial message history if available
        initial_messages = []
        if task.get("initial_state", {}).get("message_history"):
            initial_messages = [
                {
                    "role": msg["role"],
                    "content": msg.get("content", "")
                }
                for msg in task["initial_state"]["message_history"]
            ]
        
        # Default system prompt based on domain
        if not initial_messages:
            if domain == "retail":
                system_content = "You are a customer service agent for an online retail company."
            elif domain == "airline":
                system_content = "You are a customer service agent for an airline."
            elif domain == "telecom":
                system_content = "You are a technical support agent for a telecom company."
            else:
                system_content = "You are a helpful customer service agent."
                
            initial_messages = [{"role": "system", "content": system_content}]
        
        row = {
            "prompt": initial_messages,
            "question": scenario,
            "info": {
                "task_id": task["id"],
                "domain": domain,
                "expected_state": task.get("expected_state", {}),
                "initial_state": task.get("initial_state", {}),
                "user_instructions": user_instructions
            },
            "answer": "Successfully completed task",  # Placeholder
            "task": f"tau2_{domain}",
            # Store task_id in state for easy lookup
            "task_id": task["id"]
        }
        dataset_rows.append(row)
        
    return Dataset.from_list(dataset_rows)


def create_tau2_rubric(domain: str) -> vf.Rubric:
    """Create evaluation rubric for tau2 tasks."""
    
    def check_goal_achievement(completion, info, state, **kwargs) -> float:
        """Check if task goals were achieved."""
        expected_state = info.get("expected_state", {})
        goals = expected_state.get("goals", [])
        
        if not goals:
            # No specific goals, check termination reason
            termination = state.get("termination_reason", "")
            if termination == "goal_achieved":
                return 1.0
            elif termination == "user_stop":
                return 0.5  # Partial credit
            else:
                return 0.0
                
        # Check each goal
        achieved = 0
        for goal in goals:
            # This would need domain-specific logic
            # For now, check if relevant tools were called
            tool_execs = state.get("tool_executions", [])
            if any(exec["tool"] == goal.get("required_tool") for exec in tool_execs):
                achieved += 1
                
        return achieved / len(goals) if goals else 0.0
        
    def check_efficiency(completion, info, state, **kwargs) -> float:
        """Check efficiency of task completion."""
        turn_count = state.get("turn_count", 0)
        max_turns = 30
        
        if turn_count <= 10:
            return 1.0  # Very efficient
        elif turn_count <= 20:
            return 0.7  # Reasonably efficient
        elif turn_count < max_turns:
            return 0.4  # Completed but inefficient
        else:
            return 0.0  # Hit max turns
            
    def check_tool_usage(completion, info, state, **kwargs) -> float:
        """Check appropriate tool usage."""
        tool_execs = state.get("tool_executions", [])
        
        # Check for errors
        errors = [e for e in tool_execs if "Error" in e.get("result", "")]
        if errors:
            return 1.0 - (len(errors) / max(len(tool_execs), 1))
        else:
            return 1.0
            
    # Different weights for different domains
    if domain == "telecom":
        # Telecom values correct tool usage highly
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
    user_llm: str = "gpt-4",
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