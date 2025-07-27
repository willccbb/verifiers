# τ²-bench Implementation Guide

## Overview

This implementation brings τ²-bench's dual-control environment to verifiers, where both agents and users can execute tools. All logic is contained within `env_response` as per verifiers design patterns.

## Key Design Decisions

### 1. Everything in `env_response`
- Tool execution (agent and user)
- User simulation
- State management
- Goal checking

### 2. State Management
```python
state = {
    "agent_state": {},           # Agent's working memory
    "user_state": {              # User simulator state
        "instructions": {...},
        "context": {},
        "conversation_stage": "initial"
    },
    "env_db": {                  # Environment databases
        "agent_db": {...},       # Agent's view
        "user_db": {...}         # User's view (telecom only)
    },
    "tool_executions": [...],    # All tool calls history
    "user_simulator": UserSimulator(...),  # Tau2 user sim instance
    "turn_count": 0,
    "termination_reason": None
}
```

### 3. Message Flow in `env_response`

```python
def env_response(messages, state):
    # 1. Initialize state if needed
    self._init_state(state)
    
    # 2. Handle assistant messages
    if last_msg["role"] == "assistant":
        # Execute agent tools if any
        if "tool_calls" in last_msg:
            tool_results = self._execute_agent_tools(...)
            response_messages.extend(tool_results)
        
        # Generate user response
        user_msg = self._generate_user_response(...)
        if user_msg:
            response_messages.append(user_msg)
            
            # Execute user tools if any (telecom)
            if "tool_calls" in user_msg:
                user_tool_results = self._execute_user_tools(...)
                response_messages.extend(user_tool_results)
    
    return response_messages, state
```

### 4. Tool Execution

Agent tools:
```python
# Direct execution through tau2 environment
tool_func = getattr(self.tau2_env.tools, tool_name)
result = tool_func(**tool_args)
# Sync state
if hasattr(self.tau2_env.tools, 'db'):
    state["env_db"]["agent_db"] = self.tau2_env.tools.db.model_dump()
```

User tools (telecom only):
```python
# Similar pattern for user tools
tool_func = getattr(self.tau2_env.user_tools, tool_name)
result = tool_func(**tool_args)
state["env_db"]["user_db"] = self.tau2_env.user_tools.db.model_dump()
```

### 5. User Simulation

The tau2 UserSimulator is initialized once and stored in state:
```python
user_sim = UserSimulator(
    tools=user_tools,  # None for retail/airline, tools for telecom
    instructions=user_instructions,
    llm=self.user_llm
)
state["user_simulator"] = user_sim
```

## Domain Differences

### Retail & Airline
- Agent tools only
- User can describe problems but not execute tools
- Focus on customer service scenarios

### Telecom  
- **Dual-control**: Both agent and user have tools
- User can check their own data, run diagnostics
- More complex interaction patterns

## Integration Points

1. **Dataset Creation**: Converts tau2 tasks to verifiers format
2. **Rubric**: Domain-specific evaluation weights
3. **Goal Checking**: Supports db_state, tool_called, conversation goals
4. **Message Conversion**: Seamless tau2 ↔ verifiers format conversion

## Usage

```python
import verifiers as vf

# Load with any domain
env = vf.load_environment(
    "tau2_bench",
    domain="telecom",  # or "retail", "airline"
    num_eval_examples=10,
    user_llm="gpt-4"
)

# Standard evaluation
results = env.evaluate(model="gpt-4", num_examples=10)
```

## Key Achievements

✓ **No shortcuts**: Full tau2 logic preserved  
✓ **Clean design**: All logic in env_response  
✓ **State tracking**: Complete history of all interactions  
✓ **Dual-control**: User tools fully supported  
✓ **No core changes**: Works with standard verifiers