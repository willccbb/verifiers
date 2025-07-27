# τ²-bench Implementation Details

## Overview

This is a complete implementation of τ²-bench (tau2-bench) for the verifiers framework. The implementation fully supports the dual-control environment where both agents and users can execute tools, preserving the exact logic of the original benchmark.

## Architecture

### Key Components

1. **`tau2_bench.py`** - Main environment class
   - Extends `MultiTurnEnv` from verifiers
   - Manages state across agent, user, and environment
   - Delegates complex logic to orchestrator

2. **`orchestrator.py`** - Orchestration logic
   - Handles the three-way interaction (agent ↔ user ↔ environment)
   - Manages tool execution for both agent and user
   - Tracks task completion and goal achievement

3. **`tool_adapter.py`** - Tool conversion utilities
   - Converts tau2 tools to verifiers `BaseTool` format
   - Handles both agent and user tools
   - Maintains state synchronization

## State Management

The implementation uses a unified state dictionary with namespaces:

```python
state = {
    # Agent-related state
    "agent_state": {},
    
    # User simulator state
    "user_state": {
        "instructions": {...},
        "context": {},
        "conversation_stage": "initial"
    },
    
    # Environment database state
    "env_db": {
        "agent_db": {...},  # Agent's view of database
        "user_db": {...}    # User's view (telecom only)
    },
    
    # Conversation tracking
    "turn_count": 0,
    "termination_reason": None,
    
    # Tool execution history
    "tool_executions": [
        {
            "role": "agent",
            "tool": "search_order",
            "args": {...},
            "result": {...},
            "timestamp": "2025-01-01T12:00:00"
        }
    ],
    
    # Orchestrator instance
    "orchestrator": Tau2Orchestrator(...)
}
```

## Dual-Control Implementation

### Agent Tools
- Available in all domains (retail, airline, telecom)
- Executed through `orchestrator._execute_agent_tools()`
- State synced after each execution

### User Tools
- Only available in telecom domain
- User simulator can call tools based on instructions
- Executed through `orchestrator._execute_user_tools()`
- Separate state tracking for user's environment view

## Message Flow

1. **Agent Message** → Orchestrator
   - Execute any agent tool calls
   - Generate user response
   - Execute any user tool calls (if telecom)

2. **User Message** → Orchestrator
   - Execute any user tool calls (if telecom)
   - No automatic response generation

3. **Tool Results** → State Update
   - Sync environment state
   - Track execution history
   - Check task completion

## Task Completion Logic

Tasks are completed when:
1. All goals in `expected_state` are achieved
2. User explicitly stops/transfers
3. Maximum turns reached

Goal types supported:
- `db_state`: Database matches expected state
- `tool_called`: Required tool was successfully called
- `conversation`: Specific conversation patterns present

## Domain-Specific Features

### Retail Domain
- Agent tools: order management, search, refunds
- No user tools
- Focus on customer service scenarios

### Airline Domain
- Agent tools: flight search, booking, changes
- No user tools
- Complex multi-step booking flows

### Telecom Domain
- Agent tools: technical support, account management
- User tools: check own data, run diagnostics
- Full dual-control scenarios

## Integration with Verifiers

The implementation seamlessly integrates with verifiers:
- Uses `MultiTurnEnv` for conversation management
- Compatible with standard evaluation pipelines
- Supports all verifiers features (rubrics, logging, etc.)

## Example Usage

```python
import verifiers as vf

# Load environment
env = vf.load_environment(
    "tau2_bench",
    domain="telecom",  # or "retail", "airline"
    num_eval_examples=10,
    user_llm="gpt-4"
)

# Evaluate with any model
results = env.evaluate(
    model="gpt-4",
    num_examples=10,
    verbose=True
)

# Access results
print(f"Score: {results['score']}")
print(f"Completed tasks: {results['completed']}")
```

## Testing

Run the test suite:
```bash
python environments/tau2_bench/test_tau2.py
```

This will test:
- Environment loading for all domains
- Tool extraction and conversion
- Orchestrator flow
- State management

## Future Enhancements

1. **Enhanced Goal Checking**: More sophisticated goal verification based on domain-specific logic
2. **Policy Compliance**: Add policy document parsing and compliance checking
3. **Cost Tracking**: Implement cost tracking as in original tau2
4. **Metrics**: Add pass^k reliability metric from tau2

## Troubleshooting

### tau2-bench not found
```bash
git clone https://github.com/sierra-research/tau2-bench.git
pip install -e tau2-bench/
```

### Missing API key for user simulation
```bash
export OPENAI_API_KEY=your_key_here
```

### State serialization issues
The orchestrator instance is stored in state but may not be serializable. Consider implementing custom serialization if needed for distributed evaluation.