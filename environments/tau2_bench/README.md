# τ²-bench Implementation for Verifiers

This is a complete implementation of τ²-bench (tau2-bench) for the verifiers framework, supporting the full dual-control environment where both agents and users can execute tools.

## Features

- ✅ Full dual-control support (agent and user tools)
- ✅ Complete state management (agent, user, environment)
- ✅ Orchestrator logic preserved
- ✅ All domains supported (retail, airline, telecom)
- ✅ Exact task logic replication
- ✅ User tool execution tracking

## Architecture

The implementation uses `MultiTurnEnv` to manage the complex interactions between:
1. **Agent**: Uses agent tools to help the user
2. **User**: Can execute their own tools (in telecom domain)
3. **Environment**: Maintains database state and validates actions

State is tracked in a single dict with namespaces:
- `agent_state`: Agent's internal state
- `user_state`: User simulator state
- `env_db`: Environment database state
- `message_history`: Full conversation history
- `tool_executions`: Track of all tool calls

## Installation

1. Install τ²-bench:
```bash
pip install -e /path/to/tau2-bench
```

2. Set up environment variables:
```bash
export TAU2_DATA_DIR=/path/to/tau2-bench/data
```

3. Install this environment:
```bash
vf-install tau2_bench -p environments/
```

## Usage

```python
import verifiers as vf

# Load environment
env = vf.load_environment("tau2_bench", domain="retail")

# Evaluate with an agent
results = env.evaluate(
    model="gpt-4",
    num_examples=10
)
```

## Domains

- **retail**: Customer service for online retail (agent tools only)
- **airline**: Flight booking and changes (agent tools only)  
- **telecom**: Technical support with dual-control (both agent and user tools)

## Implementation Details

See individual module documentation for details on:
- State management strategy
- Tool execution handling
- User simulation with tools
- Orchestration logic