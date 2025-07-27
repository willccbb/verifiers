# τ²-bench Implementation for Verifiers

This is a complete implementation of τ²-bench (tau2-bench) for the verifiers framework, supporting the full dual-control environment where both agents and users can execute tools.

## Features

- ✅ Full dual-control support (agent and user tools)
- ✅ Complete state management (agent, user, environment)
- ✅ All logic handled within `env_response`
- ✅ All domains supported (retail, airline, telecom)
- ✅ Exact task logic replication from original benchmark
- ✅ User tool execution tracking

## Installation

Install this environment using:
```bash
vf-install tau2_bench
```

The τ²-bench dependency will be automatically installed from the GitHub repository.

## Usage

```python
import verifiers as vf

# Load environment
env = vf.load_environment("tau2_bench", domain="retail")

# Evaluate with an agent
results = env.evaluate(
    model="gpt-4.1-mini",
    num_examples=10
)
```

### Command-line evaluation
```bash
uv run vf-eval tau2_bench --model gpt-4.1-mini --num-examples 20 --rollouts-per-example 3 --env-args '{"domain": "retail"}'
```

## Domains

- **retail**: Customer service for online retail (agent tools only)
- **airline**: Flight booking and changes (agent tools only)  
- **telecom**: Technical support with dual-control (both agent and user tools)

## Implementation Details

### Architecture
The implementation uses `MultiTurnEnv` with all orchestration logic contained within `env_response`:
1. **Agent Tools**: Executed when assistant messages contain tool calls
2. **User Simulation**: Generated after agent actions using τ²-bench's UserSimulator
3. **State Management**: Single unified state dict tracks all actors

### Message Format Compatibility
The key challenge in porting was τ²-bench's strict message format requirements:
- τ²-bench expects `tool_calls=None` (not empty list) when no tools are called
- The user simulator's `flip_roles()` method validates this strictly
- Solution: Properly handle None vs empty list conversion in `_convert_to_tau2_messages()`

### State Components
- `agent_state`: Agent's internal state
- `user_state`: User simulator state and instructions  
- `tau2_user_state`: τ²-bench UserState object
- `env_db`: Environment database state (agent_db, user_db)
- `tool_executions`: Complete history of all tool calls
- `user_simulator`: The tau2 UserSimulator instance
- `tau2_env`: The tau2 environment instance
- `task_id`: Current task identifier
- `turn_count`: Number of conversation turns
- `error_count`: Tool execution errors

### Evaluation Logic (EXACT from τ²-bench)

The evaluation uses **binary pass/fail scoring** (1.0 or 0.0) with NO partial credit, using the official τ²-bench evaluation logic:

1. **Automatic Failure Conditions**:
   - Task terminated due to too many errors
   - Task reached maximum turns without completion

2. **Evaluation Types**:
   - **ACTION**: Checks if all required actions were performed with correct tool names and arguments
   - **DB**: Verifies final environment database state matches expected state
   - **ENV_ASSERTION**: Environment-specific assertions
   - **NL_ASSERTION**: Natural language assertions about conversation content
   - **COMMUNICATE**: Communication requirements between agent and user

3. **Official τ²-bench Evaluator**:
   - The implementation uses `tau2.evaluator.evaluator.evaluate_simulation` directly
   - Messages are properly converted to τ²-bench format before evaluation
   - Ensures exact match with original benchmark evaluation logic

### Current Limitations

1. **Tool Discovery**: The current implementation does not automatically provide tool descriptions to the agent in the system prompt. The agent must already know what tools are available and how to call them.

2. **Agent Compatibility**: Standard LLM agents may not naturally produce the exact tool calls τ²-bench expects without specific prompting or fine-tuning.

### Key Differences from Original
- Uses Verifiers' `MultiTurnEnv` pattern instead of Orchestrator
- All non-agent logic encapsulated in `env_response`
- Leverages official τ²-bench evaluation functions directly