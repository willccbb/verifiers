# τ²-bench Environment for Verifiers

Implementation of the τ²-bench (tau2-bench) dual-control agent benchmark for the Verifiers framework.

## Current Project Status

### What We've Achieved
1. **Full Dual-Control Implementation**: Successfully implemented τ²-bench's unique dual-control environment where both agent and user can execute tools
2. **Native τ² Integration**: Refactored to use τ²-bench's native functions directly rather than reimplementing:
   - Using `tau2` tool execution directly (tools raise `ValueError` exceptions for errors)
   - Using `tau2`'s `get_tasks()` functions for each domain
   - Using `tau2`'s exact prompts from `AGENT_INSTRUCTION` and `SYSTEM_PROMPT`
   - Using `tau2`'s `evaluate_simulation()` for scoring
3. **Strict Evaluation Matching**: Implemented exact matching of τ²-bench's evaluation logic including all four evaluators (Action, Environment, Communicate, NLAssertions)
4. **Error Format Compliance**: Tool errors are formatted as `Error executing {tool_name}: {message}` to match τ²-bench's expectations
5. **Per-Rollout State Isolation**: Each evaluation rollout gets a fresh `tau2.environment.Environment` instance to prevent state leakage

### Current Performance
- **GPT-4.1-mini on retail domain**: ~13-17% (vs expected 40%+)
- **Primary issue**: `CommunicateEvaluator` failures - the agent performs correct actions but doesn't communicate required information strings

### Known Issues
1. **Low scores due to CommunicateEvaluator**: τ²-bench checks if specific strings (from `communicate_info`) appear in agent messages. Even when agents perform all correct actions and achieve the correct DB state, they get 0.0 if they don't say the exact required phrases.
2. **Occasional state synchronization issues**: Sometimes tools succeed when τ²-bench expects them to fail, suggesting potential timing or state management edge cases

### Next Steps
1. **Investigate communication requirements**: Deep dive into what specific phrases τ²-bench expects agents to communicate
2. **Consider prompt engineering**: May need to add guidance for agents to communicate specific information
3. **Debug remaining state issues**: Ensure perfect alignment between our environment state and τ²-bench's expectations

## Installation

```bash
# Install
vf-install tau2_bench

# Evaluate
vf-eval tau2_bench --model gpt-4.1-mini --num-examples 20 --rollouts-per-example 3 --env-args '{"domain": "retail"}'
```

## Next Steps

1.  **Debug Evaluation Scoring**: Investigate why scores are consistently 0.0
2.  **Verify Action Matching**: Ensure our tool execution tracking matches τ²-bench's expectations
3.  **Test with Multiple Models**: See if different models achieve better scores
4.  **Add Logging**: More detailed logging of evaluation criteria checks

## Architecture Notes

The implementation follows a clean separation of concerns:
- `tau2_bench.py`: Main environment implementation
- Dataset creation with proper tool schemas
- State management within the standard verifiers flow
- No custom dataset overrides or special handling

All τ²-bench specific logic is contained within the environment, making it easy to maintain and update as needed.