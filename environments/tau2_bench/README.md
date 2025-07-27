# τ²-bench Environment for Verifiers

**⚠️ IN-PROGRESS PORT - PARITY NOT YET ACHIEVED ⚠️**

This is an in-progress implementation of the τ²-bench (tau2-bench) dual-control agent benchmark for the Verifiers framework. While the core functionality is implemented, full parity with the original benchmark has not yet been achieved.

## Current Status

### What's Working
- ✅ Full dual-control environment implementation (both agent and user can execute tools)
- ✅ All three domains supported (retail, airline, telecom)
- ✅ Native τ²-bench integration using original functions
- ✅ Proper error handling and formatting
- ✅ Per-rollout state isolation
- ✅ Tool execution tracking

### Known Issues
- ❌ **Lower than expected scores**: GPT-4.1-mini achieves ~13-17% on retail (expected 40%+)
- ❌ **CommunicateEvaluator failures**: Agents don't communicate required information strings
- ❌ **Tool message pairing errors**: Some rollouts fail with "Tool message expected" errors
- ❌ **State synchronization issues**: Occasional mismatches between expected and actual states

### Technical Details

The implementation:
- Uses τ²-bench's native functions directly via `tau2` package
- Implements exact evaluation logic including all four evaluators
- Follows Verifiers' `MultiTurnEnv` pattern
- Encapsulates all logic in `env_response` method
- Handles complex message format conversions

Key challenges addressed:
- Tool error formatting as `Error executing {tool_name}: {message}`
- JSON serialization of tool schemas to avoid HuggingFace conflicts
- Proper handling of `ChatCompletionMessageToolCall` objects
- Turn handoff between agent → tool → agent → user

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