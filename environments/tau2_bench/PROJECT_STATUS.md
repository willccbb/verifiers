# τ²-bench Implementation Status

## Overview
This is a complete implementation of τ²-bench (tau2-bench) for the verifiers framework. τ²-bench is an advanced LLM agent benchmark that tests dual-control environments where both agents and users can execute tools.

## Current State

### ✅ Completed Features

1. **Full Dual-Control Environment**
   - Both agents and users can execute tools
   - User simulator from τ²-bench integrated for realistic interactions
   - Proper turn handoff between agent → tool → agent → user

2. **All Domains Supported**
   - `retail`: Customer service with agent-only tools
   - `airline`: Flight booking with agent-only tools  
   - `telecom`: Technical support with dual-control (both agent and user tools)

3. **Exact Original Prompts & Policies**
   - System prompts match τ²-bench exactly
   - Domain-specific policies loaded from τ²-bench data files
   - Tool descriptions provided in OpenAI function calling format

4. **Complete State Management**
   - Agent state, user state, environment database state
   - Tool execution history with full tracking
   - Error counting and termination conditions

5. **Infrastructure Integration**
   - Follows verifiers `MultiTurnEnv` pattern
   - All logic encapsulated in `env_response` method
   - Automatic data setup from τ²-bench repository
   - Clean dependency management via git reference

### 🔧 Technical Implementation

1. **Tool Schema Handling**
   - Tools are JSON-serialized in dataset to avoid HuggingFace schema conflicts
   - Automatic deserialization in `environment.py`'s `a_generate` method
   - No special overrides needed in the tau2_bench environment

2. **Message Format Compatibility**
   - Handles both dict and object formats for tool calls
   - Proper conversion to τ²-bench message format (tool_calls=None vs [])
   - Supports verifiers' message format while maintaining τ²-bench compatibility

3. **Evaluation Logic**
   - Uses τ²-bench's official evaluation functions directly (no reimplementation)
   - Binary pass/fail scoring (1.0 or 0.0) matching original
   - Properly converts messages to τ²-bench format for evaluation
   - Supports all evaluation types: ACTION, DB, ENV_ASSERTION, NL_ASSERTION, COMMUNICATE

## Recent Fix

### Evaluation Scoring Issue (FIXED)
The initial implementation had 0.0 scores because:
1. We were reimplementing evaluation logic instead of using tau2-bench's evaluators
2. Our reimplementation incorrectly checked the `requestor` field when matching actions
3. The original tau2-bench does NOT check requestor in `compare_with_tool_call`

**Solution**: Now using `tau2.evaluator.evaluator.evaluate_simulation` directly with proper message conversion to ensure exact match with original evaluation logic.

### Debug Logging Added (NEW)
Added comprehensive evaluation debug output that shows:
- **Expected actions**: All required tool calls with their arguments
- **Actual tool calls**: What the agent actually called
- **Detailed comparison**: Shows exactly why evaluations fail
- **Evaluation results**: Final scores and breakdowns by evaluation type

This helps identify why agents score 0.0 - typically they're calling different tools than expected.

## Project Goals

### Primary Objective
Create a faithful port of τ²-bench that:
- Preserves the exact evaluation logic ✅
- Maintains compatibility with the verifiers framework ✅
- Enables researchers to test LLM agents on τ²-bench tasks ✅

### Key Requirements
1. **Verbatim Logic Translation**: The evaluation uses τ²-bench functions directly ✅
2. **No Framework Modifications**: Work within verifiers' existing patterns ✅
3. **Full Feature Support**: Including dual-control environments ✅

### Success Criteria
- Agents can complete τ²-bench tasks with appropriate scores ✅
- The implementation is maintainable and well-documented ✅
- Researchers can easily run τ²-bench evaluations on their models ✅

## Usage

```bash
# Install
vf-install tau2_bench

# Evaluate
vf-eval tau2_bench --model gpt-4.1-mini --num-examples 20 --rollouts-per-example 3 --env-args '{"domain": "retail"}'
```

## Architecture Notes

The implementation follows a clean separation of concerns:
- `tau2_bench.py`: Main environment implementation
- Dataset creation with proper tool schemas
- State management within the standard verifiers flow
- Direct use of tau2-bench evaluation functions (no custom evaluation logic)

All τ²-bench specific logic is contained within the environment, making it easy to maintain and update as needed.