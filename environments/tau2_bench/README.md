# τ²-bench Implementation Status

## Overview
This is a complete implementation of τ²-bench (tau2-bench) for the verifiers framework. τ²-bench is an advanced LLM agent benchmark that tests dual-control environments where both agents and users can execute tools.

## Current State

### ✅ Completed Features

1.  **Full Dual-Control Environment**
    - Both agents and users can execute tools
    - User simulator from τ²-bench integrated for realistic interactions
    - Proper turn handoff between agent → tool → agent → user

2.  **All Domains Supported**
    - `retail`: Customer service with agent-only tools
    - `airline`: Flight booking with agent-only tools
    - `telecom`: Technical support with dual-control (both agent and user tools)

3.  **Exact Original Prompts & Policies**
    - System prompts match τ²-bench exactly
    - Domain-specific policies loaded from τ²-bench data files
    - Tool descriptions provided in OpenAI function calling format

4.  **Complete State Management**
    - Agent state, user state, environment database state
    - Tool execution history with full tracking
    - Error counting and termination conditions

5.  **Infrastructure Integration**
    - Follows verifiers `MultiTurnEnv` pattern
    - All logic encapsulated in `env_response` method
    - Automatic data setup from τ²-bench repository
    - Clean dependency management via git reference

### 🔧 Technical Implementation

1.  **Tool Schema Handling**
    - Tools are JSON-serialized in dataset to avoid HuggingFace schema conflicts
    - Automatic deserialization in `environment.py`'s `a_generate` method
    - No special overrides needed in the tau2_bench environment

2.  **Message Format Compatibility**
    - Handles both dict and object formats for tool calls
    - Proper conversion to τ²-bench message format (tool_calls=None vs [])
    - Supports verifiers' message format while maintaining τ²-bench compatibility

3.  **Evaluation Logic**
    - Binary pass/fail scoring (1.0 or 0.0) matching original
    - Checks for exact action sequences with tool name, requestor, and arguments
    - Simplified environment state checking (full hash comparison not implemented)

### 🐛 Recent Fixes

1.  **Tool Response Format**: Fixed mismatch between JSON and τ²-bench's expected format by using `tau2_env.to_json_str()` method
2.  **Error Message Format**: Tool errors now properly formatted as "Error executing {tool_name}: {error}" to match τ²-bench expectations
3.  **Evaluation AttributeErrors**: Fixed accessing τ²-bench ToolCall attributes (direct `name`/`arguments` instead of `function.name`/`function.arguments`)

## Known Issues

1.  **Evaluation Scores**: Currently showing 0.0 scores even when tasks appear to complete successfully. This suggests either:
    - The evaluation criteria are extremely strict (likely)
    - There's a mismatch in how we track/report tool executions
    - The expected actions in the dataset don't match what agents naturally do

2.  **Agent Behavior**: Generic LLMs don't always follow τ²-bench's expected patterns:
    - May not use the exact tool names expected
    - May not follow the exact sequence of actions
    - May format tool calls differently than expected

## Project Goals

### Primary Objective
Create a faithful port of τ²-bench that:
- Preserves the exact evaluation logic
- Maintains compatibility with the verifiers framework
- Enables researchers to test LLM agents on τ²-bench tasks

### Key Requirements
1.  **Verbatim Logic Translation**: The evaluation must match τ²-bench exactly
2.  **No Framework Modifications**: Work within verifiers' existing patterns
3.  **Full Feature Support**: Including dual-control environments

### Success Criteria
- Agents can complete τ²-bench tasks with appropriate scores
- The implementation is maintainable and well-documented
- Researchers can easily run τ²-bench evaluations on their models

## Usage

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