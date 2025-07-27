# τ²-bench Implementation Differences Analysis

## Overview
This document analyzes the key differences between our Verifiers implementation of τ²-bench and the original ground truth implementation.

## Architecture Differences

### 1. Orchestration Pattern
**Original τ²-bench:**
- Uses a centralized `Orchestrator` class that manages the simulation loop
- Explicit role transitions: AGENT → ENV → USER → AGENT
- Step-by-step execution with `step()` method
- Tracks `from_role` and `to_role` for message routing

**Our Implementation:**
- Uses `MultiTurnEnv.env_response()` to handle all non-agent logic
- Implicit role handling within a single method
- No explicit step tracking - everything happens in `env_response()`

### 2. Message Format
**Original τ²-bench:**
- Custom message classes: `AssistantMessage`, `UserMessage`, `ToolMessage`, `MultiToolMessage`
- Rich message objects with validation, timestamps, and metadata
- Separate tool call structure

**Our Implementation:**
- Standard Verifiers message format (dict-based)
- OpenAI-compatible message structure
- Tool calls embedded in message dicts

## Logic Differences

### 1. Tool Execution
**Original τ²-bench:**
```python
# Original: Direct environment method call
tool_msg = self.environment.get_response(tool_call)
```

**Our Implementation:**
```python
# Ours: Manual tool lookup and execution
if hasattr(self.tau2_env.tools, tool_name):
    tool_func = getattr(self.tau2_env.tools, tool_name)
    result = tool_func(**tool_args)
```

**Key Differences:**
- Original uses environment's built-in response handling
- Ours manually manages tool execution and state synchronization
- We track all executions in `state["tool_executions"]` for evaluation

### 2. User Simulation
**Original τ²-bench:**
```python
# Original: Direct state passing
user_msg, self.user_state = self.user.generate_next_message(
    self.message, self.user_state
)
```

**Our Implementation:**
```python
# Ours: Full message history conversion
tau2_messages = self._convert_to_tau2_messages(messages)
user_msg, new_user_state = user_sim.generate_next_message(
    message_history=tau2_messages,
    user_state=state["user_state"]
)
```

**Key Differences:**
- Original passes single message, ours passes full history
- We convert between message formats
- State management happens in the verifiers state dict

### 3. Completion Checking
**Original τ²-bench:**
- Uses complex evaluation system with multiple evaluators
- Compares environment states (DB hashes)
- Runs environment assertions
- Supports multiple evaluation types (ENV, NL_ASSERTIONS, COMMUNICATE, ACTION)

**Our Implementation:**
- Simplified goal checking in `_check_task_completion()`
- Basic DB state comparison (field-by-field)
- Tool usage verification
- No environment assertions or hash comparison

### 4. State Management
**Original τ²-bench:**
- Separate `agent_state` and `user_state` objects
- Environment maintains its own internal state
- State transitions managed by orchestrator

**Our Implementation:**
- Unified state dict with nested components
- Manual state initialization in `_init_state()`
- State passed through verifiers framework

## Functional Equivalence

Despite architectural differences, our implementation maintains functional equivalence by:

1. **Preserving Dual-Control Logic**: Both agent and user can execute tools
2. **Maintaining Turn Order**: Agent → Tools → User → Tools cycle is preserved
3. **Using Original Components**: We use the actual τ²-bench user simulator and environment
4. **Tracking All Actions**: Tool executions are recorded for evaluation

## Areas Needing Alignment

### 1. Evaluation Logic
Our simplified evaluation doesn't match the original's sophistication:
- Missing: Environment hash comparison
- Missing: Environment assertions
- Missing: Action sequence validation
- Simplified: Goal achievement checking

### 2. Error Handling
Original has sophisticated error tracking with `max_errors` limit. Our implementation catches errors but doesn't limit retries.

### 3. Termination Conditions
Original tracks multiple termination reasons:
- MAX_STEPS
- AGENT_STOP
- USER_STOP
- TOO_MANY_ERRORS
- Goal achievement

Our implementation only tracks some of these.

### 4. Message Validation
Original validates messages with `.validate()` method. We don't perform explicit validation.

## Recommendations for Full Alignment

1. **Implement Hash-Based State Comparison**
   ```python
   def _check_task_completion(self, state):
       # Get environment hashes like original
       agent_db_hash = self.tau2_env.get_db_hash()
       expected_hash = self.get_expected_hash(state["task_id"])
       return agent_db_hash == expected_hash
   ```

2. **Add Environment Assertions**
   ```python
   def _run_env_assertions(self, state):
       task = self.task_lookup.get(state["task_id"])
       assertions = task.evaluation_criteria.env_assertions
       for assertion in assertions:
           success = self.tau2_env.run_env_assertion(assertion)
   ```

3. **Track Error Counts**
   ```python
   if "error_count" not in state:
       state["error_count"] = 0
   # In tool execution catch blocks:
   state["error_count"] += 1
   if state["error_count"] >= self.max_errors:
       state["termination_reason"] = "too_many_errors"
   ```

4. **Validate Messages**
   Add validation before processing messages to ensure they meet τ²-bench requirements.

## Implementation Updates

Based on this analysis, we've made several updates to better align with the original:

1. **Added Error Tracking**: 
   - Added `max_errors` parameter (default 3)
   - Track `error_count` in state
   - Terminate on too many errors

2. **Enhanced Completion Checking**:
   - Added environment hash comparison support
   - Check for required actions from evaluation criteria
   - Better termination reason handling

3. **Improved Rubric**:
   - Updated scoring to consider errors and termination reasons
   - Added efficiency penalties for errors
   - Better alignment with original evaluation weights

4. **Action Sequence Validation**:
   - Added `check_action_sequence` helper function
   - Support for action sequence goals

## Remaining Differences

1. **Environment Assertions**: Not yet implemented
2. **Golden Environment Comparison**: Simplified hash comparison
3. **Message Validation**: No explicit validation
4. **Multi-evaluator System**: Using single rubric instead of multiple evaluators

## Conclusion

With these updates, our implementation now captures both the core dual-control mechanics AND more closely aligns with the original evaluation logic. The main functional behavior is preserved, and the evaluation criteria now better match the original's approach, though some advanced features (environment assertions, golden environment comparison) remain simplified.