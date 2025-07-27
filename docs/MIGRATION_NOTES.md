# Documentation Migration Notes

## Type System Updates

### Pydantic Models Migration

The verifiers codebase has migrated from using TypedDicts to Pydantic models for certain structured data types:

1. **GenerateInputs** - Now a Pydantic BaseModel with validated fields:
   - `prompt: List[Messages]`
   - `answer: Optional[List[str]]`
   - `info: Optional[List[Dict]]`
   - `task: Optional[List[str]]`
   - `completion: Optional[List[Messages]]`

2. **ProcessedOutputs** - Now a Pydantic BaseModel for processed training data:
   - `prompt_ids: List[List[int]]`
   - `prompt_mask: List[List[int]]`
   - `completion_ids: List[List[int]]`
   - `completion_mask: List[List[int]]`
   - `completion_logprobs: List[List[float]]`
   - `rewards: List[float]`

### MultiTurnEnv Corrections

The documentation has been updated to reflect correct usage patterns:

1. **Message Types**: `env_response` should return a list of ChatMessage dictionaries, not strings:
   ```python
   # Correct
   return [{"role": "user", "content": "feedback"}], state
   
   # Incorrect (was shown in docs)
   return "feedback", state
   ```

2. **State Handling**: Always modify the existing state dictionary, never create a new one:
   ```python
   # Correct
   state["turn"] = state.get("turn", 0) + 1
   return response, state
   
   # Incorrect (was shown in docs)
   new_state = {"turn": 1, ...}
   return response, new_state
   ```

3. **Return Type**: `env_response` returns `Tuple[Messages, State]` where Messages is typically `List[ChatMessage]`

### Import Updates

ChatMessage and related types are imported from OpenAI's types:
```python
from openai.types.chat.chat_completion_message_param import (
    ChatCompletionMessageParam as ChatMessage,
)
```

## Documentation Files Updated

1. **api_reference.md**:
   - Added Pydantic models section
   - Fixed env_response return type documentation
   - Added proper import statements for types
   - Clarified state modification patterns

2. **environments.md**:
   - Fixed MultiTurnEnv example to use ChatMessage lists
   - Corrected state handling in examples

3. **components.md**:
   - Updated WordleEnv example with correct message types
   - Fixed state modification patterns

4. **overview.md**:
   - Clarified env_response return types in example

## Common Patterns

### Correct env_response Implementation
```python
def env_response(self, messages: Messages, state: State) -> Tuple[Messages, State]:
    # Get last assistant message
    last_msg = messages[-1]
    if last_msg["role"] != "assistant":
        return [], state  # No response if not assistant
    
    # Process action
    action = last_msg["content"]
    
    # Modify state in-place
    state["turn"] = state.get("turn", 0) + 1
    state["last_action"] = action
    
    # Return list of ChatMessage dicts
    response = [{"role": "user", "content": "Environment feedback"}]
    return response, state
```

### State Best Practices
- Always modify the state dictionary passed to you
- Use `state.get()` for optional fields with defaults
- Track turn counts, history, and environment-specific data
- Never create a new state dictionary from scratch in env_response

## Remaining Considerations

1. The codebase shows that `Messages` can be either a string (for completion mode) or `List[ChatMessage]` (for chat mode), but in practice, MultiTurnEnv implementations should use chat mode with proper message lists.

2. When implementing custom environments, always refer to existing environment examples in the `environments/` directory as the ground truth for correct patterns.

3. The type system is designed to be flexible but the documentation now emphasizes the most common and correct usage patterns.