# Pydantic Migration Guide

This document describes the lightweight reworking of the types system to use Pydantic instead of TypedDicts.

## Summary of Changes

1. **Converted TypedDicts to Pydantic Models**:
   - `ChatMessage` (TypedDict) → `ChatMessage` (BaseModel)
   - `GenerateInputs` (TypedDict) → `GenerateInputs` (BaseModel)
   - `ProcessedOutputs` (TypedDict) → `ProcessedOutputs` (BaseModel)

2. **Maintained Backward Compatibility**:
   - Added dict-like access methods (`__getitem__`, `__setitem__`, `get`) to `ChatMessage`
   - All models allow extra fields with `model_config = {"extra": "allow"}`
   - Added `from_dict()` class method for easy conversion

3. **Created Utility Functions** (`verifiers/types_utils.py`):
   - `ensure_chat_message()` - Convert dict or ChatMessage to ChatMessage instance
   - `ensure_chat_messages()` - Convert list of dicts/ChatMessages to ChatMessages
   - `create_user_message()` - Convenience function for user messages
   - `create_assistant_message()` - Convenience function for assistant messages
   - `create_system_message()` - Convenience function for system messages
   - `create_tool_message()` - Convenience function for tool messages

## Benefits of Pydantic Models

1. **Type Validation**: Automatic validation of data types at runtime
2. **Serialization**: Built-in JSON serialization with `model_dump()` and `model_dump_json()`
3. **IDE Support**: Better autocomplete and type hints
4. **Data Parsing**: Automatic type coercion and parsing
5. **Schema Generation**: Can generate JSON schemas for API documentation

## Migration Examples

### Before (TypedDict):
```python
# Creating a message
msg = {"role": "user", "content": "Hello"}

# Type annotation
def process(msg: ChatMessage):
    role = msg["role"]
```

### After (Pydantic - Option 1: Direct):
```python
from verifiers.types import ChatMessage

# Creating a message
msg = ChatMessage(role="user", content="Hello")

# Dict-like access still works
role = msg["role"]  # Backward compatible
role = msg.role     # Pydantic style
```

### After (Pydantic - Option 2: Using utilities):
```python
from verifiers.types_utils import create_user_message

# Using convenience functions
msg = create_user_message("Hello")
```

### For Gradual Migration:
```python
from verifiers.types_utils import ensure_chat_message

# Works with both dict and ChatMessage
def process(msg):
    msg = ensure_chat_message(msg)  # Converts if needed
    # Now msg is guaranteed to be a ChatMessage instance
```

## Code Changes Made

1. **verifiers/types.py**: Replaced TypedDict imports with Pydantic BaseModel
2. **verifiers/types_utils.py**: Added utility functions for migration
3. **verifiers/envs/multiturn_env.py**: Updated to use Pydantic model instantiation

## Next Steps for Full Migration

1. **Gradual Updates**: Update code that creates messages as dicts to use `ChatMessage()` directly
2. **Use Utilities**: Leverage functions from `verifiers.types_utils` for cleaner code
3. **Validation**: Add Pydantic validators where needed for business logic
4. **Testing**: Existing tests should continue to work due to backward compatibility

## No Functionality Changes

This migration preserves all existing functionality. The dict-like access ensures that existing code continues to work without modification.