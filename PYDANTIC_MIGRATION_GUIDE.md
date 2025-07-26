# Pydantic Migration Guide

This document describes the lightweight reworking of the types system to use Pydantic instead of TypedDicts where appropriate.

## Summary of Changes

1. **Converted Internal TypedDicts to Pydantic Models**:
   - `GenerateInputs` (TypedDict) → `GenerateInputs` (BaseModel)
   - `ProcessedOutputs` (TypedDict) → `ProcessedOutputs` (BaseModel)
   - **Note**: `ChatMessage` remains a TypedDict as it's a standard interface expected by downstream methods

2. **Maintained Compatibility**:
   - All Pydantic models allow extra fields with `model_config = {"extra": "allow"}`
   - Optional fields default to None for easy instantiation

## Benefits of Pydantic Models

1. **Type Validation**: Automatic validation of data types at runtime
2. **Serialization**: Built-in JSON serialization with `model_dump()` and `model_dump_json()`
3. **IDE Support**: Better autocomplete and type hints
4. **Data Parsing**: Automatic type coercion and parsing
5. **Schema Generation**: Can generate JSON schemas for API documentation

## Migration Examples

### Example 1: GenerateInputs

**Before (TypedDict):**
```python
inputs = {
    "prompt": [messages],
    "answer": ["expected answer"],
    "info": [{"key": "value"}]
}
```

**After (Pydantic):**
```python
from verifiers.types import GenerateInputs

inputs = GenerateInputs(
    prompt=[messages],
    answer=["expected answer"],
    info=[{"key": "value"}]
)

# Access data
prompts = inputs.prompt  # Direct attribute access
inputs_dict = inputs.model_dump()  # Convert to dict
```

### Example 2: ProcessedOutputs

**Before (TypedDict):**
```python
outputs = {
    "prompt_ids": [1, 2, 3],
    "prompt_mask": [1, 1, 1],
    "completion_ids": [4, 5, 6],
    "completion_mask": [1, 1, 1],
    "completion_logprobs": [0.1, 0.2, 0.3],
    "rewards": [1.0, 0.8, 0.9]
}
```

**After (Pydantic):**
```python
from verifiers.types import ProcessedOutputs

outputs = ProcessedOutputs(
    prompt_ids=[1, 2, 3],
    prompt_mask=[1, 1, 1],
    completion_ids=[4, 5, 6],
    completion_mask=[1, 1, 1],
    completion_logprobs=[0.1, 0.2, 0.3],
    rewards=[1.0, 0.8, 0.9]
)

# Validation happens automatically
# JSON serialization is built-in
json_str = outputs.model_dump_json()
```

### ChatMessage Remains Unchanged

```python
# ChatMessage stays as TypedDict for compatibility
msg: ChatMessage = {
    "role": "user",
    "content": "Hello"
}
# This remains exactly as before
```

## How to Use ProcessedOutputs

When calling methods that return `ProcessedOutputs`:

```python
# Method returns ProcessedOutputs instance
results = env.process_completions_vllm(...)

# Option 1: Direct attribute access (recommended)
prompt_ids = results.prompt_ids
rewards = results.rewards

# Option 2: Convert to dict if needed for compatibility
results_dict = results.model_dump()
prompt_ids = results_dict["prompt_ids"]

# Option 3: Pass individual fields
some_function(
    prompt_ids=results.prompt_ids,
    rewards=results.rewards
)
```

For detailed usage patterns, see `PYDANTIC_USAGE_GUIDE.md`.

## Code Changes Made

1. **verifiers/types.py**: 
   - Converted `GenerateInputs` and `ProcessedOutputs` from TypedDict to Pydantic BaseModel
   - `ChatMessage` remains TypedDict to maintain downstream compatibility
   - Added `model_config = {"extra": "allow"}` to allow additional fields

2. **verifiers/envs/environment.py**:
   - Updated `process_completions()` to return `ProcessedOutputs` instance instead of dict
   - Updated `process_completions_vllm()` to return `ProcessedOutputs` instance instead of dict

3. **verifiers/trainers/async_batch_generator.py**:
   - Updated to call `.model_dump()` on ProcessedOutputs before storing in BatchResult
   - This maintains compatibility with downstream code expecting dictionaries

## Next Steps for Full Migration

1. **Update Usage**: When creating `GenerateInputs` or `ProcessedOutputs`, use the Pydantic models directly
2. **Add Validation**: Leverage Pydantic validators where needed for business logic
3. **Gradual Adoption**: Convert other internal TypedDicts to Pydantic as appropriate
4. **Testing**: Existing code using these types may need updates to use model instantiation

## No Functionality Changes

This migration preserves all existing functionality. ChatMessage remains a TypedDict to maintain compatibility with downstream methods that expect dictionary objects.