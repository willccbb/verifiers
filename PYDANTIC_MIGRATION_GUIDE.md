# Pydantic Migration Guide

This document describes the reworking of the types system to use Pydantic instead of TypedDicts where appropriate.

## Summary of Changes

1. **Converted Internal Types to Pydantic Models**:
   - `GenerateInputs` → Pydantic BaseModel
   - `ProcessedOutputs` → Pydantic BaseModel
   - `BatchRequest` → Pydantic BaseModel
   - `BatchResult` → Pydantic BaseModel
   - **Note**: `ChatMessage` remains a TypedDict as it's a standard interface expected by downstream methods

## Usage

### Creating Instances

```python
from verifiers.types import GenerateInputs, ProcessedOutputs

# Create GenerateInputs
inputs = GenerateInputs(
    prompt=[messages],
    answer=["expected answer"],
    info=[{"key": "value"}]
)

# Create ProcessedOutputs (typically returned by methods)
outputs = ProcessedOutputs(
    prompt_ids=[1, 2, 3],
    prompt_mask=[1, 1, 1],
    completion_ids=[4, 5, 6],
    completion_mask=[1, 1, 1],
    completion_logprobs=[0.1, 0.2, 0.3],
    rewards=[1.0, 0.8, 0.9]
)
```

### Accessing Fields

Use direct attribute access:

```python
# When you receive a ProcessedOutputs instance
results = env.process_completions_vllm(...)

# Access fields directly
prompt_ids = results.prompt_ids
rewards = results.rewards

# Same for BatchResult
batch_result = async_generator.get_batch(batch_id)
processed_results = batch_result.processed_results
prompt_ids = processed_results.prompt_ids
```

### Benefits

1. **Type Validation**: Automatic validation of data types at runtime
2. **IDE Support**: Better autocomplete and type hints
3. **Serialization**: Built-in JSON serialization with `model_dump()` if needed
4. **Cleaner Code**: No more TypedDict imports or type annotations

## Code Changes Made

1. **verifiers/types.py**: 
   - Converted `GenerateInputs` and `ProcessedOutputs` to Pydantic BaseModel
   - `ChatMessage` remains TypedDict for downstream compatibility

2. **verifiers/envs/environment.py**:
   - `process_completions()` returns `ProcessedOutputs` instance
   - `process_completions_vllm()` returns `ProcessedOutputs` instance

3. **verifiers/trainers/async_batch_generator.py**:
   - Converted `BatchRequest` and `BatchResult` to Pydantic BaseModel
   - Updated `BatchResult.processed_results` type to `ProcessedOutputs`

4. **verifiers/trainers/grpo_trainer.py**:
   - Updated to use attribute access instead of dict-style access

## No Functionality Changes

This migration preserves all existing functionality while providing better type safety and developer experience.