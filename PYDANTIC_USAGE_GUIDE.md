# Pydantic ProcessedOutputs Usage Guide

This guide explains how to use the `ProcessedOutputs` Pydantic model in code that previously expected dictionaries.

## The ProcessedOutputs Model

```python
class ProcessedOutputs(BaseModel):
    prompt_ids: List[int]
    prompt_mask: List[int]
    completion_ids: List[int]
    completion_mask: List[int]
    completion_logprobs: List[float]
    rewards: List[float]
```

## Usage Patterns

### 1. Accessing Fields (Attribute Access)

**Recommended approach** - Use attribute access:

```python
# When you receive a ProcessedOutputs instance
processed_results = env.process_completions_vllm(...)

# Access fields directly
prompt_ids = processed_results.prompt_ids
rewards = processed_results.rewards
```

### 2. Dictionary-Style Access (When Needed)

If you need dict-style access for compatibility:

```python
# Option A: Convert to dict when needed
processed_dict = processed_results.model_dump()
prompt_ids = processed_dict["prompt_ids"]

# Option B: Access specific fields
prompt_ids = processed_results.prompt_ids  # Preferred
```

### 3. Passing to Functions Expecting Dicts

When passing to functions that expect dictionaries:

```python
# Convert the Pydantic model to a dict
batch_result = BatchResult(
    batch_id=request.batch_id,
    processed_results=processed_results.model_dump(),  # Convert to dict
    # ... other fields
)
```

### 4. Type Hints

Update type hints to use the Pydantic model:

```python
from verifiers.types import ProcessedOutputs

def process_data(results: ProcessedOutputs) -> None:
    # Direct attribute access
    for reward in results.rewards:
        print(reward)
```

## Migration Examples

### Example 1: GRPO Trainer Usage

**Current code (expecting dict):**
```python
processed_results = batch_result.processed_results
broadcast_data = {
    "prompt_ids": processed_results["prompt_ids"],
    "prompt_mask": processed_results["prompt_mask"],
    # ...
}
```

**Updated code (with Pydantic):**
```python
processed_results = batch_result.processed_results
# If processed_results is a ProcessedOutputs instance:
broadcast_data = {
    "prompt_ids": processed_results.prompt_ids,
    "prompt_mask": processed_results.prompt_mask,
    # ...
}

# OR, if you need to keep dict interface:
processed_dict = processed_results.model_dump()
broadcast_data = {
    "prompt_ids": processed_dict["prompt_ids"],
    "prompt_mask": processed_dict["prompt_mask"],
    # ...
}
```

### Example 2: Async Batch Generator

**Update needed in `async_batch_generator.py`:**
```python
# Change the BatchResult dataclass
@dataclass
class BatchResult:
    batch_id: int
    processed_results: Dict[str, Any]  # Keep as dict for compatibility
    # ...

# In the generation method, convert to dict:
processed_results = self.env.process_env_results_vllm(...)
return BatchResult(
    batch_id=request.batch_id,
    processed_results=processed_results.model_dump(),  # Convert to dict
    # ...
)
```

## Best Practices

1. **Use attribute access** when working directly with ProcessedOutputs
2. **Convert to dict** only when interfacing with code expecting dictionaries
3. **Update type hints** to use `ProcessedOutputs` instead of `Dict[str, Any]`
4. **Validate data** - Pydantic will raise errors if data doesn't match expected types

## Gradual Migration Strategy

1. **Phase 1**: Update return points to use `.model_dump()` for compatibility
2. **Phase 2**: Update consumers to work with Pydantic models directly
3. **Phase 3**: Remove `.model_dump()` calls once all consumers are updated