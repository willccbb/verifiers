# GRPO Trainer Data Flow Documentation

## Overview

This document traces the complete data flow in the GRPO trainer from the beginning of training through deep async training with parallel batches. We'll track key variables, their types, shapes, and which process handles what operations.

## Key Configuration Parameters

```python
# Example configuration (from math_python.py)
per_device_train_batch_size = 8
num_generations = 8  # Each prompt generates 8 completions
gradient_accumulation_steps = 2
num_iterations = 2  # Reuse generations for 2 iterations
num_processes = 4  # 4 GPUs for training
```

## Core Concepts

### Batch Size Calculations

1. **Local batch size** = `per_device_train_batch_size = 8`
2. **Generation batch size** = `per_device_train_batch_size * gradient_accumulation_steps = 16` (per GPU)
3. **Global generation batch size** = `generation_batch_size * num_processes = 64` (across all GPUs)
4. **Total samples per generation round** = `global_generation_batch_size * num_generations = 512`

### Generation Frequency

Generations happen every `gradient_accumulation_steps * num_iterations = 4` steps.

## 1. Training Initialization

### 1.1 Dataloader Creation (`get_train_dataloader`)

```python
# On each process
batch_size = per_device_train_batch_size * gradient_accumulation_steps  # 16
dataloader = DataLoader(train_dataset, sampler=RepeatSampler(...), batch_size=16)
wrapped_dataloader = AsyncDataLoaderWrapper(dataloader, buffer_size=max(5, num_steps_async * 2))
```

### 1.2 RepeatSampler Logic

The `RepeatSampler` creates a special sampling pattern:

```
# For prompts [0, 1, 2, ...], the sampler produces:
# GPU 0: [0, 0, 1, 1, 2, 2, ...]  (each prompt repeated num_generations=8 times)
# GPU 1: [3, 3, 4, 4, 5, 5, ...]
# GPU 2: [6, 6, 7, 7, 8, 8, ...]
# GPU 3: [9, 9, 10, 10, 11, 11, ...]
```

**Key parameters:**
- `mini_repeat_count = num_generations = 8`
- `batch_size = generation_batch_size // num_generations = 2`
- `repeat_count = num_iterations * gradient_accumulation_steps = 4`

## 2. Training Loop Flow

### 2.1 Step Counter Management

```python
# _prepare_inputs tracks steps
self._step = 0  # Increments after each call
generate_every = gradient_accumulation_steps * num_iterations = 4
```

### 2.2 Generation and Buffering Pattern

```
Step 0: Generate (512 completions) → Buffer → Return slice 0
Step 1: Use buffer → Return slice 1
Step 2: Use buffer → Return slice 0 (shuffled)
Step 3: Use buffer → Return slice 1 (shuffled)
Step 4: Generate (new 512 completions) → Buffer → Return slice 0
...
```

## 3. Async Pipeline Priming (Step 0)

### 3.1 Pipeline Initialization

```python
# Main process only
if self._step == 0 and num_steps_async > 0:
    self.async_generator.start()  # Starts worker thread
    
    # Submit num_steps_async batches ahead
    for i in range(num_steps_async):
        # All processes peek ahead
        future_batch = _async_dataloader.peek_ahead(i + 1)
        # Gather from all processes
        all_future_prompts = gather_object(prompts)  # Main gets 64 prompts
        # Main process submits
        if is_main_process:
            async_generator.submit_batch(BatchRequest(...))
```

## 4. Data Flow Through `_prepare_inputs`

### 4.1 Input Structure (Chat Format)

```python
# Input to _prepare_inputs (per process)
inputs = [
    {
        'prompt': [
            {'role': 'system', 'content': '...'},
            {'role': 'user', 'content': 'Solve x^2 = 4'}
        ],
        'answer': '2',
        'task': 'math'
    },
    # ... 15 more examples (16 total for gradient accumulation)
]
```

### 4.2 Generation Decision

```python
if self._step % generate_every == 0:  # Every 4 steps
    # Sync weights to vLLM
    _move_model_to_vllm()
    # Generate new completions
    processed_batch = _handle_async_generation(generation_batch)
    # Shuffle and split for reuse
    processed_batch = shuffle_tensor_dict(processed_batch)
    self._buffered_inputs = split_tensor_dict(processed_batch, gradient_accumulation_steps)
```

## 5. Async Generation Flow (`_handle_async_generation`)

### 5.1 Data Gathering Phase

```python
# Each process contributes its local data
prompts = [x['prompt'] for x in generation_batch]  # 16 prompts per process
all_prompts = gather_object(prompts)  # Main gets List[List[prompts]]
all_prompts = flatten(all_prompts)  # Main has 64 prompts total

# Types:
# all_prompts: List[List[Dict[str, str]]]  # 64 chat conversations
# all_answers: List[str]  # 64 answers
# all_tasks: List[str]  # 64 task names
```

### 5.2 Batch Submission (Main Process Only)

```python
if is_main_process:
    request = BatchRequest(
        batch_id=batch_id,
        env_inputs={
            'prompt': all_prompts,  # 64 prompts
            'answer': all_answers,  # 64 answers
            'task': all_tasks       # 64 tasks
        },
        processing_class=tokenizer,
        local_batch_size=16,  # For slice calculation later
        ...
    )
    async_generator.submit_batch(request)
```

### 5.3 Async Worker Thread Processing

```python
# In separate thread on main process
def _generate_batch(request):
    # Environment generates 8 completions per prompt
    env_results = env.generate(
        request.env_inputs,  # 64 prompts → 512 completions
        ...
    )
    
    # env_results structure:
    # {
    #     'prompt': List[List[Dict]] (64 items),
    #     'completion': List[List[Dict]] (512 items, 8 per prompt),
    #     'reward': List[float] (512 rewards),
    #     'state': List[Dict] (512 states),
    #     'reward_func1': List[float] (512 scores),
    #     ...
    # }
```

### 5.4 Token Processing (Main Process)

```python
# Process chat format for each of 512 completions
for i in range(512):
    prompt_ids, prompt_mask, completion_ids, completion_mask = process_chat_format(
        prompt=prompts[i // 8],  # Same prompt for 8 completions
        completion=completions[i],
        tokenizer=tokenizer
    )
    
# Shapes after padding:
# all_prompt_ids: torch.Tensor[512, max_prompt_len]
# all_completion_ids: torch.Tensor[512, max_completion_len]
# all_advantages: torch.Tensor[512]
```

### 5.5 Advantage Computation

```python
# Compute advantages using full batch statistics
rewards = torch.tensor(all_rewards)  # [512]
mean_grouped = rewards.view(-1, num_generations).mean(dim=1)  # [64]
std_grouped = rewards.view(-1, num_generations).std(dim=1)   # [64]

# Expand back to full size
mean_grouped = mean_grouped.repeat_interleave(num_generations)  # [512]
advantages = (rewards - mean_grouped) / (std_grouped + 1e-4)     # [512]
```

### 5.6 Broadcasting Results

```python
# Main process packages everything
broadcast_data = {
    'prompt_ids': all_prompt_ids,      # [512, max_prompt_len]
    'prompt_mask': all_prompt_mask,    # [512, max_prompt_len]
    'completion_ids': all_completion_ids,  # [512, max_completion_len]
    'completion_mask': all_completion_mask,  # [512, max_completion_len]
    'advantages': all_advantages,      # [512]
    'rewards': all_rewards,           # [512]
}

# Broadcast to all processes
broadcast_object_list([broadcast_data], from_process=0)
```

### 5.7 Process-Specific Slicing

```python
# Each process takes its slice
process_slice = slice(
    process_index * 16,      # Start
    (process_index + 1) * 16  # End
)

# Process 0: [0:16]
# Process 1: [16:32]
# Process 2: [32:48]
# Process 3: [48:64]

prompt_ids = broadcast_data['prompt_ids'][process_slice]  # [16, max_prompt_len]
```

## 6. Buffer Management and Iteration

### 6.1 Buffer Structure

```python
# After generation, main process has:
self._buffered_inputs = [
    {  # For gradient_accumulation step 0
        'prompt_ids': Tensor[8, max_prompt_len],
        'completion_ids': Tensor[8, max_completion_len],
        'advantages': Tensor[8],
        ...
    },
    {  # For gradient_accumulation step 1
        'prompt_ids': Tensor[8, max_prompt_len],
        ...
    }
]
```

### 6.2 Step-by-Step Data Flow

```
Step 0: Generate → buffer[0] → compute_loss with 8 samples
Step 1: buffer[1] → compute_loss with 8 samples → optimizer.step()
Step 2: buffer[0] (same data) → compute_loss with 8 samples
Step 3: buffer[1] (same data) → compute_loss with 8 samples → optimizer.step()
Step 4: Generate new → buffer[0] → compute_loss with 8 samples
```

## 7. Process Synchronization Points

1. **Before generation**: `accelerator.wait_for_everyone()`
2. **Before vLLM weight sync**: All processes participate in gather operations
3. **After weight sync**: `accelerator.wait_for_everyone()`
4. **Before broadcast**: `accelerator.wait_for_everyone()`
5. **After slicing**: `accelerator.wait_for_everyone()`

## 8. Common Issues and Debugging

### 8.1 Step Desynchronization

```python
# Debug check in _prepare_inputs
step_list = [self._step]
all_steps = gather_object(step_list)
if not all(s == self._step for s in all_steps):
    raise RuntimeError(f"Step desynchronization: {all_steps}")
```

### 8.2 Slice Validation

```python
# Validate slice bounds
if slice_end > total_samples:
    # Adjust slice to actual data bounds
    process_slice = slice(
        min(slice_start, total_samples),
        min(slice_end, total_samples)
    )
```

### 8.3 Empty Data Check

```python
if prompt_ids.shape[0] == 0:
    raise RuntimeError(
        f"Process {process_index} received empty data slice. "
        f"Total samples: {total_samples}, slice: [{slice_start}:{slice_end}]"
    )
```

## 9. Memory Flow Summary

1. **Dataset**: Original prompts (no duplication in memory)
2. **RepeatSampler**: Creates index patterns (minimal memory)
3. **Generation**: 512 completions generated once every 4 steps
4. **Buffering**: Split into gradient_accumulation_steps chunks
5. **Reuse**: Same 512 completions used for num_iterations passes
6. **GPU Memory**: Each GPU only holds its local slice (16 samples)

## 10. Key Invariants

1. **Generation frequency**: Every `gradient_accumulation_steps * num_iterations` steps
2. **Samples per generation**: `batch_size * num_processes * num_generations`
3. **Local batch size**: Always `per_device_train_batch_size`
4. **Advantage normalization**: Always computed on full generation batch (512 samples)
5. **Process coordination**: Main process handles all generation and processing

## 11. Detailed Type Annotations

### 11.1 Input Types

```python
# Chat format types
prompt: List[Dict[str, str]] = [
    {'role': 'system', 'content': str},
    {'role': 'user', 'content': str}
]

completion: List[Dict[str, str]] = [
    {'role': 'assistant', 'content': str},
    {'role': 'user', 'content': str},  # Optional environment response
    {'role': 'assistant', 'content': str}
]

# Dataset batch type
generation_batch: List[Dict[str, Any]] = [
    {
        'prompt': List[Dict[str, str]],
        'answer': str,
        'task': str
    }
]
```

### 11.2 Processing Types

```python
# Tokenized types
prompt_ids: List[int]  # e.g., [1, 2, 3, 4, 5]
prompt_mask: List[int]  # e.g., [1, 1, 1, 1, 1]
completion_ids: List[int]  # e.g., [6, 7, 8, 9]
completion_mask: List[int]  # e.g., [1, 1, 0, 1] (0 for masked env responses)

# Tensor types after padding
all_prompt_ids: torch.Tensor  # shape: [512, max_prompt_len]
all_completion_ids: torch.Tensor  # shape: [512, max_completion_len]
all_advantages: torch.Tensor  # shape: [512]
all_rewards: torch.Tensor  # shape: [512]
```

### 11.3 Broadcast Data Structure

```python
broadcast_data: Dict[str, torch.Tensor] = {
    'prompt_ids': torch.Tensor,      # [512, max_prompt_len]
    'prompt_mask': torch.Tensor,     # [512, max_prompt_len]
    'completion_ids': torch.Tensor,  # [512, max_completion_len]
    'completion_mask': torch.Tensor, # [512, max_completion_len]
    'advantages': torch.Tensor,      # [512]
    'rewards': torch.Tensor,         # [512]
}
```

## 12. Complete Example Walkthrough

Let's trace a complete example with concrete numbers:

### Configuration
- 4 GPUs
- `per_device_train_batch_size = 2`
- `num_generations = 4`
- `gradient_accumulation_steps = 2`
- `num_iterations = 2`

### Step 0: First Generation

1. **Each GPU loads batch**:
   - GPU 0: prompts [0, 0, 1, 1] (indices repeated by RepeatSampler)
   - GPU 1: prompts [2, 2, 3, 3]
   - GPU 2: prompts [4, 4, 5, 5]
   - GPU 3: prompts [6, 6, 7, 7]

2. **Gather to main process**:
   - All prompts: [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]
   - Unique prompts: [0, 1, 2, 3, 4, 5, 6, 7]

3. **Environment generates**:
   - Input: 8 unique prompts
   - Output: 32 completions (4 per prompt)
   - Rewards: [0.1, 0.3, -0.2, 0.4, ...] (32 values)

4. **Compute advantages**:
   ```python
   # Group by prompt
   rewards.view(-1, 4)  # Shape: [8, 4]
   # Compute stats per prompt
   mean = [0.15, 0.2, ...]  # 8 means
   std = [0.2, 0.15, ...]   # 8 stds
   # Normalize
   advantages = (rewards - mean) / std
   ```

5. **Tokenize all 32 samples**:
   ```python
   # For each completion
   prompt_ids[0] = [1, 23, 45, 67, 89]  # "System: ... User: Solve x^2=4"
   completion_ids[0] = [90, 91, 92, 93]  # "Assistant: x = ±2"
   ```

6. **Broadcast and slice**:
   - GPU 0 receives samples [0:8]
   - GPU 1 receives samples [8:16]
   - GPU 2 receives samples [16:24]
   - GPU 3 receives samples [24:32]

7. **Buffer management**:
   - Split 8 samples into 2 chunks (gradient accumulation)
   - buffer[0] = samples [0:4]
   - buffer[1] = samples [4:8]

8. **Return for step 0**:
   - Return buffer[0] with 4 samples

### Steps 1-3: Reusing Generations

- **Step 1**: Return buffer[1] (next 4 samples)
- **Step 2**: Return buffer[0] (same 4 samples, iteration 2)
- **Step 3**: Return buffer[1] (same 4 samples, iteration 2)

### Step 4: New Generation

- Repeat the entire process with next 8 prompts

## 13. Debugging Checklist

When debugging issues:

1. **Verify step synchronization**: All processes should have same `_step` value
2. **Check batch sizes**: Ensure consistent sizes across gathering operations
3. **Validate slicing**: Total samples should match expected count (batch_size * num_processes * num_generations)
4. **Monitor memory**: Each process should only hold its local slice
5. **Track generation timing**: Use `get_average_generation_time()` to monitor async performance
6. **Verify tokenization**: Check that chat format produces expected token sequences
7. **Confirm broadcasting**: All processes should receive same tensor shapes

## 14. Common Pitfalls

1. **Forgetting gather operations**: All processes must participate in gather_object
2. **Incorrect slice calculations**: Local batch size vs global batch size confusion
3. **Step counter drift**: Missing synchronization points can cause step desync
4. **Buffer index errors**: Modulo arithmetic with gradient_accumulation_steps
5. **Chat format tokenization**: Incremental tokenization must preserve prefix consistency
6. **Async queue management**: Not handling full queues or timeouts properly

## 15. Async Generation Deep Dive

### 15.1 Timing and Coordination

The async generator runs ahead of training by `num_steps_async` batches:

```python
# If num_steps_async = 3:
# While training uses batch 0, async generator works on batches 1, 2, 3
# When training moves to batch 1, async generator starts batch 4
```

### 15.2 Thread Communication Flow

```
Main Thread (Training)          Worker Thread (Generation)
─────────────────────          ─────────────────────────
1. Submit batch 0      ───→    1. Receive batch 0
2. Submit batch 1      ───→    2. Start generating batch 0
3. Submit batch 2      ───→    3. Receive batch 1
4. Wait for batch 0    ←───    4. Complete batch 0
5. Process batch 0              5. Start generating batch 1
6. Submit batch 3      ───→    6. Continue...
```

### 15.3 Queue Management

```python
# AsyncBatchGenerator queues:
request_queue: Queue()  # Unbounded input queue
result_queue: Queue(maxsize=max_queue_size)  # Bounded output queue

# Tracking structures:
pending_batches: Set[int]  # Currently being generated
completed_batches: Dict[int, BatchResult]  # Ready for retrieval
```

### 15.4 Error Handling

```python
# Generation failures are caught and returned as:
BatchResult(
    batch_id=batch_id,
    processed_results={},
    error=exception,
    generation_time=elapsed_time
)

# Training thread checks:
if batch_result.error:
    raise RuntimeError(f"Async generation failed: {batch_result.error}")
```

## 16. Performance Considerations

### 16.1 Memory Usage

- **Peak memory** occurs during broadcasting: O(total_samples * max_seq_len)
- **Per-process memory**: O(local_batch_size * max_seq_len)
- **vLLM server memory**: Separate from training GPUs

### 16.2 Optimal Configuration

```python
# Recommendations:
num_steps_async = 2-3  # Balance between memory and hiding latency
max_concurrent = 32-64  # For environment API calls
generation_batch_size = 16-32  # Per GPU
buffer_size = max(5, num_steps_async * 2)  # For AsyncDataLoaderWrapper
```

### 16.3 Bottleneck Analysis

1. **Generation latency**: Track with `get_average_generation_time()`
2. **Weight sync overhead**: Time `_move_model_to_vllm()`
3. **Gather/broadcast time**: Profile with accelerator timing tools
4. **Queue depth**: Monitor `get_pending_count()` and `get_completed_count()`

## 17. Summary

The GRPO trainer implements a sophisticated data flow that:

1. **Minimizes generation overhead** by generating many completions at once and reusing them
2. **Hides generation latency** through async pre-generation
3. **Distributes computation** with main process handling generation while all GPUs train
4. **Maintains synchronization** through careful coordination points
5. **Optimizes memory** by keeping only necessary data on each GPU

The key insight is that expensive generation happens infrequently (every 4 steps in our example) and is amortized across multiple training iterations, while the async pipeline ensures training never waits for generation. 