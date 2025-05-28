# vLLM OpenAI Server Request Lifecycle Documentation

## Overview

This document describes the complete lifecycle of a request in the vLLM OpenAI-compatible server with token-chunk dynamic batching and weight synchronization capabilities.

## Request Lifecycle

### 1. Request Reception
- Client sends POST request to `/v1/chat/completions` or `/v1/completions`
- Server creates a unique request ID and `PooledRequestState` object
- Request is enqueued to the global `request_queue` (asyncio.Queue)
- Client waits on `completion_event.wait()` with a 300-second timeout

### 2. Pool Assignment
The batch processing loop continuously:
1. **Ingests new requests** from `request_queue` into `pending_requests_by_signature`
2. **Activates a pool** when `active_pool_requests` is empty
3. **Merges matching requests** into the active pool if signatures match

### 3. Sub-batch Processing
For each iteration:
1. **Filter available requests**: `available_requests = [req for req in active_pool_requests if not req.is_complete]`
2. **Select sub-batch**: Take up to `max_batch_size` requests
3. **Calculate chunk size**: `min(token_chunk_size, min_tokens_remaining)`
4. **Prepare inputs**:
   - Chat: Separate first-chunk vs continuing requests
   - Completion: Concatenate prompts with accumulated content

### 4. LLM Communication
1. **Send request** via `send_and_recv()` using multiprocessing Pipe
2. **Worker processes** request through vLLM engine
3. **Receive results** and update request states

### 5. State Updates
For each request in the batch:
- Update `accumulated_content` with new tokens
- Update `generated_token_count` and `last_chunk_token_count`
- Set `generation_stopped` if chunk < 64 tokens
- Check completion conditions

### 6. Completion Detection
A request is considered complete (`is_complete`) when:
- `completed_and_signaled` is True
- `error` is not None
- `generation_stopped` is True
- `finish_reason` is "stop" or other non-"length" reason
- `generated_token_count >= effective_max_tokens`
- Less than 8 tokens of room remaining
- Last chunk generated < 64 tokens

### 7. Response Finalization
- Complete requests are moved to `completed_in_sub_batch`
- Response objects are created and placed in `result_container`
- `completion_event.set()` signals the waiting client
- `completed_and_signaled` is set to True

## Critical Edge Cases and Potential Deadlocks

### 1. **The 1499-Token Edge Case**
- **Scenario**: Request generates 1499 tokens, then only 1 token in next chunk
- **Issue**: Without proper detection, request keeps trying to generate more
- **Current Solution**: Track `last_chunk_token_count` and mark complete if < 64

### 2. **Insufficient Room for Generation**
- **Scenario**: Request has < 64 tokens room but still tries to generate
- **Issue**: vLLM might hang or behave unexpectedly
- **Current Solution**: Calculate `min_tokens_remaining` and limit chunk size

### 3. **Worker Communication Deadlock**
- **Scenario**: LLM worker is busy/crashed, `send_and_recv()` blocks indefinitely
- **Issue**: No timeout on pipe communication
- **Risk**: Entire batch processor hangs

### 4. **Weight Update Conflicts**
- **Scenario**: Weight update arrives while generation is active
- **Current Solution**: Wait for `active_generation_count == 0`
- **Risk**: Long-running generations block weight updates

### 5. **Empty Batch Processing**
- **Scenario**: All requests in batch are already complete
- **Issue**: Might still try to send empty request to vLLM
- **Current Solution**: Check and skip if no active requests

### 6. **Pool Starvation**
- **Scenario**: One slow request blocks entire pool
- **Issue**: Other requests with same signature wait indefinitely
- **Risk**: Cascading timeouts

## Proposed Solutions

### Solution 1: Chunk-Level Progress Timeout
```python
# Track progress at chunk level
class PooledRequestState:
    last_progress_time: float = 0
    consecutive_minimal_chunks: int = 0

# In batch processor:
if new_token_count < 8:  # Minimal progress
    req_state.consecutive_minimal_chunks += 1
    if req_state.consecutive_minimal_chunks >= 3:
        req_state.generation_stopped = True
else:
    req_state.consecutive_minimal_chunks = 0
    req_state.last_progress_time = time.time()
```

**Pros**: Prevents infinite loops on stuck generations
**Cons**: Might prematurely stop slow but valid generations

### Solution 2: Worker Health Monitoring
```python
# Add heartbeat mechanism
async def worker_heartbeat_monitor(connections):
    while True:
        for conn in connections:
            if not conn.poll(timeout=5):
                logger.error(f"Worker not responding")
                # Restart worker or mark requests as failed
        await asyncio.sleep(10)
```

**Pros**: Detects dead workers quickly
**Cons**: Adds complexity and overhead

### Solution 3: Aggressive Early Termination
```python
# More aggressive completion criteria
@property
def is_complete(self) -> bool:
    # ... existing checks ...
    
    # Stop if we're within 5% of max tokens
    if self.generated_token_count > 0.95 * self.effective_max_tokens:
        return True
    
    # Stop if last 2 chunks were small
    if hasattr(self, 'chunk_history') and len(self.chunk_history) >= 2:
        if all(c < 16 for c in self.chunk_history[-2:]):
            return True
```

**Pros**: Prevents edge cases near token limits
**Cons**: Might lose some valid generation

### Solution 4: Request-Level Timeouts
```python
# Add per-request generation timeout
class PooledRequestState:
    generation_start_time: float = 0
    max_generation_time: float = 60  # seconds

# Check in batch processor
if time.time() - req_state.generation_start_time > req_state.max_generation_time:
    req_state.error = TimeoutError("Generation timeout")
    req_state.generation_stopped = True
```

**Pros**: Guarantees no request hangs forever
**Cons**: Need to tune timeout values carefully

### Solution 5: Async Worker Communication
```python
# Replace blocking send_and_recv with async version
async def async_send_and_recv(conn, payload, timeout=30):
    loop = asyncio.get_running_loop()
    
    # Send in thread
    await loop.run_in_executor(None, conn.send, payload)
    
    # Poll with timeout
    start = time.time()
    while time.time() - start < timeout:
        if await loop.run_in_executor(None, conn.poll, 0.1):
            return await loop.run_in_executor(None, conn.recv)
        await asyncio.sleep(0.1)
    
    raise TimeoutError("Worker communication timeout")
```

**Pros**: Non-blocking, allows graceful timeout handling
**Cons**: More complex implementation

## Recommended Approach

Implement a combination of:
1. **Solution 1** (chunk-level progress timeout) for generation-level protection
2. **Solution 3** (aggressive early termination) to avoid edge cases
3. **Solution 5** (async worker communication) to prevent communication deadlocks

This provides multiple layers of protection against different failure modes while maintaining reasonable generation quality.

## Implementation Priority

1. **High Priority**: Fix worker communication timeout (Solution 5)
2. **High Priority**: Add chunk-level progress detection (Solution 1)
3. **Medium Priority**: Implement aggressive termination near limits (Solution 3)
4. **Low Priority**: Add worker health monitoring (Solution 2)
5. **Low Priority**: Add request-level timeouts (Solution 4)

## Testing Recommendations

1. Test with requests that generate exactly `max_tokens - 1` tokens
2. Test with very long prompts that leave little generation room
3. Test with concurrent weight updates during generation
4. Test with killed worker processes
5. Test with slow/unresponsive workers
6. Test with mixed request sizes in same pool 