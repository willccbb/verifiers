# vLLM Server Architecture and Data Flow

## Overview

The vLLM server handles two types of operations:
1. **Generation requests** (chat/completion) - batched by sampling parameters
2. **Weight update requests** - must maintain NCCL synchronization with trainer

## Current Architecture

### Components

1. **Main Process (FastAPI)**
   - Receives HTTP requests
   - Coordinates with LLM workers via pipes
   - Manages batch processing for generation requests

2. **LLM Workers (Subprocesses)**
   - Run vLLM engine
   - Process generation requests
   - Receive weight updates via NCCL

3. **Batch Processor (Async Task)**
   - Groups generation requests by sampling parameters
   - Manages active/pending request pools
   - Tracks active generation count

4. **NCCL Communication**
   - Trainer broadcasts weights after POST request
   - Workers must be ready to receive immediately

### Data Flow

#### Generation Requests
```
1. Client → POST /v1/chat/completions → FastAPI endpoint
2. Request → Queued in request_queue
3. Batch Processor:
   - Groups by PoolSignature (model, sampling params)
   - Activates pools when no active requests
   - Sends to LLM workers
4. Workers → Generate → Results back to client
```

#### Weight Updates
```
1. Trainer → POST /update_named_param → FastAPI endpoint
2. Server immediately notifies workers via pipe (synchronous)
3. Workers prepare for NCCL broadcast
4. Trainer → NCCL broadcast → Workers receive
5. Workers update model weights
```

## Implementation Details

### Synchronous Weight Updates

The weight update process is now simplified to use synchronous communication:

```python
@app.post("/update_named_param/")
async def update_named_param(request: UpdateWeightsRequest):
    # Notify workers IMMEDIATELY so they're ready for NCCL
    dtype = getattr(torch, request.dtype.split(".")[-1])
    kwargs = {"method": "update_named_param", "args": (request.name, dtype, tuple(request.shape))}
    
    # Send to all workers synchronously
    for i, connection in enumerate(connections):
        connection.send({"type": "fire_and_forget", "method": "collective_rpc", "kwargs": kwargs})
    
    return {"message": "Weight update processed"}
```

This approach:
- Eliminates complex async/threading logic
- Ensures workers are notified before HTTP response returns
- Maintains proper NCCL synchronization timing

### Batch Processing for Generation

Generation requests continue to use async batching for efficiency:
- Requests grouped by identical sampling parameters
- Token-chunk dynamic batching for long generations
- Async processing doesn't interfere with weight updates

## Key Design Decisions

1. **No Mode Switching**: Removed complex server mode management. Weight updates and generation can coexist.

2. **Synchronous Pipe Communication**: Direct pipe sends ensure immediate worker notification without thread pool overhead.

3. **Fire-and-Forget for Weight Updates**: Workers don't send responses for weight updates, preventing blocking.

4. **Separate Concerns**: Generation batching and weight updates are independent operations.

## Troubleshooting

### Common Issues

1. **NCCL Timeout Errors**
   - Check that all processes join the StatelessProcessGroup correctly
   - Verify world_size calculation matches between client and server
   - Ensure no network/firewall issues blocking NCCL communication

2. **Pipe Communication Failures**
   - Pipes are synchronous - ensure workers aren't blocked
   - Check for exceptions in worker processes

3. **Generation Request Timeouts**
   - Adjust `batch_request_timeout_seconds` if needed
   - Check if workers are overloaded

## Future Improvements

1. **Health Monitoring**: Add metrics for weight update latency and success rate
2. **Error Recovery**: Implement retry logic for failed weight updates
3. **Load Balancing**: Distribute generation requests across multiple workers
4. **Graceful Degradation**: Continue serving generation requests even if weight updates fail

## 2x4 GPU Hang Debugging Summary

### Problem Description
- 2x1 GPU setup (2 GPUs on single node) works perfectly with NCCL_P2P_DISABLE=1
- 2x4 GPU setup (8 GPUs on single node) hangs after completing first batch of generation/evaluation
- Hang occurs in async batch generator where `get_batch()` waits indefinitely
- Appears to be a synchronization issue between the 4 training processes when scaling beyond 2 GPUs

### Debugging Attempts

#### 1. Mode Switching Complexity
**Issue**: Original implementation had complex server mode switching between "generation" and "update" modes  
**Solution**: Removed mode switching entirely, simplified to handle requests directly  
**Result**: Improved stability but didn't fix 2x4 hang

#### 2. Asynchronous Task Management
**Issue**: Server was using fire-and-forget async tasks that could complete out of order  
**Solution**: Changed to synchronous pipe communication for weight updates  
**Result**: Better synchronization but 2x4 still hangs

#### 3. NCCL Synchronization
**Issue**: Weight updates were failing because workers weren't ready for NCCL broadcasts  
**Solution**: Ensured workers are notified synchronously before client broadcasts  
**Result**: Fixed weight update failures but generation hang persists

#### 4. Rapid Weight Updates
**Issue**: Hundreds of individual parameter updates were overwhelming the system  
**Solution**: Implemented batch updates with semaphore throttling (max 5 concurrent)  
**Result**: More efficient updates but 2x4 hang remains

#### 5. Skip Initial Weight Sync
**Issue**: Unnecessary weight sync at step 0 since vLLM already has initial weights  
**Solution**: Skip weight sync when global_step == 0  
**Result**: More efficient startup but doesn't address the hang

#### 6. Improved Logging
**Added**: Extensive debug logging for NCCL operations, batch processing, and async generation  
**Result**: Revealed hang occurs in async batch generator waiting for batches

### Root Cause Analysis

The hang appears to be in the async batch generator where:
1. Main process submits batch requests
2. All processes wait at `get_batch()` 
3. The batch never completes, causing indefinite wait

Key observations:
- Only affects multi-GPU setups with >2 GPUs per node
- Happens after first successful batch completes
- Suggests a synchronization issue in gather operations or batch submission

### Additional Ideas to Try

#### 1. Process Synchronization
- Add explicit barriers after batch submission/completion
- Verify all processes are submitting batches correctly
- Check if non-main processes are incorrectly trying to submit

#### 2. Async Generator State
- Add timeout to `get_batch()` with better error reporting
- Log batch state transitions (pending → processing → complete)
- Track which process is stuck and at what stage

#### 3. NCCL Environment
- Test with different NCCL settings:
  - `NCCL_ASYNC_ERROR_HANDLING=0`
  - `NCCL_TREE_THRESHOLD=0`
  - `NCCL_IB_DISABLE=1` (if using InfiniBand)
- Try `NCCL_DEBUG=INFO` for more detailed NCCL logs

#### 4. Gather Operation Issues
- The hang might be in `gather_object()` calls
- Try using `broadcast_object_list()` instead of gather
- Add timeouts to collective operations

#### 5. Batch Submission Logic
- Verify only main process submits to async generator
- Check if batch IDs are getting out of sync across processes
- Ensure proper cleanup of completed batches

#### 6. Memory/Resource Issues
- Monitor GPU memory during the hang
- Check if there's a resource leak in async generation
- Verify no deadlock in queue operations

#### 7. Alternative Approaches
- Try disabling async generation for 2x4 setup as a workaround
- Implement a simpler synchronous generation path for debugging
- Test with different batch sizes and generation parameters

### Next Steps

1. **Add Comprehensive Logging**:
   - Log every state transition in AsyncBatchGenerator
   - Track batch lifecycle from submission to completion
   - Monitor queue sizes and pending operations

2. **Implement Timeouts**:
   - Add configurable timeout to `get_batch()`
   - Fail gracefully with detailed error information
   - Include batch state dump on timeout

3. **Test Isolation**:
   - Run with single batch to isolate the issue
   - Disable weight updates to rule out interference
   - Test with minimal generation parameters

4. **Process Coordination**:
   - Verify accelerator.wait_for_everyone() usage
   - Check if all processes reach the same code paths
   - Add process-specific logging to track divergence 