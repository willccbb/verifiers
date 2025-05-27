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