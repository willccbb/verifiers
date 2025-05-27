# Asynchronous Generation for GRPO Training

## Overview

The async generation feature allows GRPO training to generate future batches in a separate thread while the GPU continues training on current batches. This significantly improves GPU utilization by eliminating idle time during generation.

## Architecture

### Components

1. **AsyncBatchGenerator**: Manages generation in a separate thread
   - Maintains a queue of generation requests and results
   - Handles thread synchronization and error recovery
   - Tracks generation timing statistics

2. **AsyncDataLoaderWrapper**: Wraps the training dataloader
   - Provides lookahead capability to peek at future batches
   - Maintains a buffer of upcoming batches
   - Thread-safe access to future batch data

3. **GRPOEnvTrainer Integration**: Modified trainer logic
   - Submits future batches for generation while training
   - Retrieves completed generations when needed
   - Handles cleanup on training completion

### Data Flow

```
Training Loop                    Async Generator Thread
     |                                    |
     v                                    |
[Current Batch] ----submit future-----> [Queue]
     |              batches               |
     v                                    v
[Train on GPU]                    [env.generate()]
     |                                    |
     v                                    v
[Need Next Batch] <---retrieve----  [Completed]
     |              result              Results
     v
[Continue Training]
```

## Configuration

Enable async generation by setting these parameters in `GRPOEnvConfig`:

```python
training_args = GRPOEnvConfig(
    # Enable async generation
    use_async_generation=True,
    
    # Number of batches to generate ahead (default: 1)
    num_steps_async=2,
    
    # Timeout for generation (default: 300 seconds)
    async_generation_timeout=300.0,
    
    # Maximum queue size (default: 2 * num_steps_async)
    async_max_queue_size=4,
    
    # ... other training args ...
)
```

## Performance Considerations

### Benefits
- **Improved GPU Utilization**: GPU continues training while CPU generates
- **Reduced Training Time**: Eliminates generation bottleneck
- **Scalable**: Can generate multiple batches ahead

### Trade-offs
- **Memory Usage**: Stores future batches in memory
- **Slightly Off-Policy**: Future batches use slightly older model weights
- **Complexity**: Additional thread management overhead

### Recommended Settings

For most use cases:
- `num_steps_async=1`: One-step off-policy, minimal memory overhead
- `num_steps_async=2-3`: Good balance of performance and memory

For memory-constrained environments:
- Keep `num_steps_async=1`
- Reduce `async_max_queue_size`

For maximum performance:
- Increase `num_steps_async` based on generation/training time ratio
- Monitor memory usage

## Implementation Details

### Batch Coordination

The system maintains consistency across distributed training:
1. All processes gather their local batches
2. Main process performs generation
3. Results are broadcast to all processes
4. Each process slices its portion

### Error Handling

- Generation failures are caught and re-raised in main thread
- Automatic retry with exponential backoff
- Graceful shutdown on repeated failures

### Weight Synchronization

When using async generation with vLLM:
- Weights are synced at the start of each training step
- Future batches use slightly older weights (one-step off-policy by default)
- This trade-off is generally acceptable for improved performance

## Monitoring

Track async generation performance:
- Average generation time: `trainer.async_generator.get_average_generation_time()`
- Pending batches: `trainer.async_generator.get_pending_count()`
- Queue utilization: Monitor memory usage

## Future Extensions

The architecture supports:
- Priority-based batch selection
- Dynamic batch size adjustment
- Multi-GPU generation
- Advanced sampling strategies (e.g., curriculum learning) 