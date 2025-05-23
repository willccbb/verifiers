-- user notes (don't edit) --

Let's work on a design doc for a BatchSampler feature in BATCH_SAMPLER.md (this file)

Currently, the inference/scoring and training stages are blocking, as the train steps always are requesting rollouts from the latest copy of the model weights in vLLM (as synchronized by the update_weights endpoint).

We want to instead use a BatchSampler module which will enable a few different things: 
- off-policy updates, where the inference workers are one or more steps stale when generating rollouts
- aggregation and "wrapping" of computed/processed env (at token group level) for sending to the core trainer loop
- clean abstractions for easy experimentation with different sampling methods -- for example, the user might want to do different "replay buffer" approaches, where rollouts stay in a pool (with appropriate metadata, e.g. global prompt IDs for groups, indicator of how stale) to be sampled from, potentially doing multiple uses/repackagings of past samples, reusing prompts across multiple gradient steps, etc.

This should take the place of the RepeatSampler, but I think we probably want to move the group duplication (with num_generations) and reward computation before (inside)?

What we want to achieve is:
- Data flows from the DataLoader into the BatchSampler, potentially moving "ahead" of the current train step
- BatchSampler runs in parallel to the trainer, e.g. in a separate process.
- Core trainer update loop can poll for a (complete) batch from the sampler. It receives a set of completions/rewards if ready, or it waits (polling e.g. every 0.5s).

The default mode of operation should be one-step off policy, with generation for batch N happening in parallel to training on batch N-1, with a simple flag for sync on-policy. 

Think carefully about how this will interact with other sampling related flags. I want to still keep gradient accumulation logic behaving as normal, but I'm happy to overwrite the current way that num_iterations is handled (and instead expose it as an option within BatchSampler). Also, think about any

Also --- we want the evaluation logic of the trainer to use the Environment evaluate function (instead of typical Trainer evaluation). We'll have num_generations = 1 in this step as default, and will skip computation of KL/loss for evals (just rewards).

We want to stay within the typical patterns of transformers' Trainer within reason, but our primary goal is maintainable clean code achieving the above goals. Think deeply about possible footguns we'll run into, different approaches we can take and their tradeoffs, and then ultimately propose what you think is the best solution and why. 

Don't write any code in the files, but give a full implementation spec here. Be detailed in your spec, including key code snippets with any essential data flow.

-- 

# GRPO Batch Sampler Refactor for Trainer

## Overview

This document outlines the design for a `BatchSampler` module that enables asynchronous, off-policy training for the GRPO trainer. The current implementation is blocking - training waits for generation which waits for weight synchronization. The new design allows generation and training to run in parallel with configurable staleness policies.

## Current Pain Points

1. **Blocking Architecture**: Training stops during generation and weight sync
2. **On-Policy Only**: No support for off-policy updates or replay buffers  
3. **Fixed Sampling**: Limited experimentation with different sampling strategies
4. **Inefficient Resource Usage**: GPU idle during generation, CPU idle during training

## Goals

1. **Parallel Execution**: Generation and training run concurrently
2. **Off-Policy Support**: Configurable staleness (1-step default, on-policy option)
3. **Flexible Sampling**: Design surface for future strategies (replay buffers, multi-use, etc.)
4. **Environment Integration**: Simple env.evaluate() for periodic evaluation
5. **Trainer Integration**: Work within Trainer patterns while intercepting data flow

## Architecture Overview

### High-Level Data Flow
```
DataLoader → BatchSampler Process → BatchQueue → Trainer._generate_and_score_completions()
                     ↓
             Environment.generate()
                     ↓
             process_environment_results()
                     ↓
              Ready batch tensors
```

### Key Components

1. **BatchSampler**: Separate process handling generation and buffering
2. **SamplingStrategy**: OnPolicy and OffPolicy implementations  
3. **BatchQueue**: Process-safe communication channel
4. **WeightSynchronizer**: Manages model weight updates to vLLM

## Detailed Step-Level Data Flow

### Trainer Integration Points

The core insight is to intercept the trainer's data flow at `_generate_and_score_completions()` while leaving the rest of the training loop intact.

```python
class GRPOEnvTrainer(Trainer):
    def __init__(self, *args, batch_sampler_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_sampler_config = batch_sampler_config or BatchSamplerConfig()
        self.batch_sampler_process = None
        self.batch_queue = None
        self.control_queue = None
        
    def get_train_dataloader(self):
        """Override to start BatchSampler and return proxy dataloader."""
        # Get the original dataloader for the BatchSampler to consume
        original_dataloader = super().get_train_dataloader()
        
        # Start BatchSampler process
        self._start_batch_sampler(original_dataloader)
        
        # Return a proxy dataloader that yields dummy data
        # The real data comes from BatchSampler via _generate_and_score_completions
        return ProxyDataLoader(len(original_dataloader))
    
    def _start_batch_sampler(self, dataloader):
        """Initialize and start the BatchSampler process."""
        import multiprocessing as mp
        
        # Create communication channels
        self.batch_queue = mp.Queue(maxsize=self.batch_sampler_config.queue_size)
        self.control_queue = mp.Queue()
        
        # Create and start process
        self.batch_sampler_process = BatchSamplerProcess(
            dataloader=dataloader,
            env=self.env,
            strategy=self._create_strategy(),
            batch_queue=self.batch_queue,
            control_queue=self.control_queue,
            config=self.batch_sampler_config
        )
        self.batch_sampler_process.start()
    
    def _generate_and_score_completions(self, inputs):
        """Override to poll BatchSampler instead of generating directly."""
        # Poll for ready batch from BatchSampler
        try:
            batch_data = self.batch_queue.get(
                timeout=self.batch_sampler_config.batch_timeout
            )
        except queue.Empty:
            raise RuntimeError("BatchSampler timeout - no batch ready")
        
        # Handle errors from BatchSampler process
        if "error" in batch_data:
            raise RuntimeError(f"BatchSampler error: {batch_data['error']}")
        
        # Move tensors to training device and slice for this process
        device = self.accelerator.device
        local_batch = self._prepare_local_batch(batch_data, device)
        
        # Log staleness metrics
        if "metadata" in batch_data:
            staleness = self.state.global_step - batch_data["metadata"]["generated_at_step"]
            self._metrics["train"]["batch_staleness"].append(staleness)
        
        return local_batch
    
    def _prepare_local_batch(self, batch_data, device):
        """Move tensors to device and slice for this process."""
        # Move to device
        for key, value in batch_data.items():
            if isinstance(value, torch.Tensor):
                batch_data[key] = value.to(device)
        
        # Slice for this process (multi-GPU support)
        if self.accelerator.num_processes > 1:
            process_slice = slice(
                self.accelerator.process_index * self.args.per_device_train_batch_size,
                (self.accelerator.process_index + 1) * self.args.per_device_train_batch_size
            )
            for key, value in batch_data.items():
                if isinstance(value, torch.Tensor) and len(value.shape) > 0:
                    batch_data[key] = value[process_slice]
        
        return batch_data
```

### BatchSampler Process Implementation

```python
class BatchSamplerProcess(multiprocessing.Process):
    """Separate process that continuously generates batches."""
    
    def __init__(self, dataloader, env, strategy, batch_queue, control_queue, config):
        super().__init__()
        self.dataloader = dataloader
        self.env = env
        self.strategy = strategy
        self.batch_queue = batch_queue
        self.control_queue = control_queue
        self.config = config
        self.current_step = 0
        self.should_stop = False
        
    def run(self):
        """Main process loop - continuously generate batches."""
        try:
            # Initialize environment in this process context
            self._setup_process_environment()
            
            # Main generation loop
            for raw_batch in self._infinite_dataloader():
                if self.should_stop:
                    break
                
                # Check for control messages (weight updates, stop signals)
                self._process_control_messages()
                
                # Generate batch using strategy
                processed_batch = self.strategy.generate_batch(
                    raw_batch, self.env, self.current_step
                )
                
                # Add metadata for staleness tracking
                processed_batch["metadata"] = {
                    "generated_at_step": self.current_step,
                    "timestamp": time.time(),
                    "strategy": self.strategy.__class__.__name__
                }
                
                # Put in queue (blocks if queue is full - provides backpressure)
                self.batch_queue.put(processed_batch)
                self.current_step += 1
                
        except Exception as e:
            # Send error to trainer
            self.batch_queue.put({"error": str(e), "traceback": traceback.format_exc()})
        finally:
            self._cleanup()
    
    def _infinite_dataloader(self):
        """Infinite iterator over dataloader."""
        while not self.should_stop:
            for batch in self.dataloader:
                if self.should_stop:
                    return
                yield batch
    
    def _process_control_messages(self):
        """Handle control messages from trainer process."""
        try:
            while True:  # Process all available messages
                message = self.control_queue.get_nowait()
                
                if message["type"] == "weight_update":
                    self.strategy.handle_weight_update(message["step"])
                elif message["type"] == "stop":
                    self.should_stop = True
                    break
                elif message["type"] == "config_update":
                    self.strategy.update_config(message["config"])
                    
        except queue.Empty:
            pass  # No messages available
    
    def _setup_process_environment(self):
        """Initialize environment and clients in process context."""
        # Re-initialize OpenAI client (can't pickle across processes)
        self.env.client = openai.OpenAI(
            base_url=f"http://{self.config.vllm_host}:{self.config.vllm_port}/v1",
            api_key="EMPTY"
        )
```

### Weight Synchronization Flow

```python
class GRPOEnvTrainer(Trainer):
    def _on_step_end(self, args, state, control):
        """Hook into trainer step completion for weight sync."""
        super()._on_step_end(args, state, control)
        
        # Sync weights to vLLM after each training step
        if state.global_step != self._last_synced_step:
            self._sync_weights_to_vllm(state.global_step)
            self._last_synced_step = state.global_step
            
            # Notify BatchSampler of weight update
            try:
                self.control_queue.put_nowait({
                    "type": "weight_update", 
                    "step": state.global_step,
                    "timestamp": time.time()
                })
            except queue.Full:
                self.logger.warning("Control queue full - weight update signal dropped")
    
    def _sync_weights_to_vllm(self, step):
        """Synchronize current model weights to vLLM server."""
        # Use existing _move_model_to_vllm logic
        # This happens on main process only, then vLLM serves to BatchSampler
        if self.accelerator.is_main_process:
            self._move_model_to_vllm()
            self.logger.debug(f"Synced weights to vLLM for step {step}")
```

### Sampling Strategies

```python
class SamplingStrategy(ABC):
    """Base class for batch generation strategies."""
    
    @abstractmethod
    def generate_batch(self, raw_batch: Dict, env: Environment, step: int) -> Dict:
        """Generate and process a batch for training."""
        pass
    
    def handle_weight_update(self, step: int):
        """Handle notification of weight update."""
        pass

class OnPolicyStrategy(SamplingStrategy):
    """Wait for weight sync before each batch generation."""
    
    def __init__(self, sync_timeout: float = 30.0):
        self.sync_timeout = sync_timeout
        self.last_weight_step = -1
        
    def generate_batch(self, raw_batch: Dict, env: Environment, step: int) -> Dict:
        # For on-policy, wait for weights to be synced to current step
        if step > 0:  # Skip wait for first batch
            self._wait_for_weight_sync(step)
        
        # Generate with synced weights
        env_results = env.generate(
            raw_batch,
            score_rollouts=True,
            max_concurrent=32
        )
        
        # Process into trainer format (using existing logic)
        return env.process_environment_results(
            env_results,
            processing_class=env.processing_class,  # Need to pass this somehow
            num_generations=env.num_generations,
            device=torch.device("cpu")  # Process on CPU, move to GPU in trainer
        )
    
    def _wait_for_weight_sync(self, target_step: int):
        """Block until weights are synced to target step."""
        start_time = time.time()
        while (time.time() - start_time) < self.sync_timeout:
            if self.last_weight_step >= target_step:
                return
            time.sleep(0.05)  # Short polling interval
        
        raise TimeoutError(f"Weight sync timeout waiting for step {target_step}")
    
    def handle_weight_update(self, step: int):
        self.last_weight_step = step

class OffPolicyStrategy(SamplingStrategy):
    """Generate with potentially stale weights (max staleness = 1 by default)."""
    
    def __init__(self, max_staleness: int = 1):
        self.max_staleness = max_staleness
        self.last_weight_step = -1
        
    def generate_batch(self, raw_batch: Dict, env: Environment, step: int) -> Dict:
        # Check if we're too stale
        staleness = step - self.last_weight_step
        if staleness > self.max_staleness:
            # Wait for at least one weight update
            self._wait_for_any_update()
        
        # Generate with current (possibly stale) weights
        env_results = env.generate(
            raw_batch,
            score_rollouts=True,
            max_concurrent=32
        )
        
        return env.process_environment_results(
            env_results,
            processing_class=env.processing_class,
            num_generations=env.num_generations, 
            device=torch.device("cpu")
        )
    
    def _wait_for_any_update(self):
        """Wait for any weight update to reduce staleness."""
        current_step = self.last_weight_step
        timeout = 10.0  # Shorter timeout for off-policy
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            if self.last_weight_step > current_step:
                return
            time.sleep(0.1)
        
        # Continue anyway if timeout - off-policy can tolerate staleness
        self.logger.warning(f"Off-policy timeout - continuing with staleness")
    
    def handle_weight_update(self, step: int):
        self.last_weight_step = step
```

### Proxy DataLoader

```python
class ProxyDataLoader:
    """Dummy dataloader that yields placeholder data for trainer compatibility."""
    
    def __init__(self, length: int):
        self.length = length
    
    def __iter__(self):
        # Yield dummy batches - real data comes from BatchSampler
        for i in range(self.length):
            yield {"proxy_batch_id": i}
    
    def __len__(self):
        return self.length
```

### Configuration

```python
@dataclass  
class BatchSamplerConfig:
    """Configuration for BatchSampler."""
    
    # Strategy
    strategy: str = "off_policy"  # "on_policy" or "off_policy"
    max_staleness: int = 1  # For off_policy strategy
    
    # Process communication
    queue_size: int = 5  # Max batches buffered
    batch_timeout: float = 1.0  # Trainer polling timeout
    
    # Weight sync
    sync_timeout: float = 30.0  # On-policy sync timeout
    
    # vLLM connection
    vllm_host: str = "localhost"
    vllm_port: int = 8000
```

## Simple Evaluation Integration

Evaluation should be minimal - just call env.evaluate() periodically:

```python
class GRPOEnvTrainer(Trainer):
    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        """Override to add Environment evaluation."""
        
        # Do standard trainer evaluation first
        super()._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
        
        # Add Environment evaluation if it's time
        if (self.state.global_step > 0 and 
            self.state.global_step % self.args.eval_steps == 0):
            
            # Sync weights before evaluation
            if self.state.global_step != self._last_synced_step:
                self._sync_weights_to_vllm(self.state.global_step)
            
            # Run Environment evaluation
            eval_results = self.env.evaluate(
                client=self.oai_client,
                model=self._get_model_name(),
                sampling_args={'temperature': 0.0, 'n': 1},
                num_samples=self.args.eval_samples if hasattr(self.args, 'eval_samples') else -1
            )
            
            # Log basic metrics
            if 'reward' in eval_results:
                self.log({'eval_env_reward_mean': np.mean(eval_results['reward'])})
```

## Process Lifecycle Management

```python
class GRPOEnvTrainer(Trainer):
    def train(self, *args, **kwargs):
        """Override to ensure proper process cleanup."""
        try:
            return super().train(*args, **kwargs)
        finally:
            self._cleanup_batch_sampler()
    
    def _cleanup_batch_sampler(self):
        """Ensure BatchSampler process is properly terminated."""
        if self.batch_sampler_process and self.batch_sampler_process.is_alive():
            # Send stop signal
            try:
                self.control_queue.put_nowait({"type": "stop"})
            except:
                pass
            
            # Wait for graceful shutdown
            self.batch_sampler_process.join(timeout=5.0)
            
            # Force terminate if needed
            if self.batch_sampler_process.is_alive():
                self.batch_sampler_process.terminate()
                self.batch_sampler_process.join()
```

## Implementation Plan

### Phase 1: Core Infrastructure
1. `BatchSamplerProcess` with `OnPolicyStrategy`
2. Process communication and weight sync
3. Trainer integration points
4. Basic error handling

### Phase 2: Off-Policy Support  
1. `OffPolicyStrategy` with staleness control
2. Metadata tracking and logging
3. Timeout handling and recovery

### Phase 3: Future Extensions
1. Design surface ready for replay strategies
2. Advanced batching and caching
3. Performance optimizations

The key insight is intercepting the trainer's data flow at `_generate_and_score_completions()` while keeping the rest of the training loop intact. This allows us to swap out the data source without disrupting gradient accumulation, checkpointing, or other trainer behaviors.