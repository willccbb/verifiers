import threading
import queue
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import time
import torch
from collections import deque


@dataclass
class BatchRequest:
    """Request for batch generation"""
    batch_id: int
    env_inputs: Dict[str, List[Any]]
    processing_class: Any
    mask_env_responses: bool
    max_completion_length: int
    mask_truncated_completions: bool
    max_concurrent: int
    device: torch.device
    accelerator: Any
    process_index: int
    num_processes: int
    local_batch_size: int


@dataclass
class BatchResult:
    """Result from batch generation"""
    batch_id: int
    processed_results: Dict[str, Any]
    generation_time: float = 0.0
    all_reward_dict: Dict[str, List[float]] = field(default_factory=dict)  # All reward scores
    completions: List[Any] = field(default_factory=list)  # Store completions for logging
    prompts: List[Any] = field(default_factory=list)  # Store prompts for logging


class AsyncBatchGenerator:
    """
    Manages asynchronous batch generation for GRPO training.
    
    This class runs generation in a separate thread, allowing training to continue
    while future batches are being generated. It maintains a queue of pending
    generation requests and completed results.
    """
    
    def __init__(
        self,
        env,
        client,
        model_name: str,
        sampling_args: Dict[str, Any],
        num_batches_ahead: int = 1,
        max_queue_size: Optional[int] = None,
        generation_timeout: float = 300.0,  # 5 minutes default
    ):
        self.env = env
        self.client = client
        self.model_name = model_name
        self.sampling_args = sampling_args
        self.num_batches_ahead = num_batches_ahead
        self.max_queue_size = max_queue_size or max(num_batches_ahead * 2, 4)
        self.generation_timeout = generation_timeout
        
        # Queues for communication
        self.request_queue = queue.Queue()
        self.result_queue = queue.Queue(maxsize=self.max_queue_size)
        
        # Tracking
        self.pending_batches = set()  # batch_ids currently being processed
        self.completed_batches = {}  # batch_id -> BatchResult
        self.next_expected_batch = 0
        self.generation_times = deque(maxlen=100)  # Track recent generation times
        
        # Thread management
        self.worker_thread = None
        self.stop_event = threading.Event()
        self.started = False
        
        # Synchronization
        self._lock = threading.Lock()
        
    def start(self):
        """Start the async generation worker thread"""
        if self.started:
            return
            
        self.worker_thread = threading.Thread(
            target=self._generation_worker,
            daemon=True,
            name="AsyncBatchGenerator"
        )
        self.worker_thread.start()
        self.started = True
        
    def stop(self):
        """Stop the async generation worker thread"""
        if not self.started:
            return
            
        self.stop_event.set()
        # Send poison pill
        self.request_queue.put(None)
        
        if self.worker_thread:
            self.worker_thread.join(timeout=10.0)
            
        self.started = False
        
    def submit_batch(self, request: BatchRequest) -> bool:
        """
        Submit a batch for async generation.
        
        Returns:
            bool: True if submitted successfully, False if queue is full
        """
        if not self.started:
            raise RuntimeError("AsyncBatchGenerator not started")
            
        with self._lock:
            if request.batch_id in self.pending_batches:
                return True  # Already submitted
                
            if len(self.pending_batches) >= self.max_queue_size:
                return False  # Queue full
                
            self.pending_batches.add(request.batch_id)
            
        self.request_queue.put(request)
        return True
        
    def get_batch(self, batch_id: int, timeout: Optional[float] = None) -> BatchResult:
        """
        Get a completed batch result. Blocks until the batch is ready.
        
        Args:
            batch_id: The batch ID to retrieve
            timeout: Maximum time to wait (uses generation_timeout if None)
            
        Returns:
            BatchResult: The completed batch result
            
        Raises:
            TimeoutError: If batch doesn't complete within timeout
            RuntimeError: If generation failed
        """
        timeout = timeout or self.generation_timeout
        start_time = time.time()
        
        while True:
            # Check if already completed
            with self._lock:
                if batch_id in self.completed_batches:
                    return self.completed_batches.pop(batch_id)
                    
            # Check for new results
            try:
                result = self.result_queue.get(timeout=0.1)
                with self._lock:
                    self.completed_batches[result.batch_id] = result
                    self.pending_batches.discard(result.batch_id)
                    
                # If this is our batch, return it
                if result.batch_id == batch_id:
                    with self._lock:
                        return self.completed_batches.pop(batch_id)
                        
            except queue.Empty:
                pass
                
            # Check timeout
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Batch {batch_id} generation timed out after {timeout}s")
                
    def get_pending_count(self) -> int:
        """Get number of batches currently being generated"""
        with self._lock:
            return len(self.pending_batches)
            
    def get_completed_count(self) -> int:
        """Get number of completed batches waiting to be retrieved"""
        with self._lock:
            return len(self.completed_batches)
            
    def get_average_generation_time(self) -> float:
        """Get average generation time for recent batches"""
        if not self.generation_times:
            return 0.0
        return sum(self.generation_times) / len(self.generation_times)
        
    def should_submit_more(self) -> bool:
        """Check if we should submit more batches for generation"""
        with self._lock:
            total_pending = len(self.pending_batches) + len(self.completed_batches)
            return total_pending < self.num_batches_ahead
            
    def _generation_worker(self):
        """Worker thread that processes generation requests"""
        while not self.stop_event.is_set():
            try:
                # Get next request
                request = self.request_queue.get(timeout=0.1)
                if request is None:  # Poison pill
                    break
                    
                # Generate batch
                start_time = time.time()
                result = self._generate_batch(request)
                generation_time = time.time() - start_time
                result.generation_time = generation_time
                self.generation_times.append(generation_time)
                        
                # Put result in queue
                self.result_queue.put(result)
                
            except queue.Empty:
                continue

    def _generate_batch(self, request: BatchRequest) -> BatchResult:
        """
        Generate a single batch. This runs in the worker thread.
        """
        # Call environment generation
        env_results = self.env.generate(
            request.env_inputs,
            client=self.client,
            model=self.model_name,
            sampling_args=self.sampling_args,
            score_rollouts=True,
            max_concurrent=request.max_concurrent,
        )
        
        # Extract all reward-related keys
        all_reward_dict = {}
        reward_keys = [k for k in env_results.keys() if k.endswith('_func') or k == 'reward']
        for key in reward_keys:
            all_reward_dict[key] = env_results[key]
        
        # Process results
        processed_results = self.env.process_env_results(
            env_results['prompt'],
            env_results['completion'],
            env_results['state'],
            env_results['reward'],
            processing_class=request.processing_class,
            mask_env_responses=request.mask_env_responses,
            max_completion_length=request.max_completion_length,
            mask_truncated_completions=request.mask_truncated_completions
        )
        
        return BatchResult(
            batch_id=request.batch_id,
            processed_results=processed_results,
            all_reward_dict=all_reward_dict,
            completions=env_results['completion'],
            prompts=env_results['prompt']
        ) 