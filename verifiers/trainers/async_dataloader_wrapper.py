from typing import Any, List
from torch.utils.data import DataLoader
from collections import deque
import threading


class AsyncDataLoaderWrapper:
    """
    Wraps a DataLoader to provide batch prefetching capabilities for async generation.
    
    This wrapper maintains a buffer of upcoming batches that can be accessed
    without advancing the main iterator, allowing async generation to work
    ahead while training continues on current batches.
    """
    
    def __init__(self, dataloader: DataLoader, buffer_size: int = 5):
        self.dataloader = dataloader
        self.buffer_size = buffer_size
        self._buffer = deque(maxlen=buffer_size)
        self._iterator = None
        self._lock = threading.Lock()
        self._exhausted = False
        self._current_epoch = 0
        self._current_batch = None  # Track the current batch
        
    def __iter__(self):
        """Reset and return iterator"""
        self._iterator = iter(self.dataloader)
        self._buffer.clear()
        self._exhausted = False
        self._current_batch = None
        return self
        
    def __next__(self):
        """Get next batch, refilling buffer as needed"""
        with self._lock:
            # If buffer is empty, try to fill it
            if not self._buffer and not self._exhausted:
                self._fill_buffer()
                
            if not self._buffer:
                raise StopIteration
                
            # Store current batch before returning
            self._current_batch = self._buffer.popleft()
            return self._current_batch
            
    def peek_ahead(self, n: int = 1) -> List[Any]:
        """
        Peek at the next n batches without consuming them.
        If n=0, returns the current batch (if available).
        Returns fewer batches if not enough are available.
        """
        with self._lock:
            if n == 0:
                # Return current batch if available
                return [self._current_batch] if self._current_batch is not None else []
            
            # Ensure buffer has enough items
            while len(self._buffer) < n and not self._exhausted:
                self._fill_buffer_single()
                
            # Return up to n items from buffer
            return list(self._buffer)[:n]
            
    def _fill_buffer(self):
        """Fill the buffer up to buffer_size"""
        while len(self._buffer) < self.buffer_size and not self._exhausted:
            self._fill_buffer_single()
            
    def _fill_buffer_single(self):
        """Add a single batch to the buffer"""
        if self._iterator is None:
            self._iterator = iter(self.dataloader)
            
        try:
            batch = next(self._iterator)
            self._buffer.append(batch)
        except StopIteration:
            self._exhausted = True
            
    def get_future_batches(self, start_offset: int, count: int) -> List[Any]:
        """
        Get future batches starting from start_offset positions ahead.
        This is used by async generation to get batches for future steps.
        
        Args:
            start_offset: How many batches ahead to start
            count: Number of batches to return
            
        Returns:
            List of batches (may be fewer than requested if not available)
        """
        with self._lock:
            # Ensure we have enough batches in buffer
            needed = start_offset + count
            while len(self._buffer) < needed and not self._exhausted:
                self._fill_buffer_single()
                
            # Extract the requested range
            result = []
            for i in range(start_offset, min(start_offset + count, len(self._buffer))):
                result.append(self._buffer[i])
                
            return result

    def __len__(self):
        """Return length of underlying dataloader if available"""
        return len(self.dataloader)
        
    @property
    def batch_size(self):
        """Return batch size of underlying dataloader"""
        return self.dataloader.batch_size
        
    @property
    def dataset(self):
        """Return dataset of underlying dataloader"""
        return self.dataloader.dataset 