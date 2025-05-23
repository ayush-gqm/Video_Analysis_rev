"""
Memory profiling utilities for tracking memory usage in the video analysis pipeline.
Helps identify and debug memory leaks and optimize memory-intensive operations.
"""

import os
import time
import logging
import functools
import gc
from typing import Callable, Any

import psutil
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)

class MemoryTracker:
    """
    Track memory usage of the Python process and CUDA devices (if available).
    """
    
    def __init__(self, name: str = "default"):
        """
        Initialize a memory tracker.
        
        Args:
            name: Name for this tracker (used in logs)
        """
        self.name = name
        self.process = psutil.Process(os.getpid())
        self.cuda_available = TORCH_AVAILABLE and torch.cuda.is_available()
        self.timestamps = []
        self.cpu_memory = []
        self.gpu_memory = []
        
    def snapshot(self) -> dict:
        """
        Take a memory snapshot.
        
        Returns:
            Dictionary with memory usage information
        """
        # Collect garbage to get consistent readings
        gc.collect()
        
        # Get CPU memory info
        cpu_mem = self.process.memory_info().rss / (1024 * 1024)  # MB
        
        # Get GPU memory info if available
        gpu_mem = {}
        if self.cuda_available:
            for i in range(torch.cuda.device_count()):
                gpu_mem[i] = {
                    "allocated": torch.cuda.memory_allocated(i) / (1024 * 1024),  # MB
                    "reserved": torch.cuda.memory_reserved(i) / (1024 * 1024),  # MB
                    "max_allocated": torch.cuda.max_memory_allocated(i) / (1024 * 1024)  # MB
                }
            # Force sync to get accurate readings
            torch.cuda.synchronize()
        
        timestamp = time.time()
        
        # Store history
        self.timestamps.append(timestamp)
        self.cpu_memory.append(cpu_mem)
        self.gpu_memory.append(gpu_mem if self.cuda_available else {})
        
        return {
            "timestamp": timestamp,
            "cpu_mb": cpu_mem,
            "gpu_mb": gpu_mem
        }
    
    def clear_history(self):
        """Clear the recorded history."""
        self.timestamps = []
        self.cpu_memory = []
        self.gpu_memory = []
    
    def log_summary(self):
        """Log a summary of memory usage."""
        if not self.timestamps:
            logger.info(f"[{self.name}] No memory data recorded")
            return
        
        cpu_max = max(self.cpu_memory)
        cpu_min = min(self.cpu_memory)
        cpu_avg = sum(self.cpu_memory) / len(self.cpu_memory)
        
        logger.info(f"[{self.name}] Memory summary:")
        logger.info(f"  CPU: max={cpu_max:.1f} MB, min={cpu_min:.1f} MB, avg={cpu_avg:.1f} MB")
        
        if self.cuda_available and self.gpu_memory:
            # Extract for each GPU device
            for device in range(torch.cuda.device_count()):
                if self.gpu_memory and device in self.gpu_memory[0]:
                    allocated = [snapshot.get(device, {}).get("allocated", 0) for snapshot in self.gpu_memory]
                    reserved = [snapshot.get(device, {}).get("reserved", 0) for snapshot in self.gpu_memory]
                    max_allocated = max([snapshot.get(device, {}).get("max_allocated", 0) for snapshot in self.gpu_memory])
                    
                    logger.info(f"  GPU {device}:")
                    logger.info(f"    Allocated: max={max(allocated):.1f} MB, avg={sum(allocated)/len(allocated):.1f} MB")
                    logger.info(f"    Reserved: max={max(reserved):.1f} MB, avg={sum(reserved)/len(reserved):.1f} MB")
                    logger.info(f"    Peak allocated: {max_allocated:.1f} MB")
    
    def reset_peak_stats(self):
        """Reset peak memory statistics for CUDA devices."""
        if self.cuda_available:
            try:
                # Get the current device
                current_device = torch.cuda.current_device()
                
                # Only reset stats for the current device to avoid invalid device errors
                torch.cuda.reset_peak_memory_stats(current_device)
                logger.debug(f"Reset peak memory stats for CUDA device {current_device}")
            except RuntimeError as e:
                logger.warning(f"Could not reset CUDA peak memory stats: {str(e)}")
                # Continue without resetting stats

def track_memory(function_name: str = None, log_interval: int = 1, final_summary: bool = True, reset_cuda_stats: bool = True):
    """
    Decorator to track memory usage during function execution.
    
    Args:
        function_name: Optional name override for the function (default: function.__name__)
        log_interval: Log memory usage every N seconds
        final_summary: Whether to log a summary at the end
        reset_cuda_stats: Whether to reset CUDA memory stats before tracking
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            name = function_name or func.__name__
            tracker = MemoryTracker(name)
            
            if reset_cuda_stats and tracker.cuda_available:
                tracker.reset_peak_stats()
            
            logger.info(f"[{name}] Starting memory tracking")
            tracker.snapshot()  # Initial snapshot
            
            # Create a logging thread that takes snapshots at intervals
            if log_interval > 0:
                import threading
                import time
                
                stop_thread = False
                
                def logging_thread():
                    last_log = time.time()
                    while not stop_thread:
                        current_time = time.time()
                        if current_time - last_log >= log_interval:
                            snapshot = tracker.snapshot()
                            logger.info(f"[{name}] Memory: CPU {snapshot['cpu_mb']:.1f} MB, " + 
                                      (f"GPU {sum(dev['allocated'] for dev in snapshot['gpu_mb'].values()):.1f} MB" 
                                       if tracker.cuda_available else "No GPU"))
                            last_log = current_time
                        time.sleep(0.1)  # Sleep to avoid busy waiting
                
                thread = threading.Thread(target=logging_thread)
                thread.daemon = True
                thread.start()
            
            try:
                # Execute the function
                start_time = time.time()
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Take a final snapshot
                tracker.snapshot()
                
                if final_summary:
                    logger.info(f"[{name}] Finished in {duration:.2f}s")
                    tracker.log_summary()
                
                return result
            finally:
                if log_interval > 0:
                    stop_thread = True
                    # Wait a bit for the thread to finish
                    time.sleep(0.2)
        
        return wrapper
    
    # Handle the case where the decorator is used without arguments
    if callable(function_name):
        func = function_name
        function_name = func.__name__
        return decorator(func)
    
    return decorator

def get_memory_usage() -> dict:
    """
    Get current memory usage.
    
    Returns:
        Dictionary with memory usage information
    """
    tracker = MemoryTracker("current")
    return tracker.snapshot()

def log_memory_usage(label: str = "Current memory usage"):
    """
    Log current memory usage.
    
    Args:
        label: Label to include in the log message
    """
    snapshot = get_memory_usage()
    logger.info(f"{label}: CPU {snapshot['cpu_mb']:.1f} MB, " + 
              (f"GPU {sum(dev['allocated'] for dev in snapshot['gpu_mb'].values()):.1f} MB" 
               if snapshot['gpu_mb'] else "No GPU"))

def optimize_batch_size(starting_batch_size: int = 32, min_batch_size: int = 4) -> int:
    """
    Optimize batch size based on available memory.
    
    Args:
        starting_batch_size: Initial batch size to try
        min_batch_size: Minimum batch size to consider
        
    Returns:
        Optimized batch size
    """
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return starting_batch_size
    
    # Get current free memory
    current_snapshot = get_memory_usage()
    total_gpu_memory = sum(dev.get('reserved', 0) for dev in current_snapshot['gpu_mb'].values())
    
    # Estimate free memory (roughly)
    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory / (1024 * 1024)  # MB
    free_memory = total_memory - total_gpu_memory
    
    # Scale batch size based on available memory
    if free_memory < 1000:  # Less than 1 GB free
        return max(min_batch_size, starting_batch_size // 4)
    elif free_memory < 2000:  # Less than 2 GB free
        return max(min_batch_size, starting_batch_size // 2) 
    elif free_memory < 4000:  # Less than 4 GB free
        return max(min_batch_size, int(starting_batch_size * 0.75))
    else:
        return starting_batch_size  # Plenty of memory, use the starting size

def cleanup_memory():
    """
    Aggressively clean up memory by forcing garbage collection and clearing CUDA cache.
    """
    # Force garbage collection
    gc.collect()
    
    # Clear CUDA cache if available
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # Log current memory usage after cleanup
    log_memory_usage("Memory after cleanup")

if __name__ == "__main__":
    # Simple test when this module is run directly
    logging.basicConfig(level=logging.INFO)
    
    @track_memory
    def test_function():
        # Create a large array to use memory
        data = np.zeros((1000, 1000, 10), dtype=np.float32)
        time.sleep(2)
        
        # Create more data
        more_data = [np.random.rand(500, 500) for _ in range(10)]
        time.sleep(2)
        
        # Clean up
        del data
        del more_data
        gc.collect()
        time.sleep(1)
        
        return "Done"
    
    log_memory_usage("Before test")
    test_function()
    log_memory_usage("After test")
    
    # Test batch size optimization
    optimal_batch_size = optimize_batch_size(32)
    logger.info(f"Optimal batch size: {optimal_batch_size}") 