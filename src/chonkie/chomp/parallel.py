"""Parallel processing capabilities for the Chomp pipeline."""

import multiprocessing
import os
import concurrent.futures
from typing import Any, Callable, Dict, List, Optional, Union, Tuple, Iterable

from chonkie.types import Chunk

# Determine the default number of workers based on CPU cores
DEFAULT_WORKERS = min(32, (os.cpu_count() or 4))


class ParallelProcessor:
    """Helper class for parallel processing in the Chomp pipeline."""
    
    def __init__(
        self, 
        workers: Optional[int] = None,
        use_processes: bool = True,
        chunk_size: int = 1,
        timeout: Optional[float] = None
    ):
        """Initialize the parallel processor.
        
        Args:
            workers: Number of worker processes/threads to use. 
                    If None, uses the number of CPU cores.
            use_processes: Whether to use processes (True) or threads (False).
                          Processes are better for CPU-bound tasks, 
                          threads for I/O-bound tasks.
            chunk_size: Chunk size for parallel processing of iterables.
            timeout: Timeout for parallel processing in seconds.
                    If None, no timeout is applied.
        """
        self.workers = workers or DEFAULT_WORKERS
        self.use_processes = use_processes
        self.chunk_size = chunk_size
        self.timeout = timeout
        
    def map(self, func: Callable, items: Iterable[Any]) -> List[Any]:
        """Process items in parallel.
        
        Args:
            func: Function to apply to each item.
            items: Iterable of items to process.
            
        Returns:
            List of results from applying the function to each item.
        """
        executor_class = (
            concurrent.futures.ProcessPoolExecutor if self.use_processes 
            else concurrent.futures.ThreadPoolExecutor
        )
        
        with executor_class(max_workers=self.workers) as executor:
            # For a small number of items, submit all at once
            if hasattr(items, "__len__") and len(items) <= self.workers * 2:
                futures = [executor.submit(func, item) for item in items]
                results = []
                
                for future in concurrent.futures.as_completed(futures, timeout=self.timeout):
                    try:
                        results.append(future.result())
                    except Exception as e:
                        # Collect any exceptions that occurred
                        results.append(e)
                        
                return results
            else:
                # For larger iterables, use map with chunk_size
                try:
                    return list(executor.map(
                        func, items, 
                        chunksize=self.chunk_size, 
                        timeout=self.timeout
                    ))
                except concurrent.futures.TimeoutError:
                    raise TimeoutError(
                        f"Parallel processing timed out after {self.timeout} seconds"
                    )
    
    def process_chunks(
        self, 
        func: Callable[[Chunk], Chunk], 
        chunks: List[Chunk]
    ) -> List[Chunk]:
        """Process a list of chunks in parallel.
        
        Args:
            func: Function to apply to each chunk.
            chunks: List of chunks to process.
            
        Returns:
            List of processed chunks.
        """
        return self.map(func, chunks)
    
    def batch_process(
        self, 
        func: Callable[[List[Any]], List[Any]], 
        items: List[Any], 
        batch_size: int
    ) -> List[Any]:
        """Process items in batches, with each batch processed in parallel.
        
        Args:
            func: Function to apply to each batch.
            items: List of items to process.
            batch_size: Size of each batch.
            
        Returns:
            List of processed items, aggregated from all batches.
        """
        # Split items into batches
        batches = [
            items[i:i + batch_size] 
            for i in range(0, len(items), batch_size)
        ]
        
        # Process each batch in parallel
        batch_results = self.map(func, batches)
        
        # Flatten the results
        results = []
        for batch_result in batch_results:
            if isinstance(batch_result, list):
                results.extend(batch_result)
            else:
                results.append(batch_result)
                
        return results
    
    def __repr__(self) -> str:
        """Return the string representation of the parallel processor."""
        return (
            f"ParallelProcessor(workers={self.workers}, "
            f"use_processes={self.use_processes}, "
            f"chunk_size={self.chunk_size}, timeout={self.timeout})"
        ) 