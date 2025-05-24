#!/usr/bin/env python
"""Benchmark script to compare synchronous and asynchronous chunking performance."""

import asyncio
import time
from typing import List

import tiktoken
from tqdm import tqdm

from chonkie import TokenChunker


def generate_sample_texts(count: int = 100, size_range: tuple = (5000, 20000)) -> List[str]:
    """Generate sample texts for benchmarking.
    
    Args:
        count: Number of texts to generate
        size_range: Range of characters for each text
        
    Returns:
        List of generated text strings
    """
    import random
    import string
    
    texts = []
    
    for _ in range(count):
        length = random.randint(size_range[0], size_range[1])
        text = ''.join(random.choices(string.ascii_letters + ' ' * 10, k=length))
        texts.append(text)
        
    return texts


async def simulate_network_delay() -> None:
    """Simulate network delay that might occur in real-world applications."""
    await asyncio.sleep(0.05)  # 50ms delay


async def run_async_benchmark(chunker: TokenChunker, texts: List[str], with_delay: bool = False) -> float:
    """Run async benchmark.
    
    Args:
        chunker: Configured chunker
        texts: List of texts to process
        with_delay: Whether to simulate network delay
        
    Returns:
        Total processing time in seconds
    """
    start_time = time.time()
    
    if with_delay:
        # Custom implementation with delay simulation
        semaphore = asyncio.Semaphore(chunker._max_concurrency)
        
        async def process_with_delay(text):
            async with semaphore:
                chunks = await chunker.async_chunk(text)
                # Simulate network operation (e.g., storing chunks in a database)
                await simulate_network_delay()
                return chunks
        
        await asyncio.gather(*[process_with_delay(text) for text in texts])
    else:
        # Normal async batch processing
        await chunker.async_chunk_batch(texts, show_progress=False)
        
    end_time = time.time()
    return end_time - start_time


def run_sync_benchmark(chunker: TokenChunker, texts: List[str], with_delay: bool = False) -> float:
    """Run sync benchmark.
    
    Args:
        chunker: Configured chunker
        texts: List of texts to process
        with_delay: Whether to simulate network delay
        
    Returns:
        Total processing time in seconds
    """
    start_time = time.time()
    
    results = chunker(texts, batch_size=len(texts), show_progress_bar=False)
    
    if with_delay:
        # Simulate network operations in synchronous mode
        for chunks in results:
            # Sleep for the equivalent of the network delay
            time.sleep(0.05)  # 50ms delay
    
    end_time = time.time()
    return end_time - start_time


async def main():
    """Run benchmarks comparing sync vs async performance."""
    print("Generating test data...")
    texts = generate_sample_texts(count=50, size_range=(20000, 50000))
    print(f"Generated {len(texts)} texts")
    
    # Create chunker instance
    tokenizer = tiktoken.get_encoding("gpt2")
    chunker = TokenChunker(tokenizer=tokenizer, chunk_size=512, chunk_overlap=128)
    
    # Run sync benchmarks
    print("\nRunning synchronous benchmark (sequential)...")
    chunker.configure(use_multiprocessing=False)
    sync_sequential_time = run_sync_benchmark(chunker, texts)
    
    print("\nRunning synchronous benchmark (multiprocessing)...")
    chunker.configure(use_multiprocessing=True)
    sync_multiprocessing_time = run_sync_benchmark(chunker, texts)
    
    # Run sync benchmarks with network delay
    print("\nRunning synchronous benchmark with network delay (sequential)...")
    chunker.configure(use_multiprocessing=False)
    sync_sequential_time_with_delay = run_sync_benchmark(chunker, texts, with_delay=True)
    
    print("\nRunning synchronous benchmark with network delay (multiprocessing)...")
    chunker.configure(use_multiprocessing=True)
    sync_multiprocessing_time_with_delay = run_sync_benchmark(chunker, texts, with_delay=True)
    
    # Run async benchmarks with different concurrency levels
    concurrency_levels = [5, 10, 20]
    async_times = []
    async_times_with_delay = []
    
    for concurrency in concurrency_levels:
        print(f"\nRunning asynchronous benchmark (concurrency={concurrency})...")
        chunker.configure(max_concurrency=concurrency)
        async_time = await run_async_benchmark(chunker, texts)
        async_times.append(async_time)
        
        print(f"\nRunning asynchronous benchmark with network delay (concurrency={concurrency})...")
        async_time_with_delay = await run_async_benchmark(chunker, texts, with_delay=True)
        async_times_with_delay.append(async_time_with_delay)
    
    # Print results
    print("\n==== BENCHMARK RESULTS ====")
    print(f"Synchronous (sequential): {sync_sequential_time:.2f}s")
    print(f"Synchronous (multiprocessing): {sync_multiprocessing_time:.2f}s")
    print(f"Synchronous with delay (sequential): {sync_sequential_time_with_delay:.2f}s")
    print(f"Synchronous with delay (multiprocessing): {sync_multiprocessing_time_with_delay:.2f}s")
    
    for i, concurrency in enumerate(concurrency_levels):
        print(f"Asynchronous (concurrency={concurrency}): {async_times[i]:.2f}s")
        print(f"Asynchronous with delay (concurrency={concurrency}): {async_times_with_delay[i]:.2f}s")
        
    # Calculate improvements for regular processing
    best_async_time = min(async_times)
    best_sync_time = min(sync_sequential_time, sync_multiprocessing_time)
    improvement = (best_sync_time - best_async_time) / best_sync_time * 100
    
    # Calculate improvements for processing with network delay
    best_async_time_with_delay = min(async_times_with_delay)
    best_sync_time_with_delay = min(sync_sequential_time_with_delay, sync_multiprocessing_time_with_delay)
    improvement_with_delay = (best_sync_time_with_delay - best_async_time_with_delay) / best_sync_time_with_delay * 100
    
    print("\n==== PERFORMANCE IMPROVEMENT ====")
    print(f"Best synchronous: {best_sync_time:.2f}s")
    print(f"Best asynchronous: {best_async_time:.2f}s")
    print(f"Improvement: {improvement:.1f}%")
    
    print("\n==== PERFORMANCE IMPROVEMENT WITH NETWORK DELAY ====")
    print(f"Best synchronous with delay: {best_sync_time_with_delay:.2f}s")
    print(f"Best asynchronous with delay: {best_async_time_with_delay:.2f}s")
    print(f"Improvement with delay: {improvement_with_delay:.1f}%")


if __name__ == "__main__":
    asyncio.run(main()) 