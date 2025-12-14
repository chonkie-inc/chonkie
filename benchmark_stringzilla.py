#!/usr/bin/env python3
"""Benchmark RecursiveChunker with and without StringZilla optimizations."""

import time
import statistics
from typing import List, Callable
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from stringzilla import Str
    STRINGZILLA_AVAILABLE = True
except ImportError:
    STRINGZILLA_AVAILABLE = False
    print("WARNING: StringZilla not available, will only benchmark standard library")

from chonkie.chunker.recursive import RecursiveChunker
from chonkie.types import RecursiveRules, RecursiveLevel


def load_test_data() -> List[str]:
    """Load test data of various sizes."""
    # Small text (1KB)
    small_text = "This is a test sentence. " * 40  # ~1KB

    # Medium text (100KB) - simulate a small article
    medium_text = """
    The recursive chunking algorithm is designed to split large text documents into smaller,
    manageable pieces while preserving semantic boundaries. It operates by applying a series
    of splitting rules hierarchically, starting with document-level delimiters like double
    newlines, then paragraph-level delimiters, sentence boundaries, and finally word boundaries.

    Each level of the hierarchy attempts to split the text using specific delimiters. If a
    chunk is still too large after splitting at one level, the algorithm recursively applies
    the next level of rules. This ensures that chunks respect natural text boundaries whenever
    possible, leading to better results in downstream applications like semantic search,
    question answering, and text generation.

    Performance is critical in production environments where millions of documents need to be
    processed. String operations like split, replace, and join are called thousands of times
    during the chunking process, making them prime candidates for optimization.
    """ * 400  # ~100KB

    # Large text (1MB) - simulate a long document
    large_text = medium_text * 10  # ~1MB

    return [
        ("small_1kb", small_text),
        ("medium_100kb", medium_text),
        ("large_1mb", large_text),
    ]


def benchmark_function(func: Callable, *args, iterations: int = 100) -> dict:
    """Benchmark a function and return timing statistics."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args)
        end = time.perf_counter()
        times.append(end - start)

    return {
        'mean': statistics.mean(times),
        'median': statistics.median(times),
        'stdev': statistics.stdev(times) if len(times) > 1 else 0,
        'min': min(times),
        'max': max(times),
        'iterations': iterations,
    }


def split_text_standard(text: str, delimiters: List[str], sep: str = "✄") -> List[str]:
    """Standard Python implementation of text splitting with delimiters."""
    # Simulate the fallback implementation in RecursiveChunker
    for delimiter in delimiters:
        text = text.replace(delimiter, sep)

    splits = [split for split in text.split(sep) if split != ""]
    return splits


def split_text_stringzilla(text: str, delimiters: List[str], sep: str = "✄") -> List[str]:
    """StringZilla-optimized implementation of text splitting."""
    if not STRINGZILLA_AVAILABLE:
        return split_text_standard(text, delimiters, sep)

    sz_text = Str(text)

    # Use StringZilla's optimized replace
    for delimiter in delimiters:
        sz_text = Str(str(sz_text).replace(delimiter, sep))

    # Use StringZilla's optimized split
    splits = [str(s) for s in sz_text.split(sep) if s]
    return splits


def benchmark_split_operations():
    """Benchmark individual split operations."""
    print("\n" + "="*80)
    print("MICRO-BENCHMARK: Split Operations")
    print("="*80)

    test_data = load_test_data()
    delimiters = ["\n\n", "\n", ". ", ", "]

    results = {}

    for size_name, text in test_data:
        print(f"\n{size_name.upper()} ({len(text):,} bytes)")
        print("-" * 80)

        # Determine iterations based on text size
        if "small" in size_name:
            iterations = 1000
        elif "medium" in size_name:
            iterations = 100
        else:
            iterations = 10

        # Benchmark standard library
        std_result = benchmark_function(
            split_text_standard, text, delimiters,
            iterations=iterations
        )
        results[f"{size_name}_standard"] = std_result
        print(f"Standard Library:")
        print(f"  Mean: {std_result['mean']*1000:.3f}ms")
        print(f"  Median: {std_result['median']*1000:.3f}ms")
        print(f"  Min/Max: {std_result['min']*1000:.3f}ms / {std_result['max']*1000:.3f}ms")

        # Benchmark StringZilla
        if STRINGZILLA_AVAILABLE:
            sz_result = benchmark_function(
                split_text_stringzilla, text, delimiters,
                iterations=iterations
            )
            results[f"{size_name}_stringzilla"] = sz_result
            print(f"\nStringZilla:")
            print(f"  Mean: {sz_result['mean']*1000:.3f}ms")
            print(f"  Median: {sz_result['median']*1000:.3f}ms")
            print(f"  Min/Max: {sz_result['min']*1000:.3f}ms / {sz_result['max']*1000:.3f}ms")

            # Calculate speedup
            speedup = std_result['mean'] / sz_result['mean']
            print(f"\n  Speedup: {speedup:.2f}x")

            if speedup < 1.0:
                print(f"  (StringZilla is {1/speedup:.2f}x SLOWER)")

    return results


def benchmark_recursive_chunker():
    """Benchmark the full RecursiveChunker."""
    print("\n" + "="*80)
    print("END-TO-END BENCHMARK: RecursiveChunker")
    print("="*80)

    test_data = load_test_data()

    # Create rules for recursive chunking (similar to default recipe)
    rules = RecursiveRules(levels=[
        RecursiveLevel(delimiters=["\n\n"], include_delim="next"),
        RecursiveLevel(delimiters=["\n"], include_delim="next"),
        RecursiveLevel(delimiters=[". ", "! ", "? "], include_delim="prev"),
        RecursiveLevel(delimiters=[", ", "; "], include_delim="prev"),
        RecursiveLevel(whitespace=True),
    ])

    # Create chunker
    chunker = RecursiveChunker(
        tokenizer="character",
        chunk_size=512,
        rules=rules,
        min_characters_per_chunk=24,
    )

    results = {}

    for size_name, text in test_data:
        print(f"\n{size_name.upper()} ({len(text):,} bytes)")
        print("-" * 80)

        # Determine iterations
        if "small" in size_name:
            iterations = 100
        elif "medium" in size_name:
            iterations = 10
        else:
            iterations = 3

        # Benchmark chunking
        result = benchmark_function(
            chunker.chunk, text,
            iterations=iterations
        )
        results[size_name] = result

        # Get chunk count
        chunks = chunker.chunk(text)

        print(f"Chunks created: {len(chunks)}")
        print(f"Mean time: {result['mean']*1000:.3f}ms")
        print(f"Median time: {result['median']*1000:.3f}ms")
        print(f"Min/Max: {result['min']*1000:.3f}ms / {result['max']*1000:.3f}ms")
        print(f"Throughput: {len(text)/result['mean']/1024/1024:.2f} MB/s")

    return results


def check_cython_availability():
    """Check which Cython extensions are available."""
    print("\n" + "="*80)
    print("CYTHON EXTENSION AVAILABILITY")
    print("="*80)

    try:
        from chonkie.chunker.c_extensions.split import split_text
        print("✓ Split extension (Cython) is available")
        split_available = True
    except ImportError:
        print("✗ Split extension (Cython) is NOT available")
        split_available = False

    try:
        from chonkie.chunker.c_extensions.merge import _merge_splits
        print("✓ Merge extension (Cython) is available")
        merge_available = True
    except ImportError:
        print("✗ Merge extension (Cython) is NOT available")
        merge_available = False

    if STRINGZILLA_AVAILABLE:
        print("✓ StringZilla is available")
    else:
        print("✗ StringZilla is NOT available")

    return split_available, merge_available


def main():
    """Run all benchmarks."""
    print("="*80)
    print("StringZilla vs Standard Library Benchmark for RecursiveChunker")
    print("="*80)

    # Check availability
    split_avail, merge_avail = check_cython_availability()

    # Run micro-benchmarks
    split_results = benchmark_split_operations()

    # Run end-to-end benchmark
    e2e_results = benchmark_recursive_chunker()

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if STRINGZILLA_AVAILABLE:
        print("\nSplit Operation Speedups:")
        for size in ["small_1kb", "medium_100kb", "large_1mb"]:
            if f"{size}_standard" in split_results and f"{size}_stringzilla" in split_results:
                std_mean = split_results[f"{size}_standard"]['mean']
                sz_mean = split_results[f"{size}_stringzilla"]['mean']
                speedup = std_mean / sz_mean
                print(f"  {size}: {speedup:.2f}x")
    else:
        print("\nStringZilla not available - no comparison possible")
        print("Install with: pip install stringzilla")

    print("\nNote: These benchmarks test the Python fallback path.")
    print("The Cython extensions (when available) already provide significant optimization.")
    print("StringZilla could be useful when Cython is not available or as an alternative.")


if __name__ == "__main__":
    main()
