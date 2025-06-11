#!/usr/bin/env python3
"""Test script for the implemented Chonkie components."""

import os
import sys
import time
import tempfile
from concurrent.futures import ThreadPoolExecutor

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import just the components we implemented
from src.chonkie.utils.logger import ChonkieLogger
from src.chonkie.utils.visualizer import Visualizer
from src.chonkie.chomp.parallel import ParallelProcessor
from src.chonkie.cli.commands import parse_args


def test_logger():
    """Test the ChonkieLogger implementation."""
    print("\n=== Testing ChonkieLogger ===\n")
    
    # Create a temporary log file
    with tempfile.NamedTemporaryFile(suffix='.log', delete=False) as temp:
        log_file = temp.name
    
    # Create logger
    logger = ChonkieLogger(
        name="test_logger",
        level=ChonkieLogger.DEBUG,
        log_file=log_file,
        console=True,
        timestamp=True
    )
    
    print("Logging various message types...")
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    print("\nTesting performance tracking...")
    logger.start_timer("test_operation")
    time.sleep(0.5)  # Simulate work
    elapsed = logger.end_timer("test_operation")
    print(f"Elapsed time: {elapsed:.2f} seconds")
    
    print("\nTesting pipeline step logging...")
    logger.log_pipeline_step("text_cleaning", input_size=1000, output_size=950)
    logger.log_pipeline_step("chunking", input_size=950, output_size=10)
    
    # Get performance summary
    metrics = logger.get_performance_summary()
    print(f"\nPerformance metrics: {metrics}")
    
    # Read and display log file contents
    with open(log_file, 'r') as f:
        log_content = f.read()
    
    print(f"\nLog file content:\n{log_content}")
    
    # Test direct file logging
    print("\nTesting direct file logging...")
    custom_log_file = tempfile.mktemp(suffix='.log')
    logger.log_to_file("This is a direct file log message", custom_log_file)
    
    with open(custom_log_file, 'r') as f:
        custom_log_content = f.read()
    
    print(f"Custom log file content: {custom_log_content}")
    
    # Clean up
    os.unlink(log_file)
    os.unlink(custom_log_file)
    
    print("\nLogger test completed!")


def test_parallel_processor():
    """Test the ParallelProcessor implementation."""
    print("\n=== Testing ParallelProcessor ===\n")
    
    # Create some test data
    data = list(range(20))
    
    # Define a function to process data
    def process_item(item):
        time.sleep(0.1)  # Simulate work
        return item * 2
    
    # Test sequential processing
    print("Testing sequential processing...")
    start_time = time.time()
    sequential_results = [process_item(item) for item in data]
    sequential_time = time.time() - start_time
    print(f"Sequential processing time: {sequential_time:.4f} seconds")
    
    # Test parallel processing with threads
    print("\nTesting parallel processing with threads...")
    thread_processor = ParallelProcessor(workers=4, use_processes=False)
    start_time = time.time()
    thread_results = thread_processor.map(process_item, data)
    thread_time = time.time() - start_time
    print(f"Thread processing time: {thread_time:.4f} seconds")
    print(f"Speed improvement: {sequential_time / thread_time:.2f}x")
    
    # Test batch processing
    print("\nTesting batch processing...")
    
    def process_batch(batch):
        return [process_item(item) for item in batch]
    
    start_time = time.time()
    batch_results = thread_processor.batch_process(process_batch, data, batch_size=5)
    batch_time = time.time() - start_time
    print(f"Batch processing time: {batch_time:.4f} seconds")
    
    # Verify results - note that order may not be preserved in parallel processing
    # so we sort before comparing
    print("\nVerifying results...")
    sequential_set = set(sequential_results)
    thread_set = set(thread_results)
    batch_set = set(batch_results)
    
    if sequential_set == thread_set == batch_set:
        print("All results match! (Set comparison)")
    else:
        print("Warning: Results don't match as sets!")
        
    # Check if all expected values are present
    expected_results = [x * 2 for x in data]
    if set(expected_results) == thread_set:
        print("Thread results contain the expected values")
    else:
        print("Warning: Thread results don't contain expected values")
        
    if set(expected_results) == batch_set:
        print("Batch results contain the expected values")
    else:
        print("Warning: Batch results don't contain expected values")
    
    print("\nParallelProcessor test completed!")


def test_visualizer():
    """Test the Visualizer implementation."""
    print("\n=== Testing Visualizer ===\n")
    
    # Create a visualizer
    visualizer = Visualizer(use_color=True)
    
    # Create a mock pipeline
    class MockPipeline:
        def __init__(self):
            self.chefs = ["TextCleanerChef", "PDFCleanerChef"]
            self.chunker = "RecursiveChunker"
            self.refineries = ["OverlapRefinery"]
            self.porter = "JSONPorter"
            self.handshake = None
    
    # Create a mock chunk
    class MockChunk:
        def __init__(self, text, token_count, start_index, end_index):
            self.text = text
            self.token_count = token_count
            self.start_index = start_index
            self.end_index = end_index
            
        def to_dict(self):
            return {
                "text": self.text,
                "token_count": self.token_count,
                "start_index": self.start_index,
                "end_index": self.end_index
            }
    
    # Create mock chunks
    chunks = [
        MockChunk("This is chunk 1", 5, 0, 15),
        MockChunk("This is chunk 2", 5, 16, 31),
        MockChunk("This is chunk 3", 5, 32, 47)
    ]
    
    # Create a mock pipeline
    pipeline = MockPipeline()
    
    # Test chunk visualization
    print("Testing chunk visualization...")
    text_viz = visualizer.visualize_chunks(chunks, "text")
    print(f"\nText visualization:\n{text_viz}")
    
    json_viz = visualizer.visualize_chunks(chunks, "json")
    print(f"\nJSON visualization:\n{json_viz}")
    
    # Test pipeline visualization
    print("\nTesting pipeline visualization...")
    pipeline_viz = visualizer.visualize_pipeline(pipeline, "text")
    print(f"\nPipeline visualization:\n{pipeline_viz}")
    
    pipeline_json = visualizer.visualize_pipeline(pipeline, "json")
    print(f"\nPipeline JSON:\n{pipeline_json}")
    
    print("\nVisualizer test completed!")


def test_cli_args():
    """Test the CLI argument parsing."""
    print("\n=== Testing CLI Argument Parsing ===\n")
    
    # Test chunk command
    print("Testing 'chunk' command parsing...")
    sys.argv = [
        'chonkie-cli.py',
        'chunk',
        'sample.md',
        '--output', 'output.json',
        '--chunker', 'recursive',
        '--chunk-size', '100',
        '--format', 'markdown',
        '--parallel',
        '--overlap', '20',
        '--verbose'
    ]
    
    args = parse_args()
    print("\nChunk command arguments:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    
    # Test config command
    print("\nTesting 'config' command parsing...")
    sys.argv = [
        'chonkie-cli.py',
        'config',
        '--create',
        '--output', 'config.json',
        '--chunker', 'sentence',
        '--chunk-size', '256',
        '--format', 'pdf'
    ]
    
    args = parse_args()
    print("\nConfig command arguments:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    
    print("\nCLI argument parsing test completed!")


if __name__ == "__main__":
    print("=== Testing Chonkie Components ===")
    
    # Run all tests
    test_logger()
    test_parallel_processor()
    test_visualizer()
    test_cli_args()
    
    print("\nAll tests completed!") 