#!/usr/bin/env python3
"""Test script for Chonkie components."""

import os
import sys
import tempfile
import time
from pathlib import Path

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.chonkie import (
    Chomp,
    PDFCleanerChef,
    TextCleanerChef,
    RecursiveChunker,
    ChonkieLogger,
    ParallelProcessor,
    JSONPorter,
)
from src.chonkie.cli import main as cli_main


def test_pdf_processing():
    """Test PDF processing functionality."""
    print("\n=== Testing PDF Processing ===")
    
    # Check if a PDF file exists for testing
    pdf_files = list(Path('.').glob('**/*.pdf'))
    if not pdf_files:
        print("No PDF files found for testing. Skipping PDF test.")
        return
    
    pdf_file = str(pdf_files[0])
    print(f"Testing with PDF file: {pdf_file}")
    
    # Create PDF processing pipeline
    pdf_cleaner = PDFCleanerChef(
        extract_metadata=True,
        page_numbers=True,
        handle_tables=True
    )
    
    # Basic text cleaner and chunker
    text_cleaner = TextCleanerChef(normalize_whitespace=True)
    chunker = RecursiveChunker(chunk_size=200)
    
    # Create the pipeline
    pipeline = (
        Chomp()
        .add_chef(pdf_cleaner)
        .add_chef(text_cleaner)
        .set_chunker(chunker)
        .build()
    )
    
    # Process the PDF
    chunks = pipeline(pdf_file)
    
    # Print results
    print(f"Generated {len(chunks)} chunks from the PDF")
    if chunks:
        print(f"First chunk: {chunks[0].text[:100]}...")


def test_parallel_processing():
    """Test parallel processing capabilities."""
    print("\n=== Testing Parallel Processing ===")
    
    # Create test data
    texts = [
        f"This is test document {i} with some content for processing in parallel."
        for i in range(10)
    ]
    
    # Create a basic pipeline
    text_cleaner = TextCleanerChef(normalize_whitespace=True)
    chunker = RecursiveChunker(chunk_size=5)
    
    # Test sequential processing
    start_time = time.time()
    sequential_pipeline = (
        Chomp()
        .add_chef(text_cleaner)
        .set_chunker(chunker)
        .build()
    )
    
    sequential_results = [sequential_pipeline(text) for text in texts]
    sequential_time = time.time() - start_time
    print(f"Sequential processing time: {sequential_time:.4f} seconds")
    
    # Test parallel processing
    start_time = time.time()
    parallel_pipeline = (
        Chomp(enable_parallel=True)
        .add_chef(text_cleaner)
        .set_chunker(chunker)
        .enable_parallel_processing(workers=4)
        .build()
    )
    
    # Use the batch processing capability
    parallel_processor = ParallelProcessor(workers=4)
    parallel_results = parallel_processor.batch_process(
        lambda batch: [parallel_pipeline(text) for text in batch],
        texts,
        batch_size=3
    )
    
    parallel_time = time.time() - start_time
    print(f"Parallel processing time: {parallel_time:.4f} seconds")
    print(f"Speed improvement: {sequential_time / parallel_time:.2f}x")


def test_logging():
    """Test logging and performance monitoring."""
    print("\n=== Testing Logging and Performance Monitoring ===")
    
    # Create a temporary log file
    with tempfile.NamedTemporaryFile(suffix='.log', delete=False) as temp:
        log_file = temp.name
    
    # Create logger
    logger = ChonkieLogger(
        name="test_logger",
        level=ChonkieLogger.DEBUG,
        log_file=log_file,
        console=True
    )
    
    # Log various message types
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test performance tracking
    logger.start_timer("test_operation")
    time.sleep(0.5)  # Simulate work
    elapsed = logger.end_timer("test_operation")
    
    # Test pipeline step logging
    logger.log_pipeline_step("text_cleaning", input_size=1000, output_size=950)
    logger.log_pipeline_step("chunking", input_size=950, output_size=10)
    
    # Get performance summary
    metrics = logger.get_performance_summary()
    print(f"Performance metrics: {metrics}")
    
    # Read and display log file contents
    with open(log_file, 'r') as f:
        log_content = f.read()
    
    print(f"Log file content sample:\n{log_content[:300]}...")
    
    # Clean up
    os.unlink(log_file)


def test_cli():
    """Test the command-line interface."""
    print("\n=== Testing Command Line Interface ===")
    
    # Create a test file
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp:
        test_file = temp.name
        temp.write(b"This is a test file for the Chonkie CLI.\n" * 20)
    
    # Create a test output file
    output_file = test_file + '.json'
    
    # Build the command-line arguments
    sys.argv = [
        'chonkie-cli.py',
        'chunk',
        test_file,
        '--output', output_file,
        '--chunker', 'recursive',
        '--chunk-size', '100',
        '--format', 'text',
        '--verbose'
    ]
    
    # Run the CLI (don't actually exit)
    try:
        old_exit = sys.exit
        sys.exit = lambda code: None
        cli_main()
    finally:
        sys.exit = old_exit
    
    # Check if output file was created
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file)
        print(f"Output file created: {output_file} ({file_size} bytes)")
        
        # Read the first part of the file
        with open(output_file, 'r') as f:
            content = f.read(200)
        print(f"Output file content sample:\n{content}...")
        
        # Clean up
        os.unlink(output_file)
    else:
        print(f"Error: Output file not created: {output_file}")
    
    # Clean up the input file
    os.unlink(test_file)


def test_config_generation():
    """Test configuration generation and loading."""
    print("\n=== Testing Configuration Generation ===")
    
    # Create a test config file
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp:
        config_file = temp.name
    
    # Build the command-line arguments
    sys.argv = [
        'chonkie-cli.py',
        'config',
        '--create',
        '--output', config_file,
        '--chunker', 'recursive',
        '--chunk-size', '256',
        '--format', 'text'
    ]
    
    # Run the CLI (don't actually exit)
    try:
        old_exit = sys.exit
        sys.exit = lambda code: None
        cli_main()
    finally:
        sys.exit = old_exit
    
    # Check if config file was created
    if os.path.exists(config_file):
        file_size = os.path.getsize(config_file)
        print(f"Config file created: {config_file} ({file_size} bytes)")
        
        # Read the file
        with open(config_file, 'r') as f:
            content = f.read()
        print(f"Config file content sample:\n{content[:200]}...")
        
        # Try loading the config
        try:
            pipeline = Chomp.load_config(config_file)
            print(f"Successfully loaded pipeline from config: {pipeline}")
        except Exception as e:
            print(f"Error loading config: {e}")
        
        # Clean up
        os.unlink(config_file)
    else:
        print(f"Error: Config file not created: {config_file}")


if __name__ == "__main__":
    print("=== Chonkie Component Tests ===")
    
    # Run all tests
    test_pdf_processing()
    test_parallel_processing()
    test_logging()
    test_cli()
    test_config_generation()
    
    print("\nAll tests completed!") 