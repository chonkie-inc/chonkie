"""CLI commands for Chonkie."""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional
import glob
import concurrent.futures

from ..chefs import (
    TextCleanerChef, 
    HTMLCleanerChef, 
    MarkdownCleanerChef, 
    PDFCleanerChef,
    JSONCleanerChef, 
    CSVCleanerChef
)
from ..chunker import RecursiveChunker, TokenChunker, SentenceChunker
from ..refinery import OverlapRefinery
from ..friends import JSONPorter
from ..utils import ChonkieLogger
from ..chomp import Chomp, ChompConfig, ParallelProcessor


def parse_args():
    """Parse command line arguments.
    
    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Chonkie CLI - Text processing and chunking")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Chunk command
    chunk_parser = subparsers.add_parser("chunk", help="Chunk text")
    chunk_parser.add_argument("input", help="Input file or directory")
    chunk_parser.add_argument("--output", "-o", help="Output file", default="chunks.json")
    chunk_parser.add_argument("--config", "-c", help="Configuration file")
    chunk_parser.add_argument(
        "--chunker", 
        choices=["recursive", "token", "sentence"], 
        default="recursive",
        help="Chunker to use"
    )
    chunk_parser.add_argument("--chunk-size", type=int, default=512, help="Chunk size")
    chunk_parser.add_argument(
        "--format", 
        choices=["auto", "text", "html", "markdown", "pdf", "json", "csv"],
        default="auto",
        help="Input format"
    )
    chunk_parser.add_argument(
        "--parallel", 
        action="store_true", 
        help="Enable parallel processing"
    )
    chunk_parser.add_argument(
        "--overlap", 
        type=int, 
        default=0, 
        help="Overlap size between chunks"
    )
    chunk_parser.add_argument("--log", help="Log file")
    chunk_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    # Create config command
    config_parser = subparsers.add_parser("config", help="Create or manage configurations")
    config_parser.add_argument(
        "--create", 
        action="store_true", 
        help="Create a new configuration"
    )
    config_parser.add_argument(
        "--output", 
        "-o", 
        help="Output configuration file",
        default="chonkie_config.json"
    )
    config_parser.add_argument(
        "--chunker", 
        choices=["recursive", "token", "sentence"], 
        default="recursive",
        help="Chunker to use"
    )
    config_parser.add_argument("--chunk-size", type=int, default=512, help="Chunk size")
    config_parser.add_argument(
        "--format", 
        choices=["auto", "text", "html", "markdown", "pdf", "json", "csv"],
        default="text",
        help="Input format"
    )
    
    return parser.parse_args()


def _get_chef_for_format(fmt: str):
    """Get the appropriate chef for the given format.
    
    Args:
        fmt: The format to get a chef for.
        
    Returns:
        An appropriate chef, or None if no chef is needed.
    """
    if fmt == "text" or fmt == "auto":
        return TextCleanerChef(normalize_whitespace=True)
    elif fmt == "html":
        return HTMLCleanerChef()
    elif fmt == "markdown":
        return MarkdownCleanerChef()
    elif fmt == "pdf":
        return PDFCleanerChef(extract_metadata=True, page_numbers=True)
    elif fmt == "json":
        return JSONCleanerChef(flatten=True)
    elif fmt == "csv":
        return CSVCleanerChef()
    else:
        return None


def _get_chunker(chunker_type: str, chunk_size: int):
    """Get the appropriate chunker.
    
    Args:
        chunker_type: The type of chunker to use.
        chunk_size: The chunk size to use.
        
    Returns:
        The configured chunker.
    """
    if chunker_type == "recursive":
        return RecursiveChunker(chunk_size=chunk_size)
    elif chunker_type == "token":
        return TokenChunker(chunk_size=chunk_size)
    elif chunker_type == "sentence":
        return SentenceChunker(chunk_size=chunk_size)
    else:
        # Default to recursive
        return RecursiveChunker(chunk_size=chunk_size)


def _detect_format(file_path: str):
    """Auto-detect the format of a file.
    
    Args:
        file_path: Path to the file.
        
    Returns:
        The detected format.
    """
    _, ext = os.path.splitext(file_path.lower())
    if ext == '.html' or ext == '.htm':
        return "html"
    elif ext == '.md' or ext == '.markdown':
        return "markdown"
    elif ext == '.pdf':
        return "pdf"
    elif ext == '.json':
        return "json"
    elif ext == '.csv':
        return "csv"
    else:
        return "text"


def _process_single_file(
    file_path: str, 
    pipeline: Chomp,
    logger: ChonkieLogger,
):
    """Process a single file.
    
    Args:
        file_path: Path to the file.
        pipeline: The configured pipeline.
        logger: The logger to use.
        
    Returns:
        A list of chunks.
    """
    logger.info(f"Processing file: {file_path}")
    logger.start_timer(f"process_{os.path.basename(file_path)}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        chunks = pipeline(content)
        
        # Make sure chunks is a list
        if chunks is None:
            logger.warning(f"No chunks generated from {file_path}")
            chunks = []
        else:
            logger.info(f"Generated {len(chunks)} chunks from {file_path}")
            
        logger.end_timer(f"process_{os.path.basename(file_path)}")
        
        return chunks
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}", exc_info=True)
        logger.end_timer(f"process_{os.path.basename(file_path)}")
        return []


def chunk_command(args):
    """Run the chunk command.
    
    Args:
        args: The parsed arguments.
    """
    # Set up logging
    log_level = ChonkieLogger.DEBUG if args.verbose else ChonkieLogger.INFO
    logger = ChonkieLogger(
        name="chonkie-cli", 
        level=log_level, 
        log_file=args.log
    )
    
    logger.info("Starting Chonkie chunking process")
    logger.start_timer("total")
    
    # Load configuration if provided
    if args.config:
        try:
            logger.info(f"Loading configuration from {args.config}")
            pipeline = Chomp.load_config(args.config)
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return 1
    else:
        # Create pipeline from command line arguments
        format_arg = args.format.lower()
        
        # Enable parallel processing if requested
        chomp_args = {}
        if args.parallel:
            chomp_args["enable_parallel"] = True
        
        pipeline = Chomp(**chomp_args)
        
        if format_arg != "auto":
            chef = _get_chef_for_format(format_arg)
            if chef and chef.is_available():
                pipeline.add_chef(chef)
            else:
                logger.warning(f"Chef for format '{format_arg}' is not available")
                
        # Add chunker
        chunker = _get_chunker(args.chunker, args.chunk_size)
        pipeline.set_chunker(chunker)
        
        # Add overlap refinery if requested
        if args.overlap > 0:
            pipeline.add_refinery(OverlapRefinery(context_size=args.overlap))
            
        # Build pipeline (we'll set the porter after processing to use the correct output file)
        pipeline.build()
        
    # Process input
    all_chunks = []
    if os.path.isfile(args.input):
        # Detect format if auto
        if args.format == "auto":
            detected_format = _detect_format(args.input)
            logger.info(f"Detected format: {detected_format}")
            
            if detected_format != "text":
                chef = _get_chef_for_format(detected_format)
                if chef and chef.is_available():
                    # Create a new pipeline with the detected format
                    new_pipeline = Chomp()
                    new_pipeline.add_chef(chef)
                    new_pipeline.set_chunker(pipeline.chunker)
                    
                    # Copy refineries, porter, etc.
                    for refinery in pipeline.refineries:
                        new_pipeline.add_refinery(refinery)
                    
                    # Copy handshake if present
                    if pipeline.handshake:
                        new_pipeline.set_handshake(pipeline.handshake)
                        
                    pipeline = new_pipeline.build()
                    
        # Process the single file
        chunks = _process_single_file(args.input, pipeline, logger)
        if chunks:
            all_chunks.extend(chunks)
            
    elif os.path.isdir(args.input):
        # Process all files in the directory
        logger.info(f"Processing directory: {args.input}")
        
        # Find all text files
        files = []
        for ext in ["*.txt", "*.md", "*.html", "*.htm", "*.pdf", "*.json", "*.csv"]:
            files.extend(glob.glob(os.path.join(args.input, ext)))
            
        logger.info(f"Found {len(files)} files to process")
        
        if args.parallel:
            # Process files in parallel
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = []
                for file_path in files:
                    # Create a separate pipeline for each file to avoid shared state issues
                    file_pipeline = Chomp.from_config(pipeline.to_config())
                    futures.append(
                        executor.submit(_process_single_file, file_path, file_pipeline, logger)
                    )
                
                for future in concurrent.futures.as_completed(futures):
                    try:
                        file_chunks = future.result()
                        all_chunks.extend(file_chunks)
                    except Exception as e:
                        logger.error(f"Error in parallel processing: {str(e)}")
        else:
            # Process files sequentially
            for file_path in files:
                chunks = _process_single_file(file_path, pipeline, logger)
                all_chunks.extend(chunks)
    else:
        logger.error(f"Input not found: {args.input}")
        return 1
        
    # Export chunks using JSONPorter
    if all_chunks:
        logger.info(f"Writing {len(all_chunks)} chunks to {args.output}")
        
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(output_dir, exist_ok=True)
        
        # Use JSONPorter to write the chunks to the output file
        porter = JSONPorter(lines=False)
        porter.export(all_chunks, file=args.output)
            
        logger.info(f"Output written to {args.output}")
    else:
        logger.warning("No chunks were generated")
        
    # Log performance summary
    metrics = logger.get_performance_summary()
    logger.end_timer("total")
    logger.info(f"Performance summary: {json.dumps(metrics, indent=2)}")
    
    return 0


def config_command(args):
    """Run the config command.
    
    Args:
        args: The parsed arguments.
    """
    if args.create:
        # Create a new configuration
        pipeline = Chomp()
        
        # Add chef based on format
        chef = _get_chef_for_format(args.format)
        if chef and chef.is_available():
            pipeline.add_chef(chef)
            
        # Add chunker
        chunker = _get_chunker(args.chunker, args.chunk_size)
        pipeline.set_chunker(chunker)
        
        # Build the pipeline
        pipeline.build()
        
        # Save the configuration
        pipeline.save_config(args.output)
        print(f"Configuration saved to {args.output}")
    else:
        # Show help if no subcommand provided
        print("Use --create to create a new configuration")
        
    return 0


def main():
    """Main entry point for the CLI."""
    args = parse_args()
    
    if args.command == "chunk":
        return chunk_command(args)
    elif args.command == "config":
        return config_command(args)
    else:
        print("No command specified. Use 'chunk' or 'config'.")
        return 1
    

if __name__ == "__main__":
    sys.exit(main()) 