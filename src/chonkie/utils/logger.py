"""Logging utility for Chonkie."""

import logging
import os
import sys
import time
from typing import Any, Dict, Optional, Union
from datetime import datetime


class ChonkieLogger:
    """Logger utility for Chonkie.
    
    This class provides a simple logger for Chonkie with configurable output formats,
    levels, and destinations (file or console).
    """
    
    # Log levels
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    
    def __init__(
        self,
        name: str = "chonkie",
        level: int = logging.INFO,
        log_file: Optional[str] = None,
        console: bool = True,
        log_format: Optional[str] = None,
        timestamp: bool = True
    ):
        """Initialize the logger.
        
        Args:
            name: Name of the logger.
            level: Log level (DEBUG, INFO, WARNING, ERROR).
            log_file: Path to the log file. If None, logs to console only.
            console: Whether to log to console.
            log_format: Custom log format string. If None, uses default format.
            timestamp: Whether to include timestamps in log messages.
        """
        self.name = name
        self.level = level
        self.log_file = log_file
        self.console = console
        self.timestamp = timestamp
        
        # Create logger
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        self._logger.handlers = []  # Clear any existing handlers
        
        # Set log format
        if log_format is None:
            if timestamp:
                self.log_format = "[%(asctime)s] %(levelname)s: %(message)s"
            else:
                self.log_format = "%(levelname)s: %(message)s"
        else:
            self.log_format = log_format
            
        formatter = logging.Formatter(self.log_format)
        
        # Add console handler if requested
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self._logger.addHandler(console_handler)
            
        # Add file handler if log_file is specified
        if log_file:
            # Create directory if it doesn't exist
            log_dir = os.path.dirname(os.path.abspath(log_file))
            os.makedirs(log_dir, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)
            
        # Performance tracking
        self.start_time = None
        self.performance_metrics = {}
            
    def debug(self, message: str) -> None:
        """Log a debug message.
        
        Args:
            message: The message to log.
        """
        self._logger.debug(message)
        
    def info(self, message: str) -> None:
        """Log an info message.
        
        Args:
            message: The message to log.
        """
        self._logger.info(message)
        
    def warning(self, message: str) -> None:
        """Log a warning message.
        
        Args:
            message: The message to log.
        """
        self._logger.warning(message)
        
    def error(self, message: str, exc_info: bool = False) -> None:
        """Log an error message.
        
        Args:
            message: The message to log.
            exc_info: Whether to include exception information.
        """
        self._logger.error(message, exc_info=exc_info)
        
    def start_timer(self, activity: str = "task") -> None:
        """Start a performance timer.
        
        Args:
            activity: Name of the activity being timed.
        """
        self.start_time = time.time()
        self.info(f"Started {activity}")
        
    def end_timer(self, activity: str = "task") -> float:
        """End a performance timer and log the elapsed time.
        
        Args:
            activity: Name of the activity being timed.
            
        Returns:
            Elapsed time in seconds.
        """
        if self.start_time is None:
            self.warning("Timer ended without being started")
            return 0.0
            
        elapsed = time.time() - self.start_time
        self.info(f"{activity} completed in {elapsed:.2f} seconds")
        
        # Store in performance metrics
        self.performance_metrics[activity] = elapsed
        
        return elapsed
        
    def log_pipeline_step(
        self, 
        step: str, 
        input_size: Optional[int] = None, 
        output_size: Optional[int] = None, 
        status: str = "success"
    ) -> None:
        """Log a pipeline step.
        
        Args:
            step: Name of the pipeline step.
            input_size: Size of the input (e.g., number of tokens).
            output_size: Size of the output (e.g., number of chunks).
            status: Status of the step (e.g., "success", "error").
        """
        message = f"Pipeline step '{step}' - status: {status}"
        if input_size is not None:
            message += f", input size: {input_size}"
        if output_size is not None:
            message += f", output size: {output_size}"
            
        if status.lower() == "error":
            self.error(message)
        else:
            self.info(message)
            
    def get_performance_summary(self) -> Dict[str, float]:
        """Get a summary of performance metrics.
        
        Returns:
            Dictionary of activity names and their elapsed times.
        """
        return self.performance_metrics
    
    def log_to_file(self, message: str, filename: str) -> None:
        """Log a message to a specific file.
        
        Args:
            message: The message to log.
            filename: The file to log to.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(os.path.abspath(filename))
        os.makedirs(log_dir, exist_ok=True)
        
        with open(filename, "a") as f:
            if self.timestamp:
                f.write(f"[{timestamp}] {message}\n")
            else:
                f.write(f"{message}\n") 