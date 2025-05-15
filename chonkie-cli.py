#!/usr/bin/env python3
"""Command-line interface for Chonkie."""

import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.chonkie.cli import main

if __name__ == "__main__":
    sys.exit(main()) 