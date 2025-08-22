"""Legacy implementations of chunkers.

This module contains the original implementations of chunkers that have been 
superseded by improved versions. These are preserved for backward compatibility.

Example:
    To use the legacy SemanticChunker::
    
        from chonkie.legacy import SemanticChunker
        
    For the new improved version::
    
        from chonkie.chunker import SemanticChunker
"""

from .semantic import SemanticChunker

__all__ = ["SemanticChunker"]