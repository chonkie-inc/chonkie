"""Legacy implementations of chunkers.

This module contains the original implementations of chunkers that have been 
superseded by improved versions. These are preserved for backward compatibility.

Example:
    To use the legacy chunkers::
    
        from chonkie.legacy import SemanticChunker, SDPMChunker
        
    For the new improved version::
    
        from chonkie.chunker import SemanticChunker
        
Note:
    The SDPMChunker's merging capabilities are now integrated into the new
    SemanticChunker implementation.
"""

from .sdpm import SDPMChunker
from .semantic import SemanticChunker

__all__ = ["SemanticChunker", "SDPMChunker"]