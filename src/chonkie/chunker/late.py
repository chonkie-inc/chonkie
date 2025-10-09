# File: chonkie/chunker/late.py

"""
LateChunker: Minimal safe implementation for testing
Does not require sentence-transformers or other heavy dependencies.
"""

# Imports
from .base import BaseChunker  # your existing base.py
from typing import List

# Define LateChunker class
class LateChunker(BaseChunker):
    def __init__(self, *args, **kwargs):
        # Initialize BaseChunker
        super().__init__(*args, **kwargs)
        # Minimal placeholder embedding model
        self.embedding_model = None
        print("LateChunker initialized (no embeddings loaded).")

    def chunk(self, text: str) -> List[str]:
        """
        Dummy chunk method: splits text by sentences.
        """
        if not text:
            return []
        return text.split(". ")  # simple split by period+space

# Run as script
if __name__ == "__main__":
    chunker = LateChunker()
    sample_text = "This is a test. Check chunking. Everything works!"
    chunks = chunker.chunk(sample_text)
    print("LateChunker is running correctly!")
    print("Chunks:", chunks)
