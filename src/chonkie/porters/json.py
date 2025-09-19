"""JSONPorter to convert Chunks into JSON format for storage."""

import json

from chonkie.types import Chunk

from .base import BasePorter

# NOTE: Since the JSON porter is just a simple function, it doesn't need much init 
# right now, except the ability to load in lines or not. 


class JSONPorter(BasePorter):
    """Porter to convert Chunks into JSON format for storage."""

    def __init__(self, 
                 lines: bool = True):
        """Initialize the JSONPorter."""
        super().__init__()
        self.lines = lines

        # Setting the default indent to 4, as it's the most common.
        self.indent = 4

    def _export_lines(self,
                     chunks: list[Chunk],
                     file: str = "chunks.jsonl") -> None:
        """Export the Chunks as a JSONL file."""
        with open(file, "w") as f:
            for chunk in chunks: 
                f.write(json.dumps(chunk.to_dict()) + "\n")

    def _export_json(self,
                     chunks: list[Chunk],
                     file: str = "chunks.json") -> None:
        """Export the Chunks into a JSON string."""
        with open(file, "w") as f:
            json.dump([chunk.to_dict() for chunk in chunks], f, indent=self.indent)

    def export(self, chunks: list[Chunk], file: str = "chunks.jsonl") -> None: # type: ignore[override]
        """Export the Chunks into a JSON string.
        
        Args:
            chunks: The chunks to export.
            file: The file to export the chunks to.

        """
        if self.lines:
            self._export_lines(chunks, file)
        else:
            self._export_json(chunks, file)
    
    def __call__(self, chunks: list[Chunk], file: str = "chunks.jsonl") -> None: # type: ignore[override]
        """Export the Chunks into a JSON string.
        
        Args:
            chunks: The chunks to export.
            file: The file to export the chunks to.

        """
        self.export(chunks, file)
