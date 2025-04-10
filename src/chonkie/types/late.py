"""Module containing the associated types for the LateChunker."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional

from .recursive import RecursiveChunk

if TYPE_CHECKING:
    import numpy as np


@dataclass
class LateChunk(RecursiveChunk):
    """Class to represent the late chunk.

    Attributes:
        text (str): The text of the chunk.
        start_index (int): The start index of the chunk.
        end_index (int): The end index of the chunk.
        token_count (int): The number of tokens in the chunk.
        start_token (int): The start token of the chunk.
        end_token (int): The end token of the chunk.
        sentences (list[LateSentence]): The sentences in the chunk.
        embedding (Optional[np.ndarray]): The embedding of the chunk.

    """

    embedding: Optional["np.ndarray"] = field(default=None)

    def to_dict(self) -> dict:
        """Return the LateChunk as a dictionary."""
        return {
            "text": self.text,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "token_count": self.token_count,
            "embedding": self.embedding.tolist()
            if self.embedding is not None
            else None,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "LateChunk":
        """Create a LateChunk from a dictionary."""
        return cls(**data)

    def __repr__(self) -> str:
        """Return a string representation of the LateChunk."""
        return (
            f"LateChunk(text={self.text}, "
            f"start_index={self.start_index}, "
            f"end_index={self.end_index}, "
            f"token_count={self.token_count}, "
            f"embedding={self.embedding})"
        )
