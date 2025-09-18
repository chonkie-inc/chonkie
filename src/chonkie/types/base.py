"""Custom base types for Chonkie."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator, List, Optional, Union

if TYPE_CHECKING:
    import numpy as np


@dataclass
class Context:
    """Context class to hold chunk metadata.

    Attributes:
        text (str): The text of the chunk.
        start_index (Optional[int]): The starting index of the chunk in the original text.
        end_index (Optional[int]): The ending index of the chunk in the original text.
        token_count (int): The number of tokens in the chunk.

    """

    text: str
    token_count: int
    start_index: Optional[int] = None
    end_index: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate context attributes."""
        if not isinstance(self.text, str):
            raise ValueError("Text must be a string.")
        if self.token_count is not None and self.token_count < 0:
            raise ValueError("Token count must be a non-negative integer.")
        if self.start_index is not None and self.start_index < 0:
            raise ValueError("Start index must be a non-negative integer.")
        if self.end_index is not None and self.end_index < 0:
            raise ValueError("End index must be a non-negative integer.")
        if (
            self.start_index is not None
            and self.end_index is not None
            and (self.start_index > self.end_index)
        ):
            raise ValueError("Start index must be less than end index.")

    def __len__(self) -> int:
        """Return the length of the text."""
        return len(self.text)

    def __str__(self) -> str:
        """Return a string representation of the Context."""
        return self.text

    def __repr__(self) -> str:
        """Return a detailed string representation of the Context."""
        return (
            f"Context(text='{self.text}', token_count={self.token_count}, "
            f"start_index={self.start_index}, end_index={self.end_index})"
        )

    def to_dict(self) -> dict:
        """Return the Context as a dictionary."""
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, data: dict) -> "Context":
        """Create a Context object from a dictionary."""
        return cls(**data)


@dataclass
class Chunk:
    """Chunks with metadata.

    Attributes:
        text (str): The text of the chunk.
        start_index (int): The starting index of the chunk in the original text.
        end_index (int): The ending index of the chunk in the original text.
        token_count (int): The number of tokens in the chunk.
        context (Optional[Context]): Optional context metadata for the chunk.
        embedding (Union[List[float], np.ndarray, None]): Optional embedding vector for the chunk,
            either as a list of floats or a numpy array.

    """

    text: str
    start_index: int
    end_index: int
    token_count: int
    context: Optional[Context] = None
    embedding: Union[List[float], "np.ndarray", None] = field(default=None)

    def __len__(self) -> int:
        """Return the length of the text."""
        return len(self.text)

    def __str__(self) -> str:
        """Return a string representation of the Chunk."""
        return self.text

    def _preview_embedding(self) -> str:
        """Create a preview string for the embedding.

        Shows first 3 and last 2 values for long embeddings,
        or full embedding for short ones.

        Returns:
            A string preview of the embedding

        """
        if self.embedding is None:
            return ""

        try:
            # Check if it's array-like with length
            if hasattr(self.embedding, '__len__') and hasattr(self.embedding, '__getitem__'):
                emb_len = len(self.embedding)
                if emb_len > 5:
                    # Show first 3 and last 2 values
                    preview = f"[{self.embedding[0]:.4f}, {self.embedding[1]:.4f}, {self.embedding[2]:.4f}, ..., {self.embedding[-2]:.4f}, {self.embedding[-1]:.4f}]"
                else:
                    # Show all values if 5 or fewer
                    preview = "[" + ", ".join(f"{v:.4f}" for v in self.embedding) + "]"

                # Add shape info if available
                if hasattr(self.embedding, 'shape'):
                    preview += f" shape={self.embedding.shape}"

                return preview
            else:
                return str(self.embedding)
        except:
            return "<embedding>"

    def __repr__(self) -> str:
        """Return a detailed string representation of the Chunk."""
        repr = (
            f"Chunk(text='{self.text}', token_count={self.token_count}, "
            f"start_index={self.start_index}, end_index={self.end_index}"
        )
        if self.context:
            repr += f", context={self.context}"
        if self.embedding is not None:
            repr += f", embedding={self._preview_embedding()}"
        return repr + ")"

    def __iter__(self) -> Iterator[str]:
        """Return an iterator over the chunk's text."""
        return iter(self.text)

    def __getitem__(self, index: int) -> str:
        """Return a slice of the chunk's text."""
        return self.text[index]

    def to_dict(self) -> dict:
        """Return the Chunk as a dictionary."""
        result = self.__dict__.copy()
        result["context"] = self.context.to_dict() if self.context else None
        # Convert embedding to list if it has tolist method (numpy array)
        if self.embedding is not None:
            if hasattr(self.embedding, 'tolist'):
                result["embedding"] = self.embedding.tolist()
            else:
                result["embedding"] = self.embedding
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        """Create a Chunk object from a dictionary."""
        context_repr = data.get("context", None)
        embedding_data = data.get("embedding", None)

        return cls(
            text=data["text"],
            start_index=data["start_index"],
            end_index=data["end_index"],
            token_count=data["token_count"],
            context=Context.from_dict(context_repr) if context_repr else None,
            embedding=embedding_data,  # Keep as-is, whatever type it is
        )

    def copy(self) -> "Chunk":
        """Return a deep copy of the chunk."""
        return Chunk.from_dict(self.to_dict())
