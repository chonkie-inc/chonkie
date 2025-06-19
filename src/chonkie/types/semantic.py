"""Semantic types for Chonkie."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from chonkie.types.sentence import Sentence, SentenceChunk

if TYPE_CHECKING:
    import numpy as np


@dataclass
class SemanticSentence(Sentence):
    """Dataclass representing a semantic sentence with metadata.

    This class is used to represent a sentence with an embedding.

    Attributes:
        text: The text content of the sentence
        start_index: The starting index of the sentence in the original text
        end_index: The ending index of the sentence in the original text
        token_count: The number of tokens in the sentence
        embedding: The sentence embedding

    """

    embedding: Optional["np.ndarray"] = field(default=None)

    def to_dict(self) -> Dict[str, Any]:
        """Return the SemanticSentence as a dictionary."""
        result: Dict[str, Any] = super().to_dict()
        if self.embedding is not None:
            try:
                result["embedding"] = self.embedding.tolist()
            except AttributeError:
                result["embedding"] = self.embedding
        else:
            result["embedding"] = None
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "SemanticSentence":
        """Create a SemanticSentence object from a dictionary.

        NOTE: If numpy is available, `.embedding` will be a numpy array.
        If not, it will be a list.
        """
        embedding_list = data.pop("embedding", None)
        # If numpy is available, we will use it.
        # If not, skip and keep it as a list.
        if embedding_list is not None:
            try:
                import numpy as np
                if isinstance(embedding_list, list):
                    embedding = np.array(embedding_list)
                else:
                    embedding = embedding_list
            except ImportError:
                embedding = embedding_list
        else:
            embedding = None
        return cls(**data, embedding=embedding)

    def __repr__(self) -> str:
        """Return a string representation of the SemanticSentence."""
        return (
            f"SemanticSentence(text={self.text}, start_index={self.start_index}, "
            f"end_index={self.end_index}, token_count={self.token_count}, "
            f"embedding={self.embedding})"
        )


@dataclass
class SemanticChunk(SentenceChunk):
    """SemanticChunk dataclass representing a semantic chunk with metadata.

    Attributes:
        text: The text content of the chunk
        start_index: The starting index of the chunk in the original text
        end_index: The ending index of the chunk in the original text
        token_count: The number of tokens in the chunk
        sentences: List of SemanticSentence objects in the chunk

    """

    sentences: List[SemanticSentence] = field(default_factory=list)  # type: ignore[assignment]

    def to_dict(self) -> dict:
        """Return the SemanticChunk as a dictionary."""
        result = super().to_dict()
        result["sentences"] = [sentence.to_dict() for sentence in self.sentences]
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "SemanticChunk":
        """Create a SemanticChunk object from a dictionary."""
        sentences_dict = data.pop("sentences", [])
        sentences = [SemanticSentence.from_dict(sentence) for sentence in sentences_dict]
        return cls(**data, sentences=sentences)

    def __repr__(self) -> str:
        """Return a string representation of the SemanticChunk."""
        return (
            f"SemanticChunk(text={self.text}, start_index={self.start_index}, "
            f"end_index={self.end_index}, token_count={self.token_count}, "
            f"sentences={self.sentences})"
        )
