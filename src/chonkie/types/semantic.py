"""Semantic types for Chonkie."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

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

    def to_dict(self) -> dict:
        """Return the SemanticSentence as a dictionary."""
        result = super().to_dict()
        result["embedding"] = (
            self.embedding.tolist() if self.embedding is not None else None
        )
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "SemanticSentence":
        """Create a SemanticSentence object from a dictionary."""
        embedding_list = data.pop("embedding")
        # NOTE: We can't use np.array() here because we don't import numpy in this file,
        # and we don't want add 50MiB to the package size.
        embedding = embedding_list if embedding_list is not None else None
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

    sentences: List[SemanticSentence] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return the SemanticChunk as a dictionary."""
        result = super().to_dict()
        result["sentences"] = [sentence.to_dict() for sentence in self.sentences]
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "SemanticChunk":
        """Create a SemanticChunk object from a dictionary."""
        sentences_dict = data.pop("sentences")
        sentences = [SemanticSentence.from_dict(sentence) for sentence in sentences_dict]
        return cls(**data, sentences=sentences)

    def __repr__(self) -> str:
        """Return a string representation of the SemanticChunk."""
        return (
            f"SemanticChunk(text={self.text}, start_index={self.start_index}, "
            f"end_index={self.end_index}, token_count={self.token_count}, "
            f"sentences={self.sentences})"
        )
