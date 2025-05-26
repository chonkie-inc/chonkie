"""Custom types for Sentence Chunking."""

from dataclasses import dataclass, field
from typing import Dict, List, Union

from chonkie.types.base import Chunk


@dataclass
class Sentence:
    """Class to represent a sentence.

    Attributes:
        text (str): The text of the sentence.
        start_index (int): The starting index of the sentence in the original text.
        end_index (int): The ending index of the sentence in the original text.
        token_count (int): The number of tokens in the sentence.

    """

    text: str
    start_index: int
    end_index: int
    token_count: int

    def __post_init__(self) -> None:
        """Validate attributes."""
        if not isinstance(self.text, str):
            raise ValueError("Text must be a string.")
        if not isinstance(self.start_index, int) or self.start_index < 0:
            raise ValueError("Start index must be a non-negative integer.")
        if not isinstance(self.end_index, int) or self.end_index < 0:
            raise ValueError("End index must be a non-negative integer.")
        if self.start_index > self.end_index:
            raise ValueError("Start index must be less than end index.")
        if (
            not (
                isinstance(self.token_count, int) or isinstance(self.token_count, float)
            )
            or self.token_count < 0
        ):
            raise ValueError("Token count must be a non-negative integer.")

    def __repr__(self) -> str:
        """Return a string representation of the Sentence."""
        return (
            f"Sentence(text={self.text}, start_index={self.start_index}, "
            f"end_index={self.end_index}, token_count={self.token_count})"
        )

    def to_dict(self) -> Dict[str, Union[str, int]]:
        """Return the Chunk as a dictionary."""
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, data: Dict[str, Union[str, int]]) -> "Sentence":
        """Create a Sentence object from a dictionary."""
        return cls(**data)


@dataclass
class SentenceChunk(Chunk):
    """Class to represent sentence chunks.

    Attributes:
        text (str): The text of the chunk.
        start_index (int): The starting index of the chunk in the original text.
        end_index (int): The ending index of the chunk in the original text.
        token_count (int): The number of tokens in the chunk.
        sentences (list[Sentence]): List of sentences in the chunk.

    """

    sentences: List[Sentence] = field(default_factory=list)

    def __repr__(self) -> str:
        """Return a string representation of the SentenceChunk."""
        return (
            f"SentenceChunk(text={self.text}, start_index={self.start_index}, "
            f"end_index={self.end_index}, token_count={self.token_count}, "
            f"sentences={self.sentences})"
        )

    def to_dict(self) -> Dict:
        """Return the SentenceChunk as a dictionary."""
        result = super().to_dict()
        result["sentences"] = [sentence.to_dict() for sentence in self.sentences]
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "SentenceChunk":
        """Create a SentenceChunk from dictionary."""
        sentences_dict = data.pop("sentences") if "sentences" in data else None
        sentences = (
            [Sentence.from_dict(sentence) for sentence in sentences_dict]
            if sentences_dict is not None
            else []
        )
        return cls(**data, sentences=sentences)
