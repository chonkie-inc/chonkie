"""Implements the SentenceChunker class for splitting text into chunks based on sentence boundaries.

This module provides the `SentenceChunker`, a specialized chunker that segments text
by identifying sentence endings (like periods, question marks, etc.) while adhering to
specified token count limits for each chunk. It also handles overlapping chunks and
allows customization of sentence boundary delimiters and minimum sentence lengths.
"""

import os
import warnings
from typing import Literal, Optional, Sequence, Union

import chonkie_core

from chonkie.logger import get_logger
from chonkie.pipeline import chunker
from chonkie.tokenizer import TokenizerProtocol
from chonkie.types import Chunk, Sentence
from chonkie.utils import Hubbie

from .base import BaseChunker

logger = get_logger(__name__)


@chunker("sentence")
class SentenceChunker(BaseChunker):
    """SentenceChunker splits the sentences in a text based on token limits and sentence boundaries.

    Args:
        tokenizer: The tokenizer instance to use for encoding/decoding
        chunk_size: Maximum number of tokens per chunk
        chunk_overlap: Number of tokens to overlap between chunks
        min_sentences_per_chunk: Minimum number of sentences per chunk (defaults to 1)
        min_characters_per_sentence: Minimum number of characters per sentence
        approximate: Whether to use approximate token counting (defaults to False) [DEPRECATED]
        delim: Delimiters to split sentences on
        include_delim: Whether to include delimiters in current chunk, next chunk or not at all (defaults to "prev")

    Raises:
        ValueError: If parameters are invalid

    """

    def __init__(
        self,
        tokenizer: Union[str, TokenizerProtocol] = "character",
        chunk_size: int = 2048,
        chunk_overlap: int = 0,
        min_sentences_per_chunk: int = 1,
        min_characters_per_sentence: int = 12,
        approximate: bool = False,
        delim: Union[str, list[str]] = [". ", "! ", "? ", "\n"],
        include_delim: Optional[Literal["prev", "next"]] = "prev",
    ):
        """Initialize the SentenceChunker with configuration parameters.

        SentenceChunker splits the sentences in a text based on token limits and sentence boundaries.

        Args:
            tokenizer: The tokenizer instance to use for encoding/decoding (defaults to "character")
            chunk_size: Maximum number of tokens per chunk (defaults to 2048)
            chunk_overlap: Number of tokens to overlap between chunks (defaults to 0)
            min_sentences_per_chunk: Minimum number of sentences per chunk (defaults to 1)
            min_characters_per_sentence: Minimum number of characters per sentence (defaults to 12)
            approximate: Whether to use approximate token counting (defaults to False)
            delim: Delimiters to split sentences on (defaults to [". ", "! ", "? ", "newline"])
            include_delim: Whether to include delimiters in current chunk, next chunk or not at all (defaults to "prev")

        Raises:
            ValueError: If parameters are invalid

        """
        super().__init__(tokenizer=tokenizer)

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if min_sentences_per_chunk < 1:
            raise ValueError("min_sentences_per_chunk must be at least 1")
        if min_characters_per_sentence < 1:
            raise ValueError("min_characters_per_sentence must be at least 1")
        if delim is None:
            raise ValueError("delim must be a list of strings or a string")
        if include_delim not in ["prev", "next", None]:
            raise ValueError("include_delim must be 'prev', 'next' or None")
        if approximate:
            warnings.warn(
                "Approximate has been deprecated and will be removed from next version onwards!",
            )

        # Assign the values if they make sense
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_sentences_per_chunk = min_sentences_per_chunk
        self.min_characters_per_sentence = min_characters_per_sentence
        self.approximate = approximate
        self.delim = delim
        self.include_delim = include_delim
        self.sep = "✄"

    @classmethod
    def from_recipe(
        cls,
        name: Optional[str] = "default",
        lang: Optional[str] = "en",
        path: str | os.PathLike | None = None,
        tokenizer: Union[str, TokenizerProtocol] = "character",
        chunk_size: int = 2048,
        chunk_overlap: int = 0,
        min_sentences_per_chunk: int = 1,
        min_characters_per_sentence: int = 12,
        approximate: bool = False,
    ) -> "SentenceChunker":
        """Create a SentenceChunker from a recipe.

        Takes the `delim` and `include_delim` from the recipe and passes the rest of the parameters to the constructor.

        The recipes are registered in the [Chonkie Recipe Store](https://huggingface.co/datasets/chonkie-ai/recipes). If the recipe is not there, you can create your own recipe and share it with the community!

        Args:
            name: The name of the recipe to use.
            lang: The language that the recipe should support.
            path: The path to the recipe to use.
            tokenizer: The tokenizer to use.
            chunk_size: The chunk size to use.
            chunk_overlap: The chunk overlap to use.
            min_sentences_per_chunk: The minimum number of sentences per chunk to use.
            min_characters_per_sentence: The minimum number of characters per sentence to use.
            approximate: Whether to use approximate token counting.

        Returns:
            SentenceChunker: The created SentenceChunker.

        Raises:
            ValueError: If the recipe is invalid.

        """
        # Create a hubbie instance
        hub = Hubbie()
        logger.info("Loading SentenceChunker recipe", recipe_name=name, lang=lang)
        recipe = hub.get_recipe(name, lang, path)
        logger.debug(
            "Recipe loaded successfully",
            delim=recipe.get("delim"),
            include_delim=recipe.get("include_delim"),
        )
        return cls(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_sentences_per_chunk=min_sentences_per_chunk,
            min_characters_per_sentence=min_characters_per_sentence,
            delim=recipe["recipe"]["delimiters"],
            include_delim=recipe["recipe"]["include_delim"],
        )

    def _split_text(self, text: str) -> list[str]:
        """Fast sentence splitting using chonkie-core.

        This method is faster than using regex for sentence splitting and is more accurate than using the spaCy sentence tokenizer.

        Args:
            text: Input text to be split into sentences

        Returns:
            List of sentences

        """
        # Normalize delimiters to a list
        if isinstance(self.delim, str):
            delimiters = [self.delim]
        else:
            delimiters = self.delim

        text_bytes = text.encode("utf-8")
        include_mode = self.include_delim or "prev"

        # Check if we have multi-byte delimiters
        has_multibyte = any(len(d) > 1 for d in delimiters)

        if has_multibyte:
            # Use split_pattern_offsets for multi-byte patterns
            patterns = [d.encode("utf-8") for d in delimiters]
            offsets = chonkie_core.split_pattern_offsets(
                text_bytes,
                patterns=patterns,
                include_delim=include_mode,
                min_chars=self.min_characters_per_sentence,
            )
        else:
            # Use faster split_offsets for single-byte delimiters
            delim_bytes = "".join(delimiters).encode("utf-8")
            offsets = chonkie_core.split_offsets(
                text_bytes,
                delimiters=delim_bytes,
                include_delim=include_mode,
                min_chars=self.min_characters_per_sentence,
            )

        # Convert byte offsets to strings
        splits = [text_bytes[start:end].decode("utf-8") for start, end in offsets]
        return [s for s in splits if s]

    def _prepare_sentences(self, text: str) -> list[Sentence]:
        """Split text into sentences and calculate token counts for each sentence.

        Args:
            text: Input text to be split into sentences

        Returns:
            List of Sentence objects

        """
        # Split text into sentences
        sentence_texts = self._split_text(text)
        if not sentence_texts:
            return []

        # Calculate positions once
        positions = []
        current_pos = 0
        for sent in sentence_texts:
            positions.append(current_pos)
            current_pos += len(
                sent,
            )  # No +1 space because sentences are already separated by spaces

        # Get accurate token counts in batch (this is faster than estimating)
        token_counts: Sequence[int] = self.tokenizer.count_tokens_batch(sentence_texts)

        # Create sentence objects
        return [
            Sentence(
                text=sent,
                start_index=pos,
                end_index=pos + len(sent),
                token_count=count,
            )
            for sent, pos, count in zip(sentence_texts, positions, token_counts)
        ]

    def _create_chunk(self, sentences: list[Sentence]) -> Chunk:
        """Create a chunk from a list of sentences.

        Args:
            sentences: List of sentences to create chunk from

        Returns:
            Chunk object

        """
        chunk_text = "".join([sentence.text for sentence in sentences])

        # We calculate the token count here, as sum of the token counts of the sentences
        # does not match the token count of the chunk as a whole for some reason. That's to
        # say that the tokenizer encodes the text differently when the text is joined together.
        token_count = self.tokenizer.count_tokens(chunk_text)

        return Chunk(
            text=chunk_text,
            start_index=sentences[0].start_index,
            end_index=sentences[-1].end_index,
            token_count=token_count,
        )

    def chunk(self, text: str) -> list[Chunk]:
        """Split text into overlapping chunks based on sentences while respecting token limits.

        Args:
            text: Input text to be chunked

        Returns:
            List of Chunk objects containing the chunked text and metadata

        """
        if not text.strip():
            logger.debug("Empty text provided, returning empty chunk list")
            return []

        logger.debug(f"Chunking text of length {len(text)}")

        # Get prepared sentences with token counts
        sentences = self._prepare_sentences(text)  # 28mus
        if not sentences:
            logger.debug("No sentences extracted from text")
            return []

        logger.debug(f"Prepared {len(sentences)} sentences for chunking")

        chunks = []
        pos = 0

        while pos < len(sentences):
            # Use chonkie-core for merge index calculation
            remaining_token_counts = [s.token_count for s in sentences[pos:]]
            if remaining_token_counts:
                indices = chonkie_core.find_merge_indices(remaining_token_counts, self.chunk_size)
                if indices:
                    split_idx = pos + indices[0]
                else:
                    split_idx = len(sentences)
            else:
                split_idx = len(sentences)

            # Handle minimum sentences requirement
            if split_idx - pos < self.min_sentences_per_chunk:
                # If the minimum sentences per chunk can be met, set the split index to the minimum sentences per chunk
                # Otherwise, warn the user that the minimum sentences per chunk could not be met for all chunks
                if pos + self.min_sentences_per_chunk <= len(sentences):
                    split_idx = pos + self.min_sentences_per_chunk
                else:
                    warnings.warn(
                        f"Minimum sentences per chunk as {self.min_sentences_per_chunk} could not be met for all chunks. "
                        + f"Last chunk of the text will have only {len(sentences) - pos} sentences. "
                        + "Consider increasing the chunk_size or decreasing the min_sentences_per_chunk.",
                    )
                    split_idx = len(sentences)

            # Get candidate sentences and verify actual token count
            chunk_sentences = sentences[pos:split_idx]
            chunks.append(self._create_chunk(chunk_sentences))

            # TODO: This would also get deprecated when we have OverlapRefinery in the future.
            # Calculate next position with overlap
            if self.chunk_overlap > 0 and split_idx < len(sentences):
                # Calculate how many sentences we need for overlap
                overlap_tokens = 0
                overlap_idx = split_idx - 1

                while overlap_idx > pos and overlap_tokens < self.chunk_overlap:
                    sent = sentences[overlap_idx]
                    next_tokens = overlap_tokens + sent.token_count + 1  # +1 for space
                    if next_tokens > self.chunk_overlap:
                        break
                    overlap_tokens = next_tokens
                    overlap_idx -= 1

                # Move position to after the overlap
                pos = overlap_idx + 1
            else:
                pos = split_idx

        logger.info(f"Created {len(chunks)} chunks from text", text_length=len(text))
        return chunks

    def __repr__(self) -> str:
        """Return a string representation of the SentenceChunker."""
        return (
            f"SentenceChunker(tokenizer={self.tokenizer}, "
            f"chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}, "
            f"min_sentences_per_chunk={self.min_sentences_per_chunk}, "
            f"min_characters_per_sentence={self.min_characters_per_sentence}, "
            f"approximate={self.approximate}, delim={self.delim}, "
            f"include_delim={self.include_delim})"
        )
