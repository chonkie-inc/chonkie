"""Implements the SentenceChunker class for splitting text into chunks based on sentence boundaries.

This module provides the `SentenceChunker`, a specialized chunker that segments text
by identifying sentence endings (like periods, question marks, etc.) while adhering to
specified token count limits for each chunk. It also handles overlapping chunks and
allows customization of sentence boundary delimiters and minimum sentence lengths.
"""

import warnings
from bisect import bisect_left
from itertools import accumulate
from typing import Any, Callable, List, Literal, Optional, Sequence, Union

from chonkie.types.base import Chunk
from chonkie.types.sentence import Sentence, SentenceChunk
from chonkie.utils import Hubbie

from .base import BaseChunker


class SentenceChunker(BaseChunker):
    """SentenceChunker splits the sentences in a text based on token limits and sentence boundaries.

    Args:
        tokenizer_or_token_counter: The tokenizer instance to use for encoding/decoding
        chunk_size: Maximum number of tokens per chunk
        chunk_overlap: Number of tokens to overlap between chunks
        min_sentences_per_chunk: Minimum number of sentences per chunk (defaults to 1)
        min_characters_per_sentence: Minimum number of characters per sentence
        approximate: Whether to use approximate token counting (defaults to False) [DEPRECATED]
        delim: Delimiters to split sentences on
        include_delim: Whether to include delimiters in current chunk, next chunk or not at all (defaults to "prev")
        return_type: Whether to return chunks or texts

    Raises:
        ValueError: If parameters are invalid

    """

    def __init__(
        self,
        tokenizer_or_token_counter: Union[str, Callable, Any] = "gpt2",
        chunk_size: int = 512,
        chunk_overlap: int = 0,
        min_sentences_per_chunk: int = 1,
        min_characters_per_sentence: int = 12,
        approximate: bool = False,
        delim: Union[str, List[str]] = [". ", "! ", "? ", "\n"],
        include_delim: Optional[Literal["prev", "next"]] = "prev",
        return_type: Literal["chunks", "texts"] = "chunks",
    ):
        """Initialize the SentenceChunker with configuration parameters.

        SentenceChunker splits the sentences in a text based on token limits and sentence boundaries.

        Args:
            tokenizer_or_token_counter: The tokenizer instance to use for encoding/decoding (defaults to "gpt2")
            chunk_size: Maximum number of tokens per chunk (defaults to 512)
            chunk_overlap: Number of tokens to overlap between chunks (defaults to 0)
            min_sentences_per_chunk: Minimum number of sentences per chunk (defaults to 1)
            min_characters_per_sentence: Minimum number of characters per sentence (defaults to 12)
            approximate: Whether to use approximate token counting (defaults to False)
            delim: Delimiters to split sentences on (defaults to [". ", "! ", "? ", "newline"])
            include_delim: Whether to include delimiters in current chunk, next chunk or not at all (defaults to "prev")
            return_type: Whether to return chunks or texts (defaults to "chunks")

        Raises:
            ValueError: If parameters are invalid

        """
        super().__init__(tokenizer_or_token_counter=tokenizer_or_token_counter)

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
        if return_type not in ["chunks", "texts"]:
            raise ValueError("Invalid return_type. Must be either 'chunks' or 'texts'.")
        if approximate:
            warnings.warn("Approximate has been deprecated and will be removed from next version onwards!")

        # Assign the values if they make sense
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_sentences_per_chunk = min_sentences_per_chunk
        self.min_characters_per_sentence = min_characters_per_sentence
        self.approximate = approximate
        self.delim = delim
        self.include_delim = include_delim
        self.sep = "✄"
        self.return_type = return_type
    
    @classmethod
    def from_recipe(cls, 
        name: Optional[str] = "default", 
        lang: Optional[str] = "en", 
        path: Optional[str] = None, 
        tokenizer_or_token_counter: Union[str, Callable, Any] = "gpt2",
        chunk_size: int = 512,
        chunk_overlap: int = 0,
        min_sentences_per_chunk: int = 1,
        min_characters_per_sentence: int = 12,
        approximate: bool = False,
        return_type: Literal["chunks", "texts"] = "chunks",
        ) -> "SentenceChunker":
        """Create a SentenceChunker from a recipe.

        Takes the `delim` and `include_delim` from the recipe and passes the rest of the parameters to the constructor.

        The recipes are registered in the [Chonkie Recipe Store](https://huggingface.co/datasets/chonkie-ai/recipes). If the recipe is not there, you can create your own recipe and share it with the community!
        
        Args:
            name: The name of the recipe to use.
            lang: The language that the recipe should support. 
            path: The path to the recipe to use.
            tokenizer_or_token_counter: The tokenizer or token counter to use.
            chunk_size: The chunk size to use.
            chunk_overlap: The chunk overlap to use.
            min_sentences_per_chunk: The minimum number of sentences per chunk to use.
            min_characters_per_sentence: The minimum number of characters per sentence to use.
            approximate: Whether to use approximate token counting.
            return_type: Whether to return chunks or texts.
            
        Returns:
            SentenceChunker: The created SentenceChunker.

        Raises:
            ValueError: If the recipe is invalid.

        """
        # Create a hubbie instance
        hub = Hubbie()
        recipe = hub.get_recipe(name, lang, path)
        return cls(
            tokenizer_or_token_counter=tokenizer_or_token_counter,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_sentences_per_chunk=min_sentences_per_chunk,
            min_characters_per_sentence=min_characters_per_sentence,
            delim=recipe["recipe"]["delimiters"],
            include_delim=recipe["recipe"]["include_delim"],
            return_type=return_type,
        )


    def _split_text(self, text: str) -> List[str]:
        """Fast sentence splitting while maintaining accuracy.

        This method is faster than using regex for sentence splitting and is more accurate than using the spaCy sentence tokenizer.

        Args:
            text: Input text to be split into sentences

        Returns:
            List of sentences

        """
        t = text
        for c in self.delim:
            if self.include_delim == "prev":
                t = t.replace(c, c + self.sep)
            elif self.include_delim == "next":
                t = t.replace(c, self.sep + c)
            else:
                t = t.replace(c, self.sep)

        # Initial split
        splits = [s for s in t.split(self.sep) if s != ""]

        # Combine short splits with previous sentence
        current = ""
        sentences = []
        for s in splits:
            # If the split is short, add to current and if long add to sentences
            if len(s) < self.min_characters_per_sentence:
                current += s
            elif current:
                current += s
                sentences.append(current)
                current = ""
            else:
                sentences.append(s)

            # At any point if the current sentence is longer than the min_characters_per_sentence,
            # add it to the sentences
            if len(current) >= self.min_characters_per_sentence:
                sentences.append(current)
                current = ""

        # If there is a current split, add it to the sentences
        if current:
            sentences.append(current)

        return sentences

    def _prepare_sentences(self, text: str) -> List[Sentence]:
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
                sent
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

    def _create_chunk(self, sentences: List[Sentence]) -> Union[Chunk, str]:
        """Create a chunk from a list of sentences.

        Args:
            sentences: List of sentences to create chunk from

        Returns:
            Chunk object

        """
        chunk_text = "".join([sentence.text for sentence in sentences])
        if self.return_type == "texts":
            return chunk_text
        else:

            # We calculate the token count here, as sum of the token counts of the sentences
            # does not match the token count of the chunk as a whole for some reason. That's to 
            # say that the tokenizer encodes the text differently when the text is joined together.
            # This is where time savings can be made when getting only the texts. 
            token_count = self.tokenizer.count_tokens(chunk_text)

            return SentenceChunk(
                text=chunk_text,
                start_index=sentences[0].start_index,
                end_index=sentences[-1].end_index,
                token_count=token_count,
                sentences=sentences,
            )

    def chunk(self, text: str) -> Union[Sequence[Chunk], Sequence[str]]:
        """Split text into overlapping chunks based on sentences while respecting token limits.

        Args:
            text: Input text to be chunked

        Returns:
            List of Chunk objects containing the chunked text and metadata

        """
        if not text.strip():
            return []

        # Get prepared sentences with token counts
        sentences = self._prepare_sentences(text)  # 28mus
        if not sentences:
            return []

        # Pre-calculate cumulative token counts for bisect
        # Add 1 token for spaces between sentences
        token_sums = list(
            accumulate(
                [s.token_count for s in sentences],
                lambda a, b: a + b,
                initial=0,
            )
        )

        chunks = []
        pos = 0

        while pos < len(sentences):
            # Use bisect_left to find initial split point
            target_tokens = token_sums[pos] + self.chunk_size
            split_idx = bisect_left(token_sums, target_tokens) - 1
            split_idx = min(split_idx, len(sentences))

            # Ensure we include at least one sentence beyond pos
            split_idx = max(split_idx, pos + 1)

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
                        + "Consider increasing the chunk_size or decreasing the min_sentences_per_chunk."
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

        return chunks # type: ignore

    def __repr__(self) -> str:
        """Return a string representation of the SentenceChunker."""
        return (
            f"SentenceChunker(tokenizer={self.tokenizer}, "
            f"chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}, "
            f"min_sentences_per_chunk={self.min_sentences_per_chunk}, "
            f"min_characters_per_sentence={self.min_characters_per_sentence}, "
            f"approximate={self.approximate}, delim={self.delim}, "
            f"include_delim={self.include_delim}, "
            f"return_type={self.return_type})"
        )
