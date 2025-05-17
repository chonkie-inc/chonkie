"""Module containing TokenChunker class.

This module provides a TokenChunker class for splitting text into chunks of a specified token size.

"""

import asyncio
from typing import Any, Generator, List, Literal, Sequence, Union, AsyncIterator

from tqdm import trange

from chonkie.chunker.base import BaseChunker, T
from chonkie.types.base import Chunk


class TokenChunker(BaseChunker[Chunk]):
    """Chunker that splits text into chunks of a specified token size.

    Args:
        tokenizer: The tokenizer instance to use for encoding/decoding
        chunk_size: Maximum number of tokens per chunk
        chunk_overlap: Number of tokens to overlap between chunks
        return_type: Whether to return chunks or texts

    """

    def __init__(
        self,
        tokenizer: Union[str, Any] = "gpt2",
        chunk_size: int = 512,
        chunk_overlap: Union[int, float] = 0,
        return_type: Literal["chunks", "texts"] = "chunks",
    ) -> None:
        """Initialize the TokenChunker with configuration parameters.

        Args:
            tokenizer: The tokenizer instance to use for encoding/decoding
            chunk_size: Maximum number of tokens per chunk
            chunk_overlap: Number of tokens to overlap between chunks
            return_type: Whether to return chunks or texts

        Raises:
            ValueError: If chunk_size <= 0 or chunk_overlap >= chunk_size

        """
        super().__init__(tokenizer)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if isinstance(chunk_overlap, int) and chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if return_type not in ["chunks", "texts"]:
            raise ValueError("return_type must be either 'chunks' or 'texts'")

        # Assign the values if they make sense
        self.return_type = return_type
        self.chunk_size = chunk_size
        self.chunk_overlap = (
            chunk_overlap
            if isinstance(chunk_overlap, int)
            else int(chunk_overlap * chunk_size)
        )

        self._use_multiprocessing = False

    def _create_chunks(
        self,
        chunk_texts: List[str],
        token_groups: List[List[int]],
        token_counts: List[int],
    ) -> Sequence[Chunk]:
        """Create chunks from a list of texts."""
        # Find the overlap lengths for index calculation
        if self.chunk_overlap > 0:
            # we get the overlap texts, that gives you the start_index for the next chunk
            # if the token group is smaller than the overlap, we just use the whole token group
            overlap_texts = self.tokenizer.decode_batch([
                token_group[-self.chunk_overlap :]
                if (len(token_group) > self.chunk_overlap)
                else token_group
                for token_group in token_groups
            ])
            overlap_lengths = [len(overlap_text) for overlap_text in overlap_texts]
        else:
            overlap_lengths = [0] * len(token_groups)

        # Create the chunks
        chunks = []
        current_index = 0
        for chunk_text, overlap_length, token_count in zip(
            chunk_texts, overlap_lengths, token_counts
        ):
            start_index = current_index
            end_index = start_index + len(chunk_text)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    start_index=start_index,
                    end_index=end_index,
                    token_count=token_count,
                )
            )
            current_index = end_index - overlap_length

        return chunks

    def _token_group_generator(
        self, tokens: List[int]
    ) -> Generator[List[int], None, None]:
        """Generate chunks from a list of tokens."""
        for start in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
            end = min(start + self.chunk_size, len(tokens))
            yield tokens[start:end]
            if end == len(tokens):
                break

    async def _async_token_group_generator(
        self, tokens: List[int]
    ) -> AsyncIterator[List[int]]:
        """Generate chunks asynchronously from a list of tokens."""
        for start in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
            end = min(start + self.chunk_size, len(tokens))
            yield tokens[start:end]
            if end == len(tokens):
                break

    def chunk(self, text: str) -> Sequence[Chunk]:
        """Split text into overlapping chunks of specified token size.

        Args:
            text: Input text to be chunked

        Returns:
            List of Chunk objects containing the chunked text and metadata

        """
        if not text.strip():
            return []

        # Encode full text
        text_tokens = self.tokenizer.encode(text)

        # Calculate token groups and counts
        token_groups = list(self._token_group_generator(text_tokens))

        # if return_type is chunks, we need to decode the token groups into the chunk texts
        if self.return_type == "chunks":
            token_counts = [len(toks) for toks in token_groups]

            # decode the token groups into the chunk texts
            chunk_texts = self.tokenizer.decode_batch(token_groups)

            # Create the chunks from the token groups and token counts
            chunks = self._create_chunks(chunk_texts, token_groups, token_counts)

            return chunks
        # if return_type is texts, we can just return the decoded token groups
        elif self.return_type == "texts":
            return self.tokenizer.decode_batch(token_groups)

    async def async_chunk(self, text: str) -> Sequence[Chunk]:
        """Split text into overlapping chunks asynchronously.
        
        Optimized asynchronous implementation of chunk() method.

        Args:
            text: Input text to be chunked

        Returns:
            List of Chunk objects containing the chunked text and metadata
        """
        if not text.strip():
            return []

        # Create a thread-safe executor task for CPU-bound tokenization
        loop = asyncio.get_event_loop()
        text_tokens = await loop.run_in_executor(None, self.tokenizer.encode, text)
        
        # Calculate token groups
        token_groups = []
        async for token_group in self._async_token_group_generator(text_tokens):
            token_groups.append(token_group)

        # Process according to return type
        if self.return_type == "chunks":
            token_counts = [len(toks) for toks in token_groups]
            
            # Use executor for batch decoding
            chunk_texts = await loop.run_in_executor(
                None, self.tokenizer.decode_batch, token_groups
            )
            
            # Create chunks
            chunks = await loop.run_in_executor(
                None, self._create_chunks, chunk_texts, token_groups, token_counts
            )
            
            return chunks
        else:  # return_type is "texts"
            return await loop.run_in_executor(
                None, self.tokenizer.decode_batch, token_groups
            )

    async def stream_chunk(self, text: str) -> AsyncIterator[Chunk]:
        """Stream chunks as they're generated.
        
        Args:
            text: Input text to be chunked
            
        Yields:
            Chunks as they're created, one by one
        """
        if not text.strip():
            return
            
        # Create a thread-safe executor task for CPU-bound tokenization
        loop = asyncio.get_event_loop()
        text_tokens = await loop.run_in_executor(None, self.tokenizer.encode, text)
        
        current_index = 0
        previous_token_group = None
        
        async for token_group in self._async_token_group_generator(text_tokens):
            # Decode the token group
            chunk_text = await loop.run_in_executor(
                None, self.tokenizer.decode, token_group
            )
            
            # Calculate overlap length if needed
            overlap_length = 0
            if self.chunk_overlap > 0 and previous_token_group is not None:
                overlap_tokens = previous_token_group[-self.chunk_overlap:] if len(previous_token_group) > self.chunk_overlap else previous_token_group
                overlap_text = await loop.run_in_executor(None, self.tokenizer.decode, overlap_tokens)
                overlap_length = len(overlap_text)
            
            # Create chunk
            start_index = current_index
            end_index = start_index + len(chunk_text)
            
            chunk = Chunk(
                text=chunk_text,
                start_index=start_index,
                end_index=end_index,
                token_count=len(token_group),
            )
            
            yield chunk
            
            # Update for next iteration
            current_index = end_index - overlap_length
            previous_token_group = token_group

    def _process_batch(self, texts: List[str]) -> Sequence[Sequence[Chunk]]:
        """Process a batch of texts."""
        # encode the texts into tokens in a batch
        tokens_list = self.tokenizer.encode_batch(texts)
        result = []

        for tokens in tokens_list:
            if not tokens:
                result.append([])
                continue

            # get the token groups
            token_groups = list(self._token_group_generator(tokens))

            if self.return_type == "chunks":
                # get the token counts
                token_counts = [len(token_group) for token_group in token_groups]

                # decode the token groups into the chunk texts
                chunk_texts = self.tokenizer.decode_batch(token_groups)

                # create the chunks from the token groups and token counts
                chunks = self._create_chunks(chunk_texts, token_groups, token_counts)
                result.append(chunks)
            elif self.return_type == "texts":
                result.append(self.tokenizer.decode_batch(token_groups))
            else:
                raise ValueError(
                    "Invalid return_type. Must be either 'chunks' or 'texts'."
                )

        return result

    async def _async_process_batch(self, texts: List[str]) -> Sequence[Sequence[Chunk]]:
        """Process a batch of texts asynchronously."""
        loop = asyncio.get_event_loop()
        
        # Encode the texts into tokens in a batch (CPU-bound)
        tokens_list = await loop.run_in_executor(
            None, self.tokenizer.encode_batch, texts
        )
        
        result = []
        for tokens in tokens_list:
            if not tokens:
                result.append([])
                continue

            # Get the token groups
            token_groups = []
            async for token_group in self._async_token_group_generator(tokens):
                token_groups.append(token_group)

            if self.return_type == "chunks":
                # Get the token counts
                token_counts = [len(token_group) for token_group in token_groups]

                # Decode the token groups into the chunk texts (CPU-bound)
                chunk_texts = await loop.run_in_executor(
                    None, self.tokenizer.decode_batch, token_groups
                )

                # Create the chunks from the token groups and token counts
                chunks = await loop.run_in_executor(
                    None, self._create_chunks, chunk_texts, token_groups, token_counts
                )
                result.append(chunks)
            elif self.return_type == "texts":
                decoded = await loop.run_in_executor(
                    None, self.tokenizer.decode_batch, token_groups
                )
                result.append(decoded)

        return result

    def chunk_batch(
        self,
        texts: List[str],
        batch_size: int = 1,
        show_progress_bar: bool = True,
    ) -> Sequence[Sequence[Chunk]]:
        """Split a batch of texts into their respective chunks.

        Args:
            texts: List of input texts to be chunked
            batch_size: Number of texts to process in a single batch
            show_progress_bar: Whether to show a progress bar

        Returns:
            List of lists of Chunk objects containing the chunked text and metadata

        """
        chunks = []
        for i in trange(
            0,
            len(texts),
            batch_size,
            desc="🦛",
            disable=not show_progress_bar,
            unit="batch",
            bar_format="{desc} ch{bar:20}nk {percentage:3.0f}% • {n_fmt}/{total_fmt} batches chunked [{elapsed}<{remaining}, {rate_fmt}] 🌱",
            ascii=" o",
        ):
            batch_texts = texts[i : min(i + batch_size, len(texts))]
            chunks.extend(self._process_batch(batch_texts))
        return chunks

    def __call__(
        self,
        text: Union[str, List[str]],
        batch_size: int = 1,
        show_progress_bar: bool = True,
    ) -> Union[Sequence[Chunk], Sequence[Sequence[Chunk]]]:
        """Make the TokenChunker callable directly.

        Args:
            text: Input text or list of texts to be chunked
            batch_size: Number of texts to process in a single batch
            show_progress_bar: Whether to show a progress bar (for batch chunking)

        Returns:
            List of Chunk objects or list of lists of Chunk

        """
        if isinstance(text, str):
            return self.chunk(text)
        elif isinstance(text, list) and isinstance(text[0], str):
            return self.chunk_batch(text, batch_size, show_progress_bar)
        else:
            raise ValueError(
                "Invalid input type. Expected a string or a list of strings."
            )

    def __repr__(self) -> str:
        """Return a string representation of the TokenChunker."""
        return (
            f"TokenChunker(tokenizer={self.tokenizer}, "
            f"chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}, "
            f"return_type={self.return_type})"
        )
