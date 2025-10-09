from typing import Any, Callable, Generator, List, Sequence, Union

from tqdm import trange

from chonkie.chunker.base import BaseChunker
from chonkie.types import Chunk


class TokenChunker(BaseChunker):
    """Chunker that splits text into overlapping chunks based on token count."""

    def __init__(
        self,
        tokenizer: Union[str, Callable[[str], int], Any] = "character",
        chunk_size: int = 2048,
        chunk_overlap: Union[int, float] = 0,
        verbose: bool = False,
    ) -> None:
        super().__init__(tokenizer)

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")

        if isinstance(chunk_overlap, (int, float)):
            chunk_overlap = int(chunk_overlap * chunk_size) if isinstance(chunk_overlap, float) else chunk_overlap
            if chunk_overlap >= chunk_size:
                raise ValueError("chunk_overlap must be less than chunk_size.")
        else:
            raise ValueError("chunk_overlap must be an int or float.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.verbose = verbose
        self._use_multiprocessing = False

    # -------------------------
    # Helper methods
    # -------------------------
    def _token_group_generator(self, tokens: Sequence[int]) -> Generator[list[int], None, None]:
        """Yield slices of tokens respecting chunk_size and chunk_overlap."""
        step = self.chunk_size - self.chunk_overlap
        for start in range(0, len(tokens), step):
            end = min(start + self.chunk_size, len(tokens))
            yield list(tokens[start:end])
            if end == len(tokens):
                break

    def _decode_and_create_chunks(
        self, token_groups: list[list[int]], start_index: int = 0
    ) -> list[Chunk]:
        """Decode token groups and convert them into Chunk objects with proper indices."""
        # Precompute overlaps
        overlap_lengths = []
        if self.chunk_overlap > 0:
            overlap_texts = self.tokenizer.decode_batch([
                group[-self.chunk_overlap:] if len(group) > self.chunk_overlap else group
                for group in token_groups
            ])
            overlap_lengths = [len(txt) for txt in overlap_texts]
        else:
            overlap_lengths = [0] * len(token_groups)

        chunks = []
        current_index = start_index
        for group, overlap, group_token_count in zip(token_groups, overlap_lengths, map(len, token_groups)):
            chunk_text = self.tokenizer.decode_batch([group])[0]
            chunks.append(Chunk(
                text=chunk_text,
                start_index=current_index,
                end_index=current_index + len(chunk_text),
                token_count=group_token_count
            ))
            current_index += len(chunk_text) - overlap

        return chunks

    # -------------------------
    # Single text chunking
    # -------------------------
    def chunk(self, text: str) -> list[Chunk]:
        """Chunk a single text into overlapping token chunks."""
        if not text.strip():
            return []

        encoded_tokens = self.tokenizer.encode(text)
        token_groups = list(self._token_group_generator(encoded_tokens))
        return self._decode_and_create_chunks(token_groups)

    # -------------------------
    # Batch chunking
    # -------------------------
    def _process_batch(self, texts: list[str]) -> list[list[Chunk]]:
        """Process a batch of texts."""
        result = []
        tokens_batch = self.tokenizer.encode_batch(texts)

        for tokens in tokens_batch:
            if not tokens:
                result.append([])
                continue
            token_groups = list(self._token_group_generator(tokens))
            chunks = self._decode_and_create_chunks(token_groups)
            result.append(chunks)

        return result

    def chunk_batch(
        self, texts: list[str], batch_size: int = 1, show_progress_bar: bool = True
    ) -> list[list[Chunk]]:
        """Chunk a list of texts in batches."""
        all_chunks = []
        for i in trange(0, len(texts), batch_size, desc="🦛", disable=not show_progress_bar,
                        unit="batch", bar_format="{desc} ch{bar:20}nk {percentage:3.0f}% • {n_fmt}/{total_fmt} batches [{elapsed}<{remaining}] 🌱"):
            batch = texts[i:i+batch_size]
            all_chunks.extend(self._process_batch(batch))
        return all_chunks

    # -------------------------
    # Callable interface
    # -------------------------
    def __call__(self, text: Union[str, list[str]], batch_size: int = 1, show_progress_bar: bool = True):
        if isinstance(text, str):
            return self.chunk(text)
        elif isinstance(text, list) and all(isinstance(t, str) for t in text):
            return self.chunk_batch(text, batch_size=batch_size, show_progress_bar=show_progress_bar)
        else:
            raise ValueError("Input must be a string or list of strings.")

    def __repr__(self) -> str:
        return f"TokenChunker(tokenizer={self.tokenizer}, chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap})"
