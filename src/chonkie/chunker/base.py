"""Base Class for All Chunkers."""

import warnings
from abc import ABC, abstractmethod
from multiprocessing import Pool, cpu_count
from typing import Any, Callable, List, Sequence, Union

from tqdm import tqdm

from chonkie.tokenizer import Tokenizer
from chonkie.types import Chunk, Document


class BaseChunker(ABC):
    """Base class for all chunkers."""

    def __init__(
        self, tokenizer_or_token_counter: Union[str, Callable[[str], int], Any]
    ):
        """Initialize the chunker with any necessary parameters.

        Args:
            tokenizer_or_token_counter (Union[str, Callable[[str], int], Any]): The tokenizer or token counter to use.

        """
        self.tokenizer = Tokenizer(tokenizer_or_token_counter)
        self._use_multiprocessing = True

    def __repr__(self) -> str:
        """Return a string representation of the chunker."""
        return f"{self.__class__.__name__}()"

    def __call__(
        self, text: Union[str, Sequence[str]], show_progress: bool = True
    ) -> Union[List[Chunk], List[List[Chunk]]]:
        """Call the chunker with the given arguments.

        Args:
            text (Union[str, Sequence[str]]): The text to chunk.
            show_progress (bool): Whether to show progress.

        Returns:
            If the input is a string, return a list of Chunks.
            If the input is a list of strings, return a list of lists of Chunks.

        """
        if isinstance(text, str):
            return self.chunk(text)
        elif isinstance(text, Sequence):
            return self.chunk_batch(text, show_progress)
        else:
            raise ValueError("Input must be a string or a list of strings.")

    def _get_optimal_worker_count(self) -> int:
        """Get the optimal number of workers for parallel processing."""
        try:
            cpu_cores = cpu_count()
            return min(8, max(1, cpu_cores * 3 // 4))
        except Exception as e:
            warnings.warn(
                f"Proceeding with 1 worker. Error calculating optimal worker count: {e}"
            )
            return 1

    def _sequential_batch_processing(
        self, texts: Sequence[str], show_progress: bool = True
    ) -> List[List[Chunk]]:
        """Process a batch of texts sequentially."""
        results = [
            self.chunk(t)
            for t in tqdm(
                texts,
                desc="🦛",
                disable=not show_progress,
                unit="doc",
                bar_format="{desc} ch{bar:20}nk {percentage:3.0f}% • {n_fmt}/{total_fmt} docs chunked [{elapsed}<{remaining}, {rate_fmt}] 🌱",
                ascii=" o",
            )
        ]
        return results

    def _parallel_batch_processing(
        self, texts: Sequence[str], show_progress: bool = True
    ) -> List[List[Chunk]]:
        """Process a batch of texts using multiprocessing."""
        num_workers = self._get_optimal_worker_count()
        total = len(texts)
        chunk_size = max(1, min(total // (num_workers * 16), 10))

        with Pool(processes=num_workers) as pool:
            results = []
            with tqdm(
                total=total,
                desc="🦛",
                disable=not show_progress,
                unit="doc",
                bar_format="{desc} ch{bar:20}nk {percentage:3.0f}% • {n_fmt}/{total_fmt} docs chunked [{elapsed}<{remaining}, {rate_fmt}] 🌱",
                ascii=" o",
            ) as progress_bar:
                for result in pool.imap(self.chunk, texts, chunksize=chunk_size):
                    results.append(result)
                    progress_bar.update()
            return results

    @abstractmethod
    def chunk(self, text: str) -> List[Chunk]:
        """Chunk the given text.

        Args:
            text (str): The text to chunk.

        Returns:
            List[Chunk]: A list of Chunks.

        """
        pass

    def chunk_batch(
        self, texts: Sequence[str], show_progress: bool = True
    ) -> List[List[Chunk]]:
        """Chunk a batch of texts.

        Args:
            texts (Sequence[str]): The texts to chunk.
            show_progress (bool): Whether to show progress.

        Returns:
            List[List[Chunk]]: A list of lists of Chunks.

        """
        # simple handles of empty and single text cases
        if len(texts) == 0:
            return []
        if len(texts) == 1:
            return [ self.chunk(texts[0]) ] # type: ignore

        # Now for the remaining, check the self._multiprocessing bool flag
        if self._use_multiprocessing:
            return self._parallel_batch_processing(texts, show_progress)
        else:
            return self._sequential_batch_processing(texts, show_progress)
    
    def chunk_document(self, document: Document) -> Document: 
        """Chunk a document."""
        # If the document has chunks already, then we need to re-chunk the content
        if document.chunks:
            chunks: List[Chunk] = []
            for old_chunk in document.chunks:
                new_chunks: List[Chunk] = self.chunk(old_chunk.text)
                for new_chunk in new_chunks:
                    chunks.append(
                        Chunk(
                            text=new_chunk.text, 
                            start_index=new_chunk.start_index + old_chunk.start_index,
                            end_index=new_chunk.end_index + old_chunk.start_index,
                            token_count=new_chunk.token_count,
                        )
                    )
            document.chunks = chunks
        else:
            document.chunks = self.chunk(document.content)
        return document
