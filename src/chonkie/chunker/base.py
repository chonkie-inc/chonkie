"""Base Class for All Chunkers."""

import warnings
import asyncio
from abc import ABC, abstractmethod
from multiprocessing import Pool, cpu_count
from typing import Any, Callable, Sequence, Union, Optional, List, Dict, TypeVar, Generic, AsyncIterator

from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm

from chonkie.tokenizer import Tokenizer
from chonkie.types.base import Chunk

T = TypeVar('T', Chunk, str)

class BaseChunker(ABC, Generic[T]):
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
        self._max_concurrency = 10  # Default max concurrent async tasks
        self._batch_size = 50  # Default batch size for async processing

    def __repr__(self) -> str:
        """Return a string representation of the chunker."""
        return f"{self.__class__.__name__}()"

    def __call__(
        self, text: Union[str, Sequence[str]], show_progress: bool = True
    ) -> Union[Sequence[T], Sequence[Sequence[T]]]:
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
    ) -> Sequence[Sequence[T]]:
        """Process a batch of texts sequentially."""
        return [
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

    def _parallel_batch_processing(
        self, texts: Sequence[str], show_progress: bool = True
    ) -> Sequence[Sequence[T]]:
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
    def chunk(self, text: str) -> Sequence[T]:
        """Chunk the given text.

        Args:
            text (str): The text to chunk.

        Returns:
            Sequence[T]: A list of Chunks or a list of strings.

        """
        pass

    async def async_chunk(self, text: str) -> Sequence[T]:
        """Asynchronous version of chunk method.
        
        By default, this calls the synchronous chunk method. Subclasses can override
        for true asynchronous implementation.
        
        Args:
            text (str): The text to chunk.
            
        Returns:
            Sequence[T]: A list of Chunks or a list of strings.
        """
        # Default implementation runs the synchronous version in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.chunk, text)

    async def stream_chunk(self, text: str) -> AsyncIterator[T]:
        """Stream chunks as they are generated.
        
        Args:
            text (str): The text to chunk.
            
        Returns:
            AsyncIterator[T]: An async iterator yielding chunks as they're created.
        """
        # Default implementation: just get all chunks and yield them
        chunks = await self.async_chunk(text)
        for chunk in chunks:
            yield chunk

    def chunk_batch(
        self, texts: Sequence[str], show_progress: bool = True
    ) -> Sequence[Sequence[T]]:
        """Chunk a batch of texts.

        Args:
            texts (Sequence[str]): The texts to chunk.
            show_progress (bool): Whether to show progress.

        Returns:
            Sequence[Sequence[T]]: A list of lists of Chunks or a list of lists of strings.

        """
        # simple handles of empty and single text cases
        if len(texts) == 0:
            return []
        if len(texts) == 1:
            return [self.chunk(texts[0])]

        # Now for the remaining, check the self._multiprocessing bool flag
        if self._use_multiprocessing:
            return self._parallel_batch_processing(texts, show_progress)
        else:
            return self._sequential_batch_processing(texts, show_progress)
            
    async def async_chunk_batch(
        self, texts: Sequence[str], show_progress: bool = True
    ) -> Sequence[Sequence[T]]:
        """Asynchronously chunk a batch of texts.
        
        Args:
            texts (Sequence[str]): The texts to chunk.
            show_progress (bool): Whether to show progress.
            
        Returns:
            Sequence[Sequence[T]]: A list of lists of Chunks or a list of lists of strings.
        """
        # Handle empty and single text cases
        if len(texts) == 0:
            return []
        if len(texts) == 1:
            return [await self.async_chunk(texts[0])]
            
        # Use semaphore to control concurrency
        semaphore = asyncio.Semaphore(self._max_concurrency)
        
        async def bounded_chunk(text):
            async with semaphore:
                return await self.async_chunk(text)
        
        # Process in batches with progress bar
        results = []
        if show_progress:
            for i in range(0, len(texts), self._batch_size):
                batch = texts[i:i + self._batch_size]
                batch_tasks = [bounded_chunk(text) for text in batch]
                
                for task in async_tqdm.as_completed(
                    batch_tasks,
                    desc="🦛",
                    total=len(batch),
                    unit="doc",
                    bar_format="{desc} ch{bar:20}nk {percentage:3.0f}% • {n_fmt}/{total_fmt} docs chunked [{elapsed}<{remaining}] 🌱",
                    ascii=" o",
                ):
                    results.append(await task)
        else:
            tasks = [bounded_chunk(text) for text in texts]
            results = await asyncio.gather(*tasks)
            
        return results
        
    def configure(self, 
                 use_multiprocessing: Optional[bool] = None,
                 max_concurrency: Optional[int] = None,
                 batch_size: Optional[int] = None) -> 'BaseChunker':
        """Configure chunker processing options.
        
        Args:
            use_multiprocessing (Optional[bool]): Whether to use multiprocessing for sync batch processing.
            max_concurrency (Optional[int]): Maximum number of concurrent tasks for async processing.
            batch_size (Optional[int]): Batch size for async processing.
            
        Returns:
            BaseChunker: Self for method chaining.
        """
        if use_multiprocessing is not None:
            self._use_multiprocessing = use_multiprocessing
        if max_concurrency is not None:
            self._max_concurrency = max_concurrency
        if batch_size is not None:
            self._batch_size = batch_size
        return self
