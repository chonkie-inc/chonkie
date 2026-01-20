"""Base Class for All Chunkers."""

import asyncio
import warnings
from abc import ABC, abstractmethod
from typing import Sequence, Union
from dataclasses import replace

from tqdm import tqdm

from chonkie.logger import get_logger
from chonkie.tokenizer import AutoTokenizer, TokenizerProtocol
from chonkie.types import Chunk, Document

logger = get_logger(__name__)


class BaseChunker(ABC):
    """Base class for all chunkers."""

    def __init__(self, tokenizer: Union[str, TokenizerProtocol] = "gpt2"):
        """Initialize the chunker with any necessary parameters.

        Args:
            tokenizer: The tokenizer to use. Can be:
                - A string identifier (e.g., "gpt2", "character", "word")
                - An object implementing TokenizerProtocol (encode, decode, tokenize methods)

        """
        self._tokenizer = AutoTokenizer(tokenizer)
        self._use_multiprocessing = True
        logger.debug(
            f"Initialized {self.__class__.__name__}",
            tokenizer=str(tokenizer)[:50],
        )

    @property
    def tokenizer(self) -> AutoTokenizer:
        """Get the tokenizer instance."""
        return self._tokenizer

    def __repr__(self) -> str:
        """Return a string representation of the chunker."""
        return f"{self.__class__.__name__}()"

    def __call__(
        self,
        text: Union[str, Sequence[str]],
        show_progress: bool = True,
    ) -> Union[list[Chunk], list[list[Chunk]]]:
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
            from multiprocessing import cpu_count

            cpu_cores = cpu_count()
            worker_count = min(8, max(1, cpu_cores * 3 // 4))
            logger.debug(
                f"Using {worker_count} workers for parallel processing",
                cpu_cores=cpu_cores,
            )
            return worker_count
        except Exception as e:
            warnings.warn(f"Proceeding with 1 worker. Error calculating optimal worker count: {e}")
            logger.warning(
                "Failed to calculate optimal worker count, using 1 worker",
                error=str(e),
            )
            return 1

    def _sequential_batch_processing(
        self,
        texts: Sequence[str],
        show_progress: bool = True,
    ) -> list[list[Chunk]]:
        """Process a batch of texts sequentially."""
        logger.info(f"Starting sequential batch processing of {len(texts)} texts")
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
        total_chunks = sum(len(r) for r in results)
        logger.info(
            f"Completed sequential processing: {total_chunks} total chunks from {len(texts)} texts",
        )
        return results

    def _parallel_batch_processing(
        self,
        texts: Sequence[str],
        show_progress: bool = True,
    ) -> list[list[Chunk]]:
        """Process a batch of texts using multiprocessing."""
        from multiprocessing import Pool

        num_workers = self._get_optimal_worker_count()
        total = len(texts)
        chunk_size = max(1, min(total // (num_workers * 16), 10))

        logger.info(
            f"Starting parallel batch processing of {total} texts",
            workers=num_workers,
            chunk_size=chunk_size,
        )

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

            total_chunks = sum(len(r) for r in results)
            logger.info(
                f"Completed parallel processing: {total_chunks} total chunks from {total} texts",
            )
            return results

    @abstractmethod
    def chunk(self, text: str) -> list[Chunk]:
        """Chunk the given text.

        Args:
            text (str): The text to chunk.

        Returns:
            list[Chunk]: A list of Chunks.

        """
        pass

    def chunk_batch(self, texts: Sequence[str], show_progress: bool = True) -> list[list[Chunk]]:
        """Chunk a batch of texts.

        Args:
            texts (Sequence[str]): The texts to chunk.
            show_progress (bool): Whether to show progress.

        Returns:
            list[list[Chunk]]: A list of lists of Chunks.

        """
        # simple handles of empty and single text cases
        if len(texts) == 0:
            return []
        if len(texts) == 1:
            return [self.chunk(texts[0])]  # type: ignore

        # Now for the remaining, check the self._multiprocessing bool flag
        if self._use_multiprocessing:
            return self._parallel_batch_processing(texts, show_progress)
        else:
            return self._sequential_batch_processing(texts, show_progress)

    async def chunk_async(self, text: str) -> list[Chunk]:
        """Chunk the given text asynchronously.

        Args:
            text (str): The text to chunk.

        Returns:
            list[Chunk]: A list of Chunks.

        """
        return await asyncio.to_thread(self.chunk, text)

    async def chunk_batch_async(
        self, texts: Sequence[str], show_progress: bool = True
    ) -> list[list[Chunk]]:
        """Chunk a batch of texts asynchronously.

        Args:
            texts (Sequence[str]): The texts to chunk.
            show_progress (bool): Whether to show progress.

        Returns:
            list[list[Chunk]]: A list of lists of Chunks.

        """
        return await asyncio.to_thread(self.chunk_batch, texts, show_progress)

    @staticmethod
    def _merge_new_chunks(
        original_chunks: list[Chunk], new_chunk_batches: list[list[Chunk]]
    ) -> list[Chunk]:
        """Merge new chunks batches into a single list, shifting indices.

        Args:
            original_chunks: The original chunks from the document.
            new_chunk_batches: The new batches of chunks corresponding to each original chunk.

        Returns:
            list[Chunk]: The merged and shifted chunks.

        """
        return [
            replace(
                c,
                start_index=c.start_index + old_chunk.start_index,
                end_index=c.end_index + old_chunk.start_index,
            )
            for old_chunk, new_chunks in zip(original_chunks, new_chunk_batches)
            for c in new_chunks
        ]

    def chunk_document(self, document: Document) -> Document:
        """Chunk a document.

        Args:
            document: The document to chunk.

        Returns:
            The document with chunks populated.

        """
        # If the document has chunks already, then we need to re-chunk the content
        if document.chunks:
            chunk_results = [self.chunk(c.text) for c in document.chunks]
            document.chunks = self._merge_new_chunks(document.chunks, chunk_results)
        else:
            document.chunks = self.chunk(document.content)
        return document

    async def chunk_document_async(self, document: Document) -> Document:
        """Chunk a document asynchronously.

        Args:
            document: The document to chunk.

        Returns:
            The document with chunks populated.

        """
        # If the document has chunks already, then we need to re-chunk the content
        if document.chunks:
            tasks = [self.chunk_async(c.text) for c in document.chunks]
            chunk_results = await asyncio.gather(*tasks)
            document.chunks = self._merge_new_chunks(document.chunks, chunk_results)
        else:
            document.chunks = await self.chunk_async(document.content)
        return document
