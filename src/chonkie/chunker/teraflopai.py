"""Chonkie: TeraflopAI Chunker.

Splits text into chunks using the TeraflopAI Segmentation API.
"""

import os
from typing import TYPE_CHECKING, Optional, Union

from chonkie.chunker.base import BaseChunker
from chonkie.logger import get_logger
from chonkie.pipeline import chunker
from chonkie.tokenizer import TokenizerProtocol
from chonkie.types import Chunk

if TYPE_CHECKING:
    from teraflopai import TeraflopAI

logger = get_logger(__name__)

SEGMENTATION_URL = "https://api.segmentation.teraflopai.com/v1/segmentation/free"


@chunker("teraflopai")
class TeraflopAIChunker(BaseChunker):
    """Chunker that uses the TeraflopAI Segmentation API to split text into chunks.

    This chunker sends text to the TeraflopAI segmentation endpoint and converts
    the returned segments into Chonkie Chunk objects. It is useful for domain-specific
    segmentation such as legal text.

    Args:
        client: An existing TeraflopAI client instance. If provided, url and api_key are ignored.
        url: The URL for the TeraflopAI segmentation API.
        api_key: The API key for authentication. Falls back to the TERAFLOPAI_API_KEY env var.
        tokenizer: Tokenizer used to compute token counts for returned chunks.

    """

    def __init__(
        self,
        client: Optional["TeraflopAI"] = None,
        url: str = SEGMENTATION_URL,
        api_key: Optional[str] = None,
        tokenizer: Union[str, TokenizerProtocol] = "character",
    ) -> None:
        """Initialize the TeraflopAIChunker.

        Args:
            client: An existing TeraflopAI client instance.
            url: The URL for the TeraflopAI segmentation API.
            api_key: The API key for authentication.
            tokenizer: Tokenizer to use for counting tokens.

        Raises:
            ImportError: If the teraflopai package is not installed.
            ValueError: If no API key is provided and TERAFLOPAI_API_KEY is not set.

        """
        try:
            from teraflopai import TeraflopAI
        except ImportError as e:
            raise ImportError(
                "teraflopai is not installed. "
                "Please install it with `pip install chonkie[teraflopai]`.",
            ) from e

        # used only for calculating number of tokens in chunks, not for segmentation
        super().__init__(tokenizer=tokenizer)

        if client is not None:
            self.client = client
        else:
            api_key = api_key or os.getenv("TERAFLOPAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "API key is required. Provide api_key or set the "
                    "TERAFLOPAI_API_KEY environment variable."
                )
            if not url:
                raise ValueError("URL is required for TeraflopAI client.")
            self.client = TeraflopAI(api_key=api_key, url=url)

        self._use_multiprocessing = False
        logger.debug("Initialized TeraflopAIChunker", url=self.client.url)

    def chunk(self, text: str) -> list[Chunk]:
        """Segment text using the TeraflopAI Segmentation API.

        Args:
            text: The text to segment into chunks.

        Returns:
            A list of Chunk objects corresponding to the segments.

        """
        if not text or not text.strip():
            return []

        logger.debug(f"Sending text of length {len(text)} to TeraflopAI segmentation API")
        response = self.client.segment(text)

        segments = response.get("results", [])
        if not segments:
            return [
                Chunk(
                    text=text,
                    start_index=0,
                    end_index=len(text),
                    token_count=self._tokenizer.count_tokens(text),
                )
            ]

        chunks: list[Chunk] = []
        for segment in segments:
            segment_text = segment if isinstance(segment, str) else str(segment)
            start_index = text.find(segment_text)
            if start_index == -1:
                start_index = 0
            end_index = start_index + len(segment_text)
            token_count = self._tokenizer.count_tokens(segment_text)
            chunks.append(
                Chunk(
                    text=segment_text,
                    start_index=start_index,
                    end_index=end_index,
                    token_count=token_count,
                )
            )

        logger.info(f"Created {len(chunks)} chunks using TeraflopAI segmentation")
        return chunks

    def __repr__(self) -> str:
        """Get a string representation of the TeraflopAI chunker."""
        return f"TeraflopAIChunker(url={self.client.url})"
