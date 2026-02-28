"""Module containing NeuralChunker class.

This module provides a NeuralChunker class for splitting text into chunks using a 100% neural approach, inspired by the work of [Chonky](https://github.com/mirth/chonky).

It trains an encoder style model on the task of token-classification (think: NER) to predict the split points of a text.
"""

import importlib.util as importutil
from typing import Any, Optional, Union

from chonkie.logger import get_logger
from chonkie.pipeline import chunker
from chonkie.types import Chunk

from .base import BaseChunker

logger = get_logger(__name__)


@chunker("neural")
class NeuralChunker(BaseChunker):
    """Class for chunking text using a complete Neural Approach.

    This has been adapted from the implementation and models provided
    by [Chonky](https://github.com/mirth/chonky). This approach uses
    a token classification model to predict the split points in a text.

    Args:
      model: The model to use for the chunker.
      device: The device to use for the chunker.
      min_characters_per_chunk: The minimum number of characters per chunk.

    """

    SUPPORTED_MODELS = [
        "mirth/chonky_distilbert_base_uncased_1",
        "mirth/chonky_modernbert_base_1",
        "mirth/chonky_modernbert_large_1",
    ]

    SUPPORTED_MODEL_STRIDES = {
        "mirth/chonky_distilbert_base_uncased_1": 256,
        "mirth/chonky_modernbert_base_1": 512,
        "mirth/chonky_modernbert_large_1": 512,
    }

    DEFAULT_MODEL = "mirth/chonky_distilbert_base_uncased_1"

    def __init__(
        self,
        model: Union[str, Any] = DEFAULT_MODEL,
        tokenizer: Optional[Union[str, Any]] = None,
        device_map: str = "auto",
        min_characters_per_chunk: int = 10,
        stride: Optional[int] = None,
    ) -> None:
        """Initialize the NeuralChunker object.

        Args:
          model: The model to use for the chunker.
          tokenizer: The tokenizer to use for the chunker.
          device_map: The device to use for the chunker.
          min_characters_per_chunk: The minimum number of characters per chunk.
          stride: The stride to use for the chunker.

        """
        try:
            from transformers import (
                AutoModelForTokenClassification,
                AutoTokenizer,
                PreTrainedTokenizerFast,
                pipeline,
            )
        except ImportError as e:
            raise ImportError(
                "transformers is not installed. Please install it with `pip install chonkie[neural]`.",
            ) from e
        # Initialize the tokenizer to pass in to the parent class
        if isinstance(tokenizer, str):
            tokenizer_id = tokenizer
        elif tokenizer is None and isinstance(model, str):
            tokenizer_id = model
        elif isinstance(tokenizer, PreTrainedTokenizerFast):
            tokenizer_id = None  # already loaded
        else:
            raise ValueError(
                "Invalid tokenizer provided. Please provide a string or a transformers.PreTrainedTokenizerFast object."
            )

        if tokenizer_id is not None:
            try:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
            except OSError as e:
                raise ValueError(
                    f"Failed to load tokenizer '{tokenizer_id}'. "
                    "Check that the tokenizer name is correct."
                ) from e

        # Initialize the Parent class with the tokenizer
        super().__init__(tokenizer)

        # Initialize the model and stride
        if isinstance(model, str):
            # Check if the model is supported
            if model not in self.SUPPORTED_MODELS:
                raise ValueError(
                    f"Model {model} is not supported. Please choose from one of the following: {self.SUPPORTED_MODELS}"
                )
            model_id = model
        elif model is not None and "transformers" in str(type(model)):
            # Assuming that the model is a transformers model already, since it has transformers in the name, teehee~
            self.model = model
            model_id = None
            # Since a custom model is provided, we need to set the stride to 0
            stride = 0 if stride is None else stride
        else:
            raise ValueError(
                "Invalid model provided. Please provide a string or a transformers.AutoModelForTokenClassification object."
            )

        if model_id is not None:
            try:
                self.model = AutoModelForTokenClassification.from_pretrained(
                    model_id, device_map=device_map
                )
            except OSError as e:
                raise ValueError("The model is not compatible with token-classification.") from e
        # Set the attributes
        self.min_characters_per_chunk = min_characters_per_chunk

        # Initialize the pipeline
        try:
            self.pipe = pipeline(
                "token-classification",
                model=model,
                tokenizer=tokenizer,
                device_map=device_map,
                aggregation_strategy="simple",
                stride=stride,
            )
        except Exception as e:
            raise ValueError(f"Error initializing pipeline: {e}") from e

        # Set the _use_multiprocessing value to be False
        self._use_multiprocessing = False

    @classmethod
    def _is_available(cls) -> bool:
        """Check if the dependencies are installed."""
        return importutil.find_spec("transformers") is not None

    def _get_splits(self, response: list[dict[str, Any]], text: str) -> list[str]:
        """Get the text splits from the model."""
        splits = []
        current_index = 0
        for sample in response:
            splits.append(text[current_index : sample["end"]])
            current_index = sample["end"]
        if current_index < len(text):
            splits.append(text[current_index:])
        return splits

    def _merge_close_spans(self, response: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Replace the split points that are too close to each other with the last span."""
        if not response:
            return []

        merged_response = [response[0]]
        for i in range(1, len(response)):
            current_span = response[i]
            last_merged_span = merged_response[-1]

            if current_span["start"] - last_merged_span["end"] < self.min_characters_per_chunk:
                # If the current span is too close to the last merged one,
                # replace the last one with the current one.
                merged_response[-1] = current_span
            else:
                # Otherwise, append the current span.
                merged_response.append(current_span)

        return merged_response

    def _get_chunks_from_splits(self, splits: list[str]) -> list[Chunk]:
        """Create a list of Chunks from the splits."""
        chunks = []
        current_index = 0
        token_counts = self.tokenizer.count_tokens_batch(splits)
        for split, token_count in zip(splits, token_counts):
            chunks.append(
                Chunk(
                    text=split,
                    start_index=current_index,
                    end_index=current_index + len(split),
                    token_count=token_count,
                ),
            )
            current_index += len(split)
        return chunks

    def chunk(self, text: str) -> list[Chunk]:
        """Chunk the text into a list of chunks.

        Args:
          text: The text to chunk.

        Returns:
          A list of chunks.

        """
        logger.debug(f"Starting neural chunking for text of length {len(text)}")
        # Get the spans
        spans = self.pipe(text)
        logger.debug(f"Model predicted {len(spans)} split points")

        # Merge close spans, since the model sometimes predicts spans that are too close to each other
        # and we want to ensure that we don't have chunks that are too small
        merged_spans = self._merge_close_spans(spans)

        # Get the splits from the merged spans
        splits = self._get_splits(merged_spans, text)

        # Return the chunks
        chunks = self._get_chunks_from_splits(splits)
        logger.info(f"Created {len(chunks)} chunks using neural token classification")
        return chunks

    def __repr__(self) -> str:
        """Return the string representation of the object."""
        return (
            f"NeuralChunker(model={self.model},"
            f"tokenizer={self.tokenizer}, "
            f"min_characters_per_chunk={self.min_characters_per_chunk})"
        )
