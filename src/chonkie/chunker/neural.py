"""Module containing NeuralChunker class.

This module provides a NeuralChunker class for splitting text into chunks using a 100% neural approach, inspired by the work of [Chonky](https://github.com/mirth/chonky). 

It trains an encoder style model on the task of token-classification (think: NER) to predict the split points of a text.
"""

import importlib.util as importutil
from typing import Any, Dict, List, Literal, Union

from chonkie.types import Chunk

from .base import BaseChunker


class NeuralChunker(BaseChunker):
  """Class for chunking text using a complete Neural Approach.
  
  This has been adapted from the implementation and models provided
  by [Chonky](https://github.com/mirth/chonky). This approach uses 
  a token classification model to predict the split points in a text.

  Args:
    model: The model to use for the chunker.
    device: The device to use for the chunker.
    min_characters_per_chunk: The minimum number of characters per chunk.
    return_type: The type of return value.

  """

  def __init__(self,
               model: str = "mirth/chonky_modernbert_base_1",
               device: str = "cpu", 
               min_characters_per_chunk: int = 10, 
               return_type: Literal["chunks", "texts"] = "chunks") -> None:
    """Initialize the NeuralChunker object.
    
    Args:
      model: The model to use for the chunker.
      device: The device to use for the chunker.
      min_characters_per_chunk: The minimum number of characters per chunk.
      return_type: The type of return value.

    """
    # Lazily load the dependencies
    self._import_dependencies()

    # Initialize the tokenizer to pass in to the parent class
    try:
      tokenizer = AutoTokenizer.from_pretrained(model) # type: ignore
    except Exception as e:
      raise ValueError(f"Error initializing tokenizer: {e}")

    # Initialize the Parent class with the tokenizer
    super().__init__(tokenizer)

    # Set the attributes
    self.model = model
    self.device = device
    self.min_characters_per_chunk = min_characters_per_chunk
    self.return_type = return_type

    # Initialize the pipeline
    try:
      self.pipe = pipeline("token-classification", model=model, device=device) # type: ignore
    except Exception as e:
      raise ValueError(f"Error initializing pipeline: {e}")

    # Set the _use_multiprocessing value to be False
    self._use_multiprocessing = False
  
  def _is_available(self) -> bool:
    """Check if the dependencies are installed."""
    return importutil.find_spec("transformers") is not None

  def _import_dependencies(self) -> None:
    """Import the dependencies."""
    if self._is_available():
      global AutoTokenizer, pipeline
      from transformers import AutoTokenizer, pipeline
    else:
      raise ImportError("transformers is not installed. Please install it with `pip install chonkie[neural]`.")


  def _get_splits(self, 
                  response: List[Dict[str, Any]],
                  text: str) -> List[str]:
    """Get the text splits from the model."""
    splits = []
    current_index = 0
    for sample in response: 
      splits.append(text[current_index:sample['end']])
      current_index = sample['end']
    if current_index < len(text):
      splits.append(text[current_index:])
    return splits
  
  def _merge_close_spans(self, 
                        response: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
  
  def _get_chunks_from_splits(self, splits: List[str]) -> List[Chunk]:
    """Create a list of Chunks from the splits."""
    chunks = []
    current_index = 0
    token_counts = self.tokenizer.count_tokens_batch(splits)
    for split, token_count in zip(splits, token_counts): 
      chunks.append(Chunk(split, 
                          current_index,
                          current_index + len(split),
                          token_count))
      current_index += len(split)
    return chunks

  def chunk(self, text: str) -> Union[List[Chunk], List[str]]:
    """Chunk the text into a list of chunks.
    
    Args:
      text: The text to chunk.

    Returns:
      A list of chunks or a list of strings.

    """
    # Get the spans
    spans = self.pipe(text)

    # Merge close spans, since the model sometimes predicts spans that are too close to each other
    # and we want to ensure that we don't have chunks that are too small
    merged_spans = self._merge_close_spans(spans)

    # Get the splits from the merged spans
    splits = self._get_splits(merged_spans, text)

    # Return the chunks or the texts
    if self.return_type == "texts":
      return splits
    else:
      chunks = self._get_chunks_from_splits(splits)
      return chunks

  def __repr__(self) -> str:
    """Return the string representation of the object."""
    return (f"NeuralChunker(model={self.model}, device={self.device}, "
            f"min_characters_per_chunk={self.min_characters_per_chunk}, "
            f"return_type={self.return_type})")