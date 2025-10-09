from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from chonkie.types import Chunk
from .base import BaseChunker
import importlib.util as importutil

if TYPE_CHECKING:
    try:
        from transformers import PreTrainedTokenizerFast, pipeline
    except ImportError:
        class PreTrainedTokenizerFast:
            pass
        def pipeline(*args, **kwargs):
            pass

class NeuralChunker(BaseChunker):
    """NeuralChunker with safe lazy imports and optional fallback."""

    SUPPORTED_MODELS = [
        "mirth/chonky_distilbert_base_uncased_1",
        "mirth/chonky_modernbert_base_1",
        "mirth/chonky_modernbert_large_1",
    ]

    DEFAULT_MODEL = "mirth/chonky_distilbert_base_uncased_1"

    def __init__(self,
                 model: Union[str, Any] = DEFAULT_MODEL,
                 tokenizer: Optional[Union[str, Any]] = None,
                 device_map: str = "auto",
                 min_characters_per_chunk: int = 10,
                 stride: Optional[int] = None):
        """Initialize NeuralChunker safely."""
        # Lazy import
        self._import_dependencies()

        self.min_characters_per_chunk = min_characters_per_chunk

        # Initialize tokenizer
        if tokenizer is None and isinstance(model, str):
            try:
                tokenizer = self.AutoTokenizer.from_pretrained(model)
            except Exception as e:
                raise ValueError(f"Error loading tokenizer: {e}")
        elif isinstance(tokenizer, str):
            try:
                tokenizer = self.AutoTokenizer.from_pretrained(tokenizer)
            except Exception as e:
                raise ValueError(f"Error loading tokenizer: {e}")

        super().__init__(tokenizer)

        # Initialize model
        if isinstance(model, str):
            if model not in self.SUPPORTED_MODELS:
                raise ValueError(f"Model '{model}' not supported. Use one of {self.SUPPORTED_MODELS}")
            try:
                self.model = self.AutoModelForTokenClassification.from_pretrained(model, device_map=device_map)
            except Exception as e:
                raise ValueError(f"Error loading model: {e}")
        else:
            self.model = model

        # Set stride
        if stride is None:
            stride = 256  # safe default

        # Initialize pipeline
        try:
            self.pipe = self.pipeline(
                "token-classification",
                model=self.model,
                tokenizer=tokenizer,
                device_map=device_map,
                aggregation_strategy="simple",
                stride=stride
            )
        except Exception as e:
            raise ValueError(f"Error initializing pipeline: {e}")

        self._use_multiprocessing = False

    def _is_available(self) -> bool:
        return importutil.find_spec("transformers") is not None

    def _import_dependencies(self):
        if self._is_available():
            global AutoTokenizer, AutoModelForTokenClassification, pipeline, PreTrainedTokenizerFast
            from transformers import (
                AutoModelForTokenClassification,
                AutoTokenizer,
                PreTrainedTokenizerFast,
                pipeline,
            )
        else:
            raise ImportError(
                "transformers not installed. Install with `pip install chonkie[neural]`"
            )

    def chunk(self, text: str) -> List[Chunk]:
        """Chunk text using the neural model."""
        spans = self.pipe(text)
        merged_spans = self._merge_close_spans(spans)
        splits = self._get_splits(merged_spans, text)
        return self._get_chunks_from_splits(splits)
