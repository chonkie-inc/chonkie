"""Model2Vec embeddings."""

import importlib.util as importutil
from typing import TYPE_CHECKING, Union

import numpy as np

from .base import BaseEmbeddings

if TYPE_CHECKING:
    from model2vec import StaticModel
    from tokenizers import Tokenizer


class Model2VecEmbeddings(BaseEmbeddings):
    """Class for model2vec embeddings.

    This class provides an interface for the model2vec library, which provides a variety
    of pre-trained models for text embeddings.

    Args:
        model (str or StaticModel): Name of the model2vec model to load or a StaticModel instance

    """

    def __init__(self, model: Union[str, "StaticModel"] = "minishlab/potion-base-32M") -> None:
        """Initialize Model2VecEmbeddings with a str or StaticModel instance."""
        super().__init__()
        self.model_name_or_path: str | None

        try:
            from model2vec import StaticModel
        except ImportError as ie:
            raise ImportError(
                "model2vec is not available. Please install it via `pip install chonkie[model2vec]`",
            ) from ie

        if isinstance(model, str):
            self.model_name_or_path = model
            self.model = StaticModel.from_pretrained(self.model_name_or_path)
        elif isinstance(model, StaticModel):
            self.model = model

            # TODO: `base_model_name` is mentioned in here -
            # https://github.com/MinishLab/model2vec/blob/b1358a9c2e777800e8f89c7a5f830fa2176c15b5/model2vec/model.py#L165`
            # but its `None` for potion models
            self.model_name_or_path = self.model.base_model_name
        else:
            raise ValueError("model must be a string or model2vec.StaticModel instance")
        self._dimension = self.model.dim

    @property
    def dimension(self) -> int:
        """Dimension of the embedding vectors."""
        return self._dimension

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text using the model2vec model."""
        return self.model.encode(text, convert_to_numpy=True)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed multiple texts using the model2vec model."""
        return self.model.encode(texts, convert_to_numpy=True)  # type: ignore[return-value]

    def similarity(self, u: np.ndarray, v: np.ndarray) -> np.float32:
        """Compute cosine similarity of two embeddings."""
        return np.divide(np.dot(u, v), np.linalg.norm(u) * np.linalg.norm(v), dtype=np.float32)

    def get_tokenizer(self) -> "Tokenizer":
        """Get the tokenizer or token counter for the model."""
        return self.model.tokenizer

    @classmethod
    def _is_available(cls) -> bool:
        """Check if model2vec is available."""
        return importutil.find_spec("model2vec") is not None

    def __repr__(self) -> str:
        """Representation of the Model2VecEmbeddings instance."""
        return f"Model2VecEmbeddings(model={self.model_name_or_path})"
