"""Model2Vec embeddings."""

import importlib.util as importutil
from typing import TYPE_CHECKING, List, Union

from chonkie.embeddings.base import BaseEmbeddings

if TYPE_CHECKING:
    try:
        import numpy as np
        from model2vec import StaticModel
        from tokenizers import Tokenizer
    except ImportError:
        np = Any  # type: ignore
        StaticModel = Any  # type: ignore
        Tokenizer = Any  # type: ignore


class Model2VecEmbeddings(BaseEmbeddings):
    """Class for model2vec embeddings.

    This class provides an interface for the model2vec library, which provides a variety
    of pre-trained models for text embeddings.

    Args:
        model (str or StaticModel): Name of the model2vec model to load or a StaticModel instance

    """

    def __init__(
        self, model: Union[str, "StaticModel"] = "minishlab/potion-base-8M"
    ) -> None:
        """Initialize Model2VecEmbeddings with a str or StaticModel instance."""
        super().__init__()

        # Lazy import dependencies if they are not already imported
        self._import_dependencies()

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

    def embed(self, text: str) -> "np.ndarray":
        """Embed a single text using the model2vec model."""
        return self.model.encode(text, convert_to_numpy=True)  # type: ignore

    def embed_batch(self, texts: List[str]) -> List["np.ndarray"]:
        """Embed multiple texts using the model2vec model."""
        return self.model.encode(texts, convert_to_numpy=True)  # type: ignore

    def similarity(self, u: "np.ndarray", v: "np.ndarray") -> "np.float32":  # type: ignore
        """Compute cosine similarity of two embeddings."""
        return np.divide(
            np.dot(u, v), np.linalg.norm(u) * np.linalg.norm(v), dtype=np.float32
        )

    def get_tokenizer_or_token_counter(self) -> "Tokenizer":
        """Get the tokenizer or token counter for the model."""
        return self.model.tokenizer

    @classmethod
    def is_available(cls) -> bool:
        """Check if model2vec is available."""
        return importutil.find_spec("model2vec") is not None

    @classmethod
    def _import_dependencies(cls) -> None:
        """Lazy import dependencies for the embeddings implementation.

        This method should be implemented by all embeddings implementations that require
        additional dependencies. It lazily imports the dependencies only when they are needed.
        """
        if cls.is_available():
            global np, StaticModel
            import numpy as np
            from model2vec import StaticModel
        else:
            raise ImportError(
                "model2vec is not available. Please install it via `pip install chonkie[model2vec]`"
            )

    def __repr__(self) -> str:
        """Representation of the Model2VecEmbeddings instance."""
        return f"Model2VecEmbeddings(model_name_or_path={self.model_name_or_path})"
