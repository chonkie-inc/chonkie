"""Embedding Refinery."""

from typing import Any, Dict, List, Union

from chonkie.embeddings import AutoEmbeddings, BaseEmbeddings
from chonkie.logger import get_logger
from chonkie.pipeline import refinery
from chonkie.types import Chunk

from .base import BaseRefinery

logger = get_logger(__name__)


@refinery("embeddings")
class EmbeddingsRefinery(BaseRefinery):
    """Embedding Refinery.

    Embeds the text of the chunks using the embedding model and
    adds the embeddings to the chunks for use in downstream tasks
    like upserting into a vector database.

    Args:
        embedding_model: The embedding model to use.
        **kwargs: Additional keyword arguments.

    """

    def __init__(
        self,
        embedding_model: Union[
            str, BaseEmbeddings, AutoEmbeddings
        ] = "minishlab/potion-retrieval-32M",
        **kwargs: Dict[str, Any],
    ) -> None:
        """Initialize the EmbeddingRefinery."""
        super().__init__()

        # Check if the model is a string
        if isinstance(embedding_model, str):
            self.embedding_model = AutoEmbeddings.get_embeddings(
                embedding_model, **kwargs
            )
        elif isinstance(embedding_model, BaseEmbeddings):
            self.embedding_model = embedding_model
        else:
            raise ValueError("Model must be a string or a BaseEmbeddings instance.")

    def refine(self, chunks: List[Chunk]) -> List[Chunk]:
        """Refine the chunks."""
        logger.debug(f"Starting embedding refinery for {len(chunks)} chunks")
        texts = [chunk.text for chunk in chunks]
        embeds = self.embedding_model.embed_batch(texts)
        for chunk, embed in zip(chunks, embeds):
            chunk.embedding = embed  # type: ignore[attr-defined]
        logger.info(f"Embedding refinement complete: added embeddings to {len(chunks)} chunks")
        return chunks

    def __repr__(self) -> str:
        """Represent the EmbeddingRefinery."""
        return f"EmbeddingsRefinery(embedding_model={self.embedding_model})"

    @property
    def dimension(self) -> int:
        """Dimension of the embedding model."""
        return self.embedding_model.dimension
