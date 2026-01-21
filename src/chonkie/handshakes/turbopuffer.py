"""Turbopuffer Handshake to export Chonkie's Chunks into a Turbopuffer database."""

import importlib.util as importutil
import os
from typing import Any, Literal, Optional, Union
from uuid import NAMESPACE_OID, uuid5

from chonkie.embeddings import AutoEmbeddings, BaseEmbeddings
from chonkie.logger import get_logger
from chonkie.pipeline import handshake
from chonkie.types import Chunk

from .base import BaseHandshake
from .utils import generate_random_collection_name

logger = get_logger(__name__)


@handshake("turbopuffer")
class TurbopufferHandshake(BaseHandshake):
    """Turbopuffer Handshake to export Chonkie's Chunks into a Turbopuffer database."""

    def __init__(
        self,
        namespace: Optional[Any] = None,  # Will be tpuf.Namespace at runtime
        namespace_name: Union[str, Literal["random"]] = "random",
        embedding_model: Union[str, BaseEmbeddings] = "minishlab/potion-retrieval-32M",
        api_key: Optional[str] = None,
        region: Optional[str] = "gcp-us-central1",
    ) -> None:
        """Initialize the Turbopuffer Handshake.

        Args:
            namespace: The namespace to use.
            namespace_name: The name of the namespace to use, if the namespace is not provided.
            embedding_model: The embedding model to use.
            api_key: The API key to use.
            region: The region to use. Defaults to "gcp-us-central1".

        """
        super().__init__()

        # Check for the API Key
        api_key = api_key or os.getenv("TURBOPUFFER_API_KEY")
        if not api_key:
            raise ValueError(
                "Turbopuffer API key not found. Please provide an API key or set the TURBOPUFFER_API_KEY environment variable.",
            )

        try:
            import turbopuffer
        except ImportError as ie:
            raise ImportError(
                "Turbopuffer is not available. Please install it with `pip install turbopuffer`.",
            ) from ie

        # Setting the tpuf api key
        # self.tpuf.api_key = api_key  # type: ignore[attr-defined]
        self.tpuf = turbopuffer.Turbopuffer(api_key=api_key, region=region)

        # Get a list of namespaces
        namespaces = [ns.id for ns in self.tpuf.namespaces()]

        # If the namespace is not provided, generate a random one
        if namespace is None:
            if namespace_name == "random":
                # Generate a random namespace
                while True:
                    namespace_name = generate_random_collection_name()
                    if namespace_name not in namespaces:
                        break
            self.namespace = self.tpuf.namespace(namespace_name)
            logger.info(f"Chonkie has created a new namespace: {self.namespace.id}")  # type: ignore[misc]
        else:
            self.namespace = namespace

        # Initialize the embedding model
        self.embedding_model = AutoEmbeddings.get_embeddings(embedding_model)

    @classmethod
    def _is_available(cls) -> bool:
        """Check if Turbopuffer is available."""
        return importutil.find_spec("turbopuffer") is not None

    def _generate_id(self, index: int, chunk: Chunk) -> str:
        """Generate a unique ID for the chunk."""
        return str(
            uuid5(
                NAMESPACE_OID,
                f"{self.namespace.id}::chunk-{index}:{chunk.text}",  # type: ignore[misc]
            ),
        )

    def write(self, chunks: Union[Chunk, list[Chunk]]) -> None:
        """Write the chunks to the Turbopuffer database."""
        if isinstance(chunks, Chunk):
            chunks = [chunks]

        logger.debug(f"Writing {len(chunks)} chunks to Turbopuffer namespace: {self.namespace.id}")  # type: ignore[misc]
        # Embed the chunks
        ids = [self._generate_id(index, chunk) for (index, chunk) in enumerate(chunks)]
        texts = [chunk.text for chunk in chunks]
        embeddings = [embedding.tolist() for embedding in self.embedding_model.embed_batch(texts)]
        start_indices = [chunk.start_index for chunk in chunks]
        end_indices = [chunk.end_index for chunk in chunks]
        token_counts = [chunk.token_count for chunk in chunks]

        # Write the chunks to the database
        self.namespace.write(  # type: ignore[attr-defined]
            upsert_columns={
                "id": ids,
                "vector": embeddings,
                "text": texts,
                "start_index": start_indices,
                "end_index": end_indices,
                "token_count": token_counts,
            },
            distance_metric="cosine_distance",
        )

        logger.info(
            f"Chonkie has written {len(chunks)} chunks to the namespace: {self.namespace.id}",  # type: ignore[misc]
        )

    def __repr__(self) -> str:
        """Return the representation of the Turbopuffer Handshake."""
        return f"TurbopufferHandshake(namespace={self.namespace.id})"  # type: ignore[misc]

    def search(
        self,
        query: Optional[str] = None,
        embedding: Optional[list[float]] = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Search the Turbopuffer namespace for similar chunks.

        Args:
            query: The query string to search for. If provided, `embedding` is ignored.
            embedding: The embedding vector to search for.
            limit: The maximum number of results to return.

        Returns:
            list[dict[str, Any]]: A list of dictionaries containing the matching chunks and their metadata.

        """
        if query is None and embedding is None:
            raise ValueError("Either 'query' or 'embedding' must be provided.")

        if query:
            embedding = self.embedding_model.embed(query).tolist()

        # Use include_attributes to request extra fields
        results = self.namespace.query(
            rank_by=("vector", "ANN", embedding),  # type: ignore[arg-type]
            top_k=limit,
            include_attributes=["text", "start_index", "end_index", "token_count"],
        )
        assert results.rows is not None
        return [
            {
                "id": result["id"],
                "score": 1.0 - result["$dist"],
                "token_count": result["token_count"],
                "text": result["text"],
                "start_index": result["start_index"],
                "end_index": result["end_index"],
            }
            for result in results.rows
        ]
