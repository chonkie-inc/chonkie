"""Pinecone Handshake to export Chonkie's Chunks into a Pinecone index."""

import importlib.util as importutil
import os
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Optional,
    Union,
)
from uuid import NAMESPACE_OID, uuid5

from chonkie.embeddings import AutoEmbeddings, BaseEmbeddings
from chonkie.logger import get_logger
from chonkie.pipeline import handshake
from chonkie.types import Chunk

from .base import BaseHandshake
from .utils import generate_random_collection_name

logger = get_logger(__name__)

if TYPE_CHECKING:
    from pinecone import Pinecone, ServerlessSpec


@handshake("pinecone")
class PineconeHandshake(BaseHandshake):
    """Pinecone Handshake to export Chonkie's Chunks into a Pinecone index.

    This handshake is experimental and may change in the future. Not all Chonkie features are supported yet.

    Args:
        client: Optional[pinecone.Pinecone]: The Pinecone client to use. If None, will create a new client.
        api_key: str: The Pinecone API key.
        index_name: Union[str, Literal["random"]]: The name of the index to use.
        spec: Optional[pinecone.ServerlessSpec]: The pinecone ServerlessSpec to use for the index. If not provided, will use the default spec.
        embedding_model: Union[str, BaseEmbeddings]: The embedding model to use.
        embed: Optional[dict[str, str]]: The Pinecone integrated embedding model to use. If not provided, will use `embedding_model` to create a new index.
        **kwargs: Additional keyword arguments to pass to the Pinecone client.

    """

    def __init__(
        self,
        client: Optional["Pinecone"] = None,
        api_key: Optional[str] = None,
        index_name: Union[str, Literal["random"]] = "random",
        spec: Optional["ServerlessSpec"] = None,
        embedding_model: Union[str, BaseEmbeddings] = "minishlab/potion-retrieval-32M",
        embed: Optional[dict[str, str]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Pinecone handshake.

        Args:
            client: Optional[pinecone.Pinecone]: The Pinecone client to use. If None, will create a new client.
            api_key: The Pinecone API key.
            index_name: The name of the index to use, or "random" for auto-generated name.
            spec: Optional[pinecone.ServerlessSpec]: The spec to use for the index. If not provided, will use the default spec.
            embedding_model: The embedding model to use, either as string or BaseEmbeddings instance.
            embed: The Pinecone integrated embedding model to use. If not provided, will use `embedding_model` to create a new index.
            **kwargs: Additional keyword arguments to pass to the Pinecone client.

        """
        super().__init__()

        try:
            import pinecone
        except ImportError as ie:
            raise ImportError(
                "Pinecone is not installed. Please install it with `pip install chonkie[pinecone]`.",
            ) from ie

        if client is not None:
            self.client = client
        else:
            api_key = api_key or os.getenv("PINECONE_API_KEY")
            if api_key is None:
                raise ValueError(
                    "Pinecone API key is not set. Please provide it as an argument or set the PINECONE_API_KEY environment variable.",
                )

            self.client = pinecone.Pinecone(api_key=api_key, source_tag="chonkie")

        self.embed: Optional[dict[str, str]] = embed
        if embed is not None:
            self.embedding_model = None
        elif isinstance(embedding_model, str):
            self.embedding_model = AutoEmbeddings.get_embeddings(embedding_model)
            self.dimension = self.embedding_model.dimension
            self.metric = "cosine"
        elif isinstance(embedding_model, BaseEmbeddings):
            self.embedding_model = embedding_model
            self.dimension = self.embedding_model.dimension
            self.metric = "cosine"
        else:
            raise ValueError(f"Invalid embedding model: {embedding_model}")

        if index_name == "random":
            while True:
                self.index_name = generate_random_collection_name()
                if not self.client.has_index(self.index_name):
                    break
            logger.info(f"Chonkie created a new index in Pinecone: {self.index_name}")
        else:
            self.index_name = index_name

        # set default value for specs field if not present
        self.spec = spec or pinecone.ServerlessSpec(cloud="aws", region="us-east-1")

        # Create the index if it doesn't exist
        if not self.client.has_index(self.index_name):
            if self.embed is not None:
                self.client.create_index(  # type: ignore[call-arg]
                    name=self.index_name,
                    spec=self.spec,
                    embed=self.embed,
                    **kwargs,
                )
            else:
                self.client.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=self.spec,
                    **kwargs,
                )
        self.index = self.client.Index(self.index_name)

    @classmethod
    def _is_available(cls) -> bool:
        return importutil.find_spec("pinecone") is not None

    def _generate_id(self, index: int, chunk: Chunk) -> str:
        return str(uuid5(NAMESPACE_OID, f"{self.index_name}::chunk-{index}:{chunk.text}"))

    def _generate_metadata(self, chunk: Chunk) -> dict:
        return {
            "text": chunk.text,
            "start_index": chunk.start_index,
            "end_index": chunk.end_index,
            "token_count": chunk.token_count,
        }

    def _get_vectors(
        self,
        chunks: Union[Chunk, list[Chunk]],
    ) -> list[tuple[str, list[float], dict[str, Any]]]:
        """Generate vectors for the chunks.

        Args:
            chunks: A single Chunk or sequence of Chunks to generate vectors for.

        Returns:
            list[tuple[str, list[float], dict[str, Any]]]: A list of tuples containing the vector ID, embedding, and metadata.

        """
        if isinstance(chunks, Chunk):
            chunks = [chunks]
        vectors = []
        assert self.embedding_model, "Embedding model is not set."
        for index, chunk in enumerate(chunks):
            # Handle both numpy arrays and lists
            embedding = self.embedding_model.embed(chunk.text)
            if hasattr(embedding, "tolist"):
                embedding_list: list[float] = embedding.tolist()
            else:
                embedding_list = embedding  # type: ignore[assignment]
            vectors.append((
                self._generate_id(index, chunk),
                embedding_list,
                self._generate_metadata(chunk),
            ))
        return vectors

    def write(self, chunks: Union[Chunk, list[Chunk]]) -> None:
        """Write chunks to the Pinecone index.

        Args:
            chunks: A single Chunk or sequence of Chunks to write to the index.

        Returns:
            None

        """
        if isinstance(chunks, Chunk):
            chunks = [chunks]
        logger.debug(f"Writing {len(chunks)} chunks to Pinecone index: {self.index_name}")
        vectors = self._get_vectors(chunks)
        self.index.upsert(vectors)
        logger.info(f"Chonkie wrote {len(chunks)} chunks to Pinecone index: {self.index_name}")

    def __repr__(self) -> str:
        """Return a string representation of the PineconeHandshake instance.

        Returns:
            str: A string representation containing the index name.

        """
        return f"PineconeHandshake(index_name={self.index_name})"

    def search(
        self,
        query: Optional[str] = None,
        embedding: Optional[list[float]] = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Search the Pinecone index for similar chunks.

        Args:
            query: The query string to search for. If provided, `embedding` is ignored.
            embedding: The embedding vector to search for.
            limit: The maximum number of results to return.

        Returns:
            list[dict[str, Any]]: A list of dictionaries containing the matching chunks and their metadata.

        """
        logger.debug(f"Searching Pinecone index: {self.index_name} with limit={limit}")
        if self.embed is not None:
            # Use Pinecone's integrated embedding model
            results = self.index.query(query=query, top_k=limit, include_metadata=True)
        elif query is None and embedding is None:
            raise ValueError(
                "Query string or embedding must be provided when using a custom embedding model.",
            )
        elif query is not None:
            # warning if both query and embedding are provided, query is used
            if embedding is not None:
                logger.warning("Both query and embedding provided. Using query.")
            assert self.embedding_model, "Embedding model is not set."
            # Use custom embedding model to embed the query
            embedding = self.embedding_model.embed(query).tolist()
        results = self.index.query(vector=embedding, top_k=limit, include_metadata=True)

        matches = []
        for match in results.get("matches", []):  # type: ignore[union-attr,call-arg,arg-type]
            matches.append({
                "id": match.get("id"),  # type: ignore[union-attr]
                "score": match.get("score"),  # type: ignore[union-attr]
                **match.get("metadata", {}),  # type: ignore[union-attr,arg-type]
            })
        logger.info(f"Search complete: found {len(matches)} matching chunks")
        return matches
