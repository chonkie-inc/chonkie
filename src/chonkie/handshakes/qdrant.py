"""Qdrant Handshake to export Chonkie's Chunks into a Qdrant collection."""

import importlib.util as importutil
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Union,
)
from uuid import NAMESPACE_OID, uuid5

from chonkie.embeddings import AutoEmbeddings, BaseEmbeddings
from chonkie.types import Chunk

from .base import BaseHandshake
from .utils import generate_random_collection_name

if TYPE_CHECKING:
    import qdrant_client
    from qdrant_client.http.models import PointStruct

    try:
        from qdrant_client.http.models import Distance, VectorParams
    except ImportError:

        class VectorParams:  # type: ignore
            """Stub class for qdrant_client VectorParams when not available."""

            pass

        class Distance:  # type: ignore
            """Stub class for qdrant_client Distance when not available."""

            pass


class QdrantHandshake(BaseHandshake):
    """Qdrant Handshake to export Chonkie's Chunks into a Qdrant collection.

    This handshake is experimental and may change in the future. Not all Chonkie features are supported yet.

    Args:
        client: Optional[qdrant_client.QdrantClient]: The Qdrant client to use.
        collection_name: Union[str, Literal["random"]]: The name of the collection to use.
        embedding_model: Union[str, BaseEmbeddings]: The embedding model to use.
        url: Optional[str]: The URL to the Qdrant Server.
        api_key: Optional[str]: The API key to the Qdrant Server. Only needed for Qdrant Cloud.
        path: Optional[str]: The path to the Qdrant collection locally. If not provided, will create an ephemeral collection.

    """

    def __init__(
        self,
        client: Optional["qdrant_client.QdrantClient"] = None,
        collection_name: Union[str, Literal["random"]] = "random",
        embedding_model: Union[str, BaseEmbeddings] = "minishlab/potion-retrieval-32M",
        url: Optional[str] = None,
        path: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        """Initialize the Qdrant Handshake.

        Args:
            client: Optional[qdrant_client.QdrantClient]: The Qdrant client to use.
            collection_name: Union[str, Literal["random"]]: The name of the collection to use.
            embedding_model: Union[str, BaseEmbeddings]: The embedding model to use.
            url: Optional[str]: The URL to the Qdrant Server.
            path: Optional[str]: The path to the Qdrant collection locally. If not provided, will create an ephemeral collection.
            api_key: Optional[str]: The API key to the Qdrant Server. Only needed for Qdrant Cloud.
            **kwargs: Additional keyword arguments to pass to the Qdrant client.

        """
        super().__init__()

        # Lazy importing the dependencies
        self._import_dependencies()

        # Initialize the Qdrant client
        if client is None:
            if url is not None and api_key is not None:
                self.client = qdrant_client.QdrantClient(
                    url=url,
                    api_key=api_key,
                    **kwargs,  # type: ignore[arg-type]
                )
            elif url is not None:
                self.client = qdrant_client.QdrantClient(url=url, **kwargs)  # type: ignore[arg-type]
            elif path is not None:
                self.client = qdrant_client.QdrantClient(path=path, **kwargs)  # type: ignore[arg-type]
            else:
                # If no client is provided, create an ephemeral collection
                self.client = qdrant_client.QdrantClient(":memory:", **kwargs)  # type: ignore[arg-type]
        else:
            self.client = client

        # Initialize the embedding model
        if isinstance(embedding_model, str):
            self.embedding_model = AutoEmbeddings.get_embeddings(embedding_model)
        else:
            self.embedding_model = embedding_model

        self.dimension = self.embedding_model.dimension

        # Initialize the collection
        if collection_name == "random":
            while True:
                self.collection_name = generate_random_collection_name()
                # Check if the collection exists or not?
                if not self.client.collection_exists(self.collection_name):
                    break
                else:
                    pass
            print(
                f"🦛 Chonkie created a new collection in Qdrant: {self.collection_name}"
            )
        else:
            self.collection_name = collection_name

        # Create the collection, if it doesn't exist
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.dimension, distance=Distance.COSINE
                ),
            )

    def _is_available(self) -> bool:
        """Check if the dependencies are installed."""
        return importutil.find_spec("qdrant_client") is not None

    def _import_dependencies(self) -> None:
        """Lazy import the dependencies."""
        if self._is_available():
            global qdrant_client, PointStruct, VectorParams, Distance
            import qdrant_client
            from qdrant_client.http.models import PointStruct
            from qdrant_client.models import Distance, VectorParams
        else:
            raise ImportError(
                "Qdrant is not installed. "
                + "Please install it with `pip install chonkie[qdrant]`."
            )

    def _generate_id(self, index: int, chunk: Chunk) -> str:
        """Generate a unique id for the chunk."""
        return str(
            uuid5(NAMESPACE_OID, f"{self.collection_name}::chunk-{index}:{chunk.text}")
        )

    def _generate_payload(self, chunk: Chunk) -> dict:
        """Generate the payload for the chunk."""
        return {
            "text": chunk.text,
            "start_index": chunk.start_index,
            "end_index": chunk.end_index,
            "token_count": chunk.token_count,
        }

    def _get_points(self, chunks: Union[Chunk, List[Chunk]]) -> List["PointStruct"]:
        """Get the points from the chunks."""
        # Normalize input to always be a sequence
        if isinstance(chunks, Chunk):
            chunks = [chunks]

        points = []
        for index, chunk in enumerate(chunks):
            points.append(
                PointStruct(
                    id=self._generate_id(index, chunk),
                    vector=self.embedding_model.embed(chunk.text).tolist(),  # type: ignore[arg-type] # Since this passes a numpy array, we need to convert it to a list
                    payload=self._generate_payload(chunk),
                )
            )
        return points

    def write(self, chunks: Union[Chunk, List[Chunk]]) -> None:
        """Write the chunks to the collection."""
        if isinstance(chunks, Chunk):
            chunks = [chunks]

        points = self._get_points(chunks)

        # Write the points to the collection
        self.client.upsert(
            collection_name=self.collection_name, points=points, wait=True
        )

        print(
            f"🦛 Chonkie wrote {len(chunks)} chunks to Qdrant collection: {self.collection_name}"
        )

    def __repr__(self) -> str:
        """Return the string representation of the QdrantHandshake."""
        return f"QdrantHandshake(collection_name={self.collection_name})"

    def search(
        self,
        query: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve the top_k most similar chunks to the query.

        Args:
            query: Optional[str]: The query string to search for.
            embedding: Optional[List[float]]: The embedding vector to search for. If provided, `query` is ignored.
            limit: int: The number of top similar chunks to retrieve.

        Returns:
            List[Dict[str, Any]]: The list of most similar chunks with their metadata.

        """
        if embedding is None and query is None:
            raise ValueError("Either query or embedding must be provided")
        if query is not None:
            embedding = self.embedding_model.embed(query).tolist()

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=embedding,
            limit=limit,
            with_payload=True,
        )
        return [
            {"id": result["id"], "score": result["score"], **result["payload"]}
            for result in results.dict()["points"]
        ]
