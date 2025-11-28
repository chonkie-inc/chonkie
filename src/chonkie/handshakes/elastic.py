"""Elasticsearch Handshake to export Chonkie's Chunks into an Elasticsearch index."""

import importlib.util
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
from chonkie.logger import get_logger
from chonkie.pipeline import handshake
from chonkie.types import Chunk

from .base import BaseHandshake
from .utils import generate_random_collection_name

logger = get_logger(__name__)

if TYPE_CHECKING:
    from elasticsearch import Elasticsearch
    from elasticsearch.helpers import bulk


@handshake("elastic")
class ElasticHandshake(BaseHandshake):
    """Elasticsearch Handshake to export Chonkie's Chunks into an Elasticsearch index.

    This handshake connects to an Elasticsearch instance, creates an index with the
    appropriate vector mapping, and ingests chunks for similarity search.

    Args:
        client: Optional[Elasticsearch]: An existing Elasticsearch client instance. If not provided, one will be created.
        index_name: Union[str, Literal["random"]]: The name of the index to use. If "random", a unique name is generated.
        embedding_model: Union[str, BaseEmbeddings]: The embedding model to use for vectorizing chunks.
        hosts: Optional[Union[str, List[str]]]: URL(s) of the Elasticsearch instance(s).
        cloud_id: Optional[str]: The Cloud ID for connecting to an Elastic Cloud deployment.
        api_key: Optional[str]: The API key for authenticating with an Elastic Cloud deployment.
        **kwargs: Additional keyword arguments to pass to the Elasticsearch client constructor.

    """

    def __init__(
        self,
        client: Optional["Elasticsearch"] = None,
        index_name: Union[str, Literal["random"]] = "random",
        embedding_model: Union[str, BaseEmbeddings] = "minishlab/potion-retrieval-32M",
        hosts: Optional[Union[str, List[str]]] = None,
        cloud_id: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        """Initialize the Elasticsearch Handshake."""
        super().__init__()
        self._import_dependencies()

        # 1. Initialize the Elasticsearch client
        if client:
            self.client = client
        elif cloud_id and api_key:
            self.client = Elasticsearch(cloud_id=cloud_id, api_key=api_key, **kwargs) # type: ignore
        elif hosts:
            self.client = Elasticsearch(hosts=hosts, api_key=api_key, **kwargs) # type: ignore
        else:
            # Default to a standard local client if no other connection info is provided
            self.client = Elasticsearch("http://localhost:9200", **kwargs) # type: ignore

        # 2. Initialize the embedding model
        if isinstance(embedding_model, str):
            self.embedding_model = AutoEmbeddings.get_embeddings(embedding_model)
        else:
            self.embedding_model = embedding_model
        self.dimension = self.embedding_model.dimension

        # 3. Handle the index name
        if index_name == "random":
            while True:
                self.index_name = generate_random_collection_name()
                if not self.client.indices.exists(index=self.index_name):
                    break
            logger.info(f"Chonkie will create a new index in Elasticsearch: {self.index_name}")
        else:
            self.index_name = index_name

        # 4. Create the index with the correct vector mapping if it doesn't exist
        if not self.client.indices.exists(index=self.index_name):
            mapping = {
                "properties": {
                    "embedding": {"type": "dense_vector", "dims": self.dimension},
                    "text": {"type": "text"},
                    "start_index": {"type": "integer"},
                    "end_index": {"type": "integer"},
                    "token_count": {"type": "integer"},
                }
            }
            self.client.indices.create(index=self.index_name, mappings=mapping)
            logger.info(f"Index '{self.index_name}' created with vector mapping.")

    def _is_available(self) -> bool:
        """Check if the dependencies are installed."""
        return importlib.util.find_spec("elasticsearch") is not None

    def _import_dependencies(self) -> None:
        """Lazy import the dependencies."""
        if self._is_available():
            global Elasticsearch, bulk
            from elasticsearch import Elasticsearch
            from elasticsearch.helpers import bulk
        else:
            raise ImportError(
                "Elasticsearch is not installed. "
                + "Please install it with `pip install chonkie[elastic]`."
            )

    def _generate_id(self, index: int, chunk: Chunk) -> str:
        """Generate a unique id for the chunk."""
        return str(uuid5(NAMESPACE_OID, f"{self.index_name}::chunk-{index}:{chunk.text}"))

    def _create_bulk_actions(self, chunks: List[Chunk]) -> List[Dict[str, Any]]:
        """Generate a list of actions for the Elasticsearch bulk API."""
        actions = []
        # Get all embeddings in a single batch call for efficiency
        embeddings = self.embedding_model.embed_batch([chunk.text for chunk in chunks])

        for i, chunk in enumerate(chunks):
            actions.append({
                "_index": self.index_name,
                "_id": self._generate_id(i, chunk),
                "_source": {
                    "text": chunk.text,
                    "embedding": embeddings[i],
                    "start_index": chunk.start_index,
                    "end_index": chunk.end_index,
                    "token_count": chunk.token_count,
                },
            })
        return actions

    def write(self, chunks: Union[Chunk, List[Chunk]]) -> None:
        """Write the chunks to the Elasticsearch index using the bulk API."""
        if isinstance(chunks, Chunk):
            chunks = [chunks]

        actions = self._create_bulk_actions(chunks)

        # Use the bulk helper to efficiently write the documents
        success, errors = bulk(self.client, actions, raise_on_error=False)

        if errors:
            logger.warning(f"Encountered {len(errors)} errors during bulk indexing.") # type: ignore
            # Optionally log the first few errors for debugging
            for i, error in enumerate(errors[:5]): # type: ignore
                logger.error(f"Error {i+1}: {error}")

        logger.info(f"Chonkie wrote {success} chunks to Elasticsearch index: {self.index_name}")

    def __repr__(self) -> str:
        """Return the string representation of the ElasticHandshake."""
        return f"ElasticHandshake(index_name={self.index_name})"

    def search(
        self,
        query: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve the top_k most similar chunks to the query using KNN search.

        Args:
            query: The query string to search for. If provided, `embedding` is ignored.
            embedding: The embedding vector to search for.
            limit: The number of top similar chunks to retrieve.

        Returns:
            A list of dictionaries, each containing a similar chunk, its metadata, and similarity score.

        """
        if embedding is None and query is None:
            raise ValueError("Either 'query' or 'embedding' must be provided.")
        if query:
            embedding = self.embedding_model.embed(query).tolist()

        knn_query = {
            "field": "embedding",
            "query_vector": embedding,
            "k": limit,
            "num_candidates": 100,  # A standard parameter for approximate nearest neighbor search
        }

        results = self.client.search(index=self.index_name, knn=knn_query, size=limit)

        # Format the results to match the unified output of other handshakes
        matches = []
        for hit in results["hits"]["hits"]:
            source = hit["_source"]
            matches.append({
                "id": hit["_id"],
                "score": hit["_score"],
                "text": source.get("text"),
                "start_index": source.get("start_index"),
                "end_index": source.get("end_index"),
                "token_count": source.get("token_count"),
            })
        return matches