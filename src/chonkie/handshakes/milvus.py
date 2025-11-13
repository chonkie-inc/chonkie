"""Milvus Handshake to export Chonkie's Chunks into a Milvus collection."""

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

import numpy as np

from chonkie.embeddings import AutoEmbeddings, BaseEmbeddings
from chonkie.logger import get_logger
from chonkie.types import Chunk

from .base import BaseHandshake
from .utils import generate_random_collection_name

logger = get_logger(__name__)

if TYPE_CHECKING:
    from pymilvus import (
        Collection,
        CollectionSchema,
        DataType,
        FieldSchema,
    )

class MilvusHandshake(BaseHandshake):
    """Milvus Handshake to export Chonkie's Chunks into a Milvus collection.

    This handshake connects to a Milvus instance, creates a collection with a
    defined schema, and ingests chunks for similarity search.

    Args:
        client: An optional pre-initialized MilvusClient instance.
        uri: The URI to connect to Milvus (e.g., "http://localhost:19530").
        collection_name: The name of the collection to use. If "random", a unique name is generated.
        embedding_model: The embedding model to use for vectorizing chunks.
        host: The host of the Milvus instance. Defaults to "localhost".
        port: The port of the Milvus instance. Defaults to "19530".
        user: The username to connect to Milvus. Defaults to "".
        api_key: The API key to connect to Milvus. Defaults to "".
        **kwargs: Additional keyword arguments for future use.
        
    """

    def __init__(
        self,
        client: Optional[Any] = None, 
        uri: Optional[str] = None,
        collection_name: Union[str, Literal["random"]] = "random",
        embedding_model: Union[str, BaseEmbeddings] = "minishlab/potion-retrieval-32M",
        host: str = "localhost",
        port: str = "19530",
        user: Optional[str] = "",
        api_key: Optional[str] = "",
        alias: str = "default",
        **kwargs: Any,
    ) -> None:
        """Initialize the Milvus Handshake.

        Args:
            client: An optional pre-initialized MilvusClient instance.
            uri: The URI to connect to Milvus (e.g., "http://localhost:19530").
            collection_name: The name of the collection to use. If "random", a unique name is generated.
            embedding_model: The embedding model to use for vectorizing chunks.
            host: The host of the Milvus instance. Defaults to "localhost".
            port: The port of the Milvus instance. Defaults to "19530".
            user: The username to connect to Milvus. Defaults to "".
            api_key: The API key to connect to Milvus. Defaults to "".
            alias: The alias to use for the Milvus connection. Defaults to "default".
            **kwargs: Additional keyword arguments for future use.

        """
        self._import_dependencies()
        super().__init__()
        self.alias = alias

        # 1. Establish connection using the ORM's global connection manager
        if client is not None:
            self.client = client
        else:
            self.client = MilvusClient(uri=uri, host=host, port=port, user=user, password=api_key, alias=alias, **kwargs) # type: ignore
        # Always connect using ORM before any collection operations
        try:
            connections.connect(uri=uri, host=host, port=port, user=user, password=api_key, alias=alias, **kwargs) # type: ignore
        except Exception as e:
            logger.warning(f"Could not connect with ORM connections: {e}")
        # 3. Initialize the embedding model
        if isinstance(embedding_model, str):
            self.embedding_model = AutoEmbeddings.get_embeddings(embedding_model)
        else:
            self.embedding_model = embedding_model
        self.dimension = self.embedding_model.dimension

        # 4. Handle collection name and schema

        if collection_name == "random":
            while True:
                self.collection_name = generate_random_collection_name(sep="_")
                # Pass alias explicitly to utility.has_collection
                if not self.client.has_collection(self.collection_name): # type: ignore
                    break
        else:
            self.collection_name = collection_name

        # Pass alias explicitly to utility.has_collection
        if not self.client.has_collection(self.collection_name):
            self._create_collection_with_schema()

        self.collection = Collection(self.collection_name) # type: ignore
        self.collection.load()

    def _import_dependencies(self) -> None:
        """Lazy import the dependencies."""
        if self._is_available():
            global Collection, CollectionSchema, DataType, FieldSchema, connections, utility, ConnectionNotExistException, MilvusClient
            from pymilvus import (
                Collection,
                CollectionSchema,
                DataType,
                FieldSchema,
                MilvusClient,
                connections,
                utility,
            )
            from pymilvus.exceptions import ConnectionNotExistException
        else:
            raise ImportError(
                "Milvus is not installed. "
                + "Please install it with `pip install chonkie[milvus]`."
            )

    def _is_available(self) -> bool:
        """Check if the dependencies are installed."""
        return importlib.util.find_spec("pymilvus") is not None

    def _create_collection_with_schema(self) -> None:
        """Create a new collection with a predefined schema and index."""
        # Define fields: pk, text, metadata, and the vector embedding
        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True), # type: ignore
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65_535), # type: ignore
            FieldSchema(name="start_index", dtype=DataType.INT64), # type: ignore
            FieldSchema(name="end_index", dtype=DataType.INT64), # type: ignore
            FieldSchema(name="token_count", dtype=DataType.INT64), # type: ignore
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension), # type: ignore
        ]
        schema = CollectionSchema(fields, description="Chonkie Handshake Collection") # type: ignore
        collection = Collection(self.collection_name, schema) # type: ignore
        logger.info(f"Chonkie created a new collection in Milvus: {self.collection_name}")

        # Create a default index for the vector field
        index_params = {
            "metric_type": "L2",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 200},
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        logger.info("Created default HNSW index on 'embedding' field.")

    def write(self, chunks: Union[Chunk, List[Chunk]]) -> None:
        """Write the chunks to the Milvus collection."""
        if isinstance(chunks, Chunk):
            chunks = [chunks]

        # Prepare data in columnar format for Milvus
        texts = [chunk.text for chunk in chunks]
        start_indices = [chunk.start_index for chunk in chunks]
        end_indices = [chunk.end_index for chunk in chunks]
        token_counts = [chunk.token_count for chunk in chunks]
        embeddings = self.embedding_model.embed_batch(texts)

        data_to_insert = [texts, start_indices, end_indices, token_counts, embeddings]

        mutation_result = self.collection.insert(data_to_insert)
        self.collection.flush() # Essential to make data searchable

        logger.info(f"Chonkie wrote {mutation_result.insert_count} chunks to Milvus collection: {self.collection_name}")

    def __repr__(self) -> str:
        """Return the string representation of the MilvusHandshake."""
        return f"MilvusHandshake(collection_name={self.collection_name})"

    def search(
        self,
        query: Optional[str] = None,
        embedding: Optional[Union[List[float], "np.ndarray"]] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve the top_k most similar chunks to the query."""
        if embedding is None and query is None:
            raise ValueError("Either 'query' or 'embedding' must be provided.")

        if query:
            query_embedding = self.embedding_model.embed(query)
            # Milvus expects a list of vectors for searching
            query_vectors = [query_embedding.tolist()]
        else:
            # Ensure embedding is in the correct format (list of lists)
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            # If it's a flat list, wrap it in another list
            if embedding and len(embedding) > 0 and isinstance(embedding[0], float):
                query_vectors = [embedding]
            else:
                query_vectors = embedding # type: ignore

        # Default search parameters for HNSW index
        search_params = {"metric_type": "L2", "params": {"ef": 64}}
        output_fields = ["text", "start_index", "end_index", "token_count"]

        results = self.collection.search(
            data=query_vectors,
            anns_field="embedding",
            param=search_params,
            limit=limit,
            output_fields=output_fields,
        )

        # Format results into a standardized list of dicts
        matches = []
        # Results are for the first query vector (index 0)
        for hit in results[0]:
            match_data = {
                "id": hit.id,
                "score": hit.distance, # Milvus uses 'distance', which is analogous to score
                **hit.entity,
            }
            matches.append(match_data)
        return matches