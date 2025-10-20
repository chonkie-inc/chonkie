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

from chonkie.embeddings import AutoEmbeddings, BaseEmbeddings
from chonkie.types import Chunk

from .base import BaseHandshake
from .utils import generate_random_collection_name

if TYPE_CHECKING:
    import numpy as np
    from pymilvus import (
        Collection,
        CollectionSchema,
        DataType,
        FieldSchema,
        connections,
        utility,
    )
    from pymilvus.exceptions import ConnectionNotExistException

class MilvusHandshake(BaseHandshake):
    """Milvus Handshake to export Chonkie's Chunks into a Milvus collection.

    This handshake connects to a Milvus instance, creates a collection with a
    defined schema, and ingests chunks for similarity search.

    Args:
        collection_name: The name of the collection to use. If "random", a unique name is generated.
        embedding_model: The embedding model to use for vectorizing chunks.
        uri: The URI to connect to Milvus (e.g., "http://localhost:19530").
        host: The host of the Milvus instance. Defaults to "localhost".
        port: The port of the Milvus instance. Defaults to "19530".
        alias: The connection alias to use. Defaults to "default".
        **kwargs: Additional keyword arguments for future use.
        
    """

    def __init__(
        self,
        client: Optional[Any] = None,
        collection_name: Union[str, Literal["random"]] = "random",
        embedding_model: Union[str, BaseEmbeddings] = "minishlab/potion-retrieval-32M",
        uri: Optional[str] = None,
        host: str = "localhost",
        port: str = "19530",
        alias: str = "default",
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Milvus Handshake."""
        super().__init__()
        self._import_dependencies()

        # 1. Establish connection
        if client is not None:
            self.client = client
        else : 
            self.client = connections.connect(alias=self.alias, uri=uri, api_key=api_key, **kwargs)
        self.alias = alias
        try:
            # Check if a connection with this alias already exists
            connections.get_connection_addr(self.alias) # type: ignore
        except ConnectionNotExistException:
            # If not, create a new one
            if uri:
                connections.connect(alias=self.alias, uri=uri, api_key=api_key, **kwargs) # type: ignore
            else:
                connections.connect(alias=self.alias, host=host, port=port, api_key=api_key, **kwargs) # type: ignore

        # 2. Initialize the embedding model
        if isinstance(embedding_model, str):
            self.embedding_model = AutoEmbeddings.get_embeddings(embedding_model)
        else:
            self.embedding_model = embedding_model
        self.dimension = self.embedding_model.dimension

        # 3. Handle collection name and schema
        if collection_name == "random":
            while True:
                self.collection_name = generate_random_collection_name()
                if not utility.has_collection(self.collection_name, using=self.alias): # type: ignore
                    break
        else:
            self.collection_name = collection_name

        if not utility.has_collection(self.collection_name, using=self.alias): # type: ignore
            self._create_collection_with_schema()

        self.collection = Collection(self.collection_name, using=self.alias) # type: ignore
        self.collection.load()

    def _import_dependencies(self) -> None:
        """Lazy import the dependencies."""
        if self._is_available():
            global Collection, CollectionSchema, DataType, FieldSchema, connections, utility, np, ConnectionNotExistException

            import numpy as np
            from pymilvus import (
                Collection,
                CollectionSchema,
                DataType,
                FieldSchema,
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
        collection = Collection(self.collection_name, schema, using=self.alias) # type: ignore
        print(f"🦛 Chonkie created a new collection in Milvus: {self.collection_name}")

        # Create a default index for the vector field
        index_params = {
            "metric_type": "L2",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 200},
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        print("✅ Created default HNSW index on 'embedding' field.")

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

        print(
            f"🦛 Chonkie wrote {mutation_result.insert_count} chunks to Milvus collection: {self.collection_name}"
        )

    def __repr__(self) -> str:
        """Return the string representation of the MilvusHandshake."""
        return f"MilvusHandshake(collection_name={self.collection_name}, alias={self.alias})"

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