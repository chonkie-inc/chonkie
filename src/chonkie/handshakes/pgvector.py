"""Pgvector Handshake to export Chonkie's Chunks into a PostgreSQL database with pgvector using vecs."""

import importlib.util as importutil
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from uuid import NAMESPACE_OID, uuid5

from chonkie.embeddings import AutoEmbeddings, BaseEmbeddings
from chonkie.types import Chunk

from .base import BaseHandshake

if TYPE_CHECKING:
    import vecs

# Module-level variable that will be set by _import_dependencies
vecs = None


class PgvectorHandshake(BaseHandshake):
    """Pgvector Handshake to export Chonkie's Chunks into a PostgreSQL database with pgvector using vecs.
    
    This handshake allows storing Chonkie chunks in PostgreSQL with vector embeddings
    using the pgvector extension through the vecs client library from Supabase.

    Args:
        client: An existing vecs.Client instance. If provided, other connection parameters are ignored.
        host: PostgreSQL host. Defaults to "localhost".
        port: PostgreSQL port. Defaults to 5432.
        database: PostgreSQL database name. Defaults to "postgres".
        user: PostgreSQL username. Defaults to "postgres".
        password: PostgreSQL password. Defaults to "postgres".
        connection_string: Full PostgreSQL connection string. If provided, individual parameters are ignored.
        collection_name: The name of the collection to store chunks in.
        embedding_model: The embedding model to use for generating embeddings.
        vector_dimensions: The number of dimensions for the vector embeddings.

    """

    def __init__(
        self,
        client: Optional["vecs.Client"] = None,
        host: str = "localhost",
        port: int = 5432,
        database: str = "postgres",
        user: str = "postgres", 
        password: str = "postgres",
        connection_string: Optional[str] = None,
        collection_name: str = "chonkie_chunks",
        embedding_model: Union[str, BaseEmbeddings] = "minishlab/potion-retrieval-32M",
        vector_dimensions: Optional[int] = None,
    ) -> None:
        """Initialize the Pgvector Handshake.
        
        Args:
            client: An existing vecs.Client instance. If provided, other connection parameters are ignored.
            host: PostgreSQL host. Defaults to "localhost".
            port: PostgreSQL port. Defaults to 5432.
            database: PostgreSQL database name. Defaults to "postgres".
            user: PostgreSQL username. Defaults to "postgres".
            password: PostgreSQL password. Defaults to "postgres".
            connection_string: Full PostgreSQL connection string. If provided, individual parameters are ignored.
            collection_name: The name of the collection to store chunks in.
            embedding_model: The embedding model to use for generating embeddings.
            vector_dimensions: The number of dimensions for the vector embeddings.

        """         
        super().__init__()
        
        # Lazy importing the dependencies
        self._import_dependencies()

        # Initialize vecs client based on provided parameters
        if client is not None:
            # Use provided client directly
            self.client = client
        elif connection_string is not None:
            # Use provided connection string
            self.client = vecs.create_client(connection_string)
        else:
            # Build connection string from individual parameters
            conn_str = f"postgresql://{user}:{password}@{host}:{port}/{database}"
            self.client = vecs.create_client(conn_str)
        
        self.collection_name = collection_name
        
        # Initialize the embedding model
        if isinstance(embedding_model, str):
            self.embedding_model = AutoEmbeddings.get_embeddings(embedding_model)
        elif isinstance(embedding_model, BaseEmbeddings):
            self.embedding_model = embedding_model
        else:
            raise ValueError("embedding_model must be a string or a BaseEmbeddings instance.")

        # Determine vector dimensions
        if vector_dimensions is None:
            # Try to get dimensions from embedding model's dimension property first
            if hasattr(self.embedding_model, 'dimension') and self.embedding_model.dimension is not None:
                self.vector_dimensions = self.embedding_model.dimension
            else:
                # Fall back to test embedding if dimension property is not available or is None
                test_embedding = self.embedding_model.embed("test")
                self.vector_dimensions = len(test_embedding)
        else:
            self.vector_dimensions = vector_dimensions

        # Get or create the collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            dimension=self.vector_dimensions
        )

    def _is_available(self) -> bool:
        """Check if the dependencies are available."""
        return importutil.find_spec("vecs") is not None

    def _import_dependencies(self) -> None:
        """Lazy import the dependencies."""
        if not self._is_available():
            raise ImportError(
                "vecs is not installed. "
                "Please install it with `pip install chonkie[pgvector]`."
            )
        
        global vecs
        import vecs

    def _generate_id(self, index: int, chunk: Chunk) -> str:
        """Generate a unique ID for the chunk."""
        return str(
            uuid5(
                NAMESPACE_OID, 
                f"{self.collection_name}::chunk-{index}:{chunk.text}"
            )
        )

    def _generate_metadata(self, chunk: Chunk) -> Dict[str, Any]:
        """Generate metadata for the chunk."""
        metadata = {
            "text": chunk.text,
            "start_index": chunk.start_index,
            "end_index": chunk.end_index,
            "token_count": chunk.token_count,
            "chunk_type": type(chunk).__name__,
        }
        
        # Add chunk-specific metadata
        if hasattr(chunk, "sentences") and chunk.sentences:
            metadata["sentence_count"] = len(chunk.sentences)
        
        if hasattr(chunk, "words") and chunk.words:
            metadata["word_count"] = len(chunk.words)
            
        if hasattr(chunk, "language") and chunk.language:
            metadata["language"] = chunk.language
            
        return metadata

    def write(self, chunks: Union[Chunk, List[Chunk]]) -> List[str]:
        """Write chunks to the PostgreSQL database using vecs.
        
        Args:
            chunks: A single chunk or sequence of chunks to write.
            
        Returns:
            List[str]: List of IDs of the inserted chunks.

        """
        if isinstance(chunks, Chunk):
            chunks = [chunks]

        records = []
        chunk_ids = []
        
        for index, chunk in enumerate(chunks):
            # Generate ID and metadata
            chunk_id = self._generate_id(index, chunk)
            metadata = self._generate_metadata(chunk)
            
            # Generate embedding
            embedding = self.embedding_model.embed(chunk.text)
            
            # Create record tuple for vecs (id, vector, metadata)
            records.append((chunk_id, embedding, metadata))
            chunk_ids.append(chunk_id)
        
        # Upsert all records at once
        self.collection.upsert(records=records)

        print(f"🦛 Chonkie wrote {len(chunks)} chunks to PostgreSQL collection: {self.collection_name}")
        return chunk_ids

    def search(
        self, 
        query: str, 
        limit: int = 5, 
        filters: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
        include_value: bool = True
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks using vector similarity.
        
        Args:
            query: The query text to search for.
            limit: Maximum number of results to return.
            filters: Optional metadata filters in vecs format (e.g., {"year": {"$eq": 2012}}).
            include_metadata: Whether to include metadata in results.
            include_value: Whether to include similarity scores in results.
            
        Returns:
            List[Dict[str, Any]]: List of similar chunks with metadata and scores.

        """
        # Generate embedding for the query
        query_embedding = self.embedding_model.embed(query)
        
        # Search using vecs
        results = self.collection.query(
            data=query_embedding,
            limit=limit,
            filters=filters,
            include_metadata=include_metadata,
            include_value=include_value
        )
        
        # Convert vecs results to our format
        formatted_results = []
        for result in results:
            # vecs returns tuples: (id, distance) or (id, distance, metadata)
            result_dict = {"id": result[0]}
            
            if include_value:
                result_dict["similarity"] = result[1]
            
            if include_metadata and len(result) > 2:
                metadata = result[2]
                result_dict.update(metadata)
            
            formatted_results.append(result_dict)
        
        return formatted_results

    def create_index(self, method: str = "hnsw", **index_params: Any) -> None:
        """Create a vector index for improved search performance.
        
        Args:
            method: Index method to use. Currently vecs supports various methods.
            **index_params: Additional parameters for the index.

        """
        # Create index using vecs (vecs handles the specifics)
        self.collection.create_index(method=method, **index_params)
        
        print(f"🦛 Created {method} index on collection: {self.collection_name}")

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        self.client.delete_collection(self.collection_name)
        print(f"🦛 Deleted collection: {self.collection_name}")

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        # vecs collections have various properties we can inspect
        return {
            "name": self.collection.name,
            "dimension": self.collection.dimension,
            # Add more collection info as available from vecs
        }

    def __repr__(self) -> str:
        """Return the string representation of the PgvectorHandshake."""
        return f"PgvectorHandshake(collection_name={self.collection_name}, vector_dimensions={self.vector_dimensions})"