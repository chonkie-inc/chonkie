"""MongoDB Handshake to export Chonkie's Chunks into a MongoDB collection."""

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
    import pymongo


class MongoDBHandshake(BaseHandshake):
    """MongoDB Handshake to export Chonkie's Chunks into a MongoDB collection.

    This handshake is experimental and may change in the future. Not all Chonkie features are supported yet.

    Args:
        client: The MongoDB client to use. If None, will create a new client.
        uri: The MongoDB connection URI.
        username: MongoDB username for authentication.
        password: MongoDB password for authentication.
        hostname: MongoDB host address.
        port: MongoDB port number.
        db_name: The name of the database or "random" for auto-generated name.
        collection_name: The name of the collection or "random" for auto-generated name.
        embedding_model: The embedding model identifier or instance.
        **kwargs: Additional keyword arguments for MongoDB client.

    """

    def __init__(
        self,
        client: Optional["pymongo.MongoClient"] = None,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        hostname: Optional[str] = None,
        port: Optional[Union[int, str]] = None,
        db_name: Union[str, Literal["random"]] = "random",
        collection_name: Union[str, Literal["random"]] = "random",
        embedding_model: Union[str, BaseEmbeddings] = "minishlab/potion-retrieval-32M",
        **kwargs: Any,
    ) -> None:
        """Initialize MongoDB Handshake with the specified connection parameters.

        Args:
            client: The MongoDB client to use. If None, will create a new client.
            uri: The MongoDB connection URI.
            username: MongoDB username for authentication.
            password: MongoDB password for authentication.
            hostname: MongoDB host address.
            port: MongoDB port number.
            db_name: The name of the database or "random" for auto-generated name.
            collection_name: The name of the collection or "random" for auto-generated name.
            embedding_model: The embedding model identifier or instance.
            **kwargs: Additional keyword arguments for MongoDB client.

        """
        super().__init__()
        self._import_dependencies()

        if client is not None:
            self.client = client
        else:
            # use uri
            if uri is None:
                # construct the uri
                if hostname is not None:
                    if username is not None and password is not None:
                        uri = f"mongodb://{username}:{password}@{hostname}"
                    else:
                        uri = f"mongodb://{hostname}"
                # use localhost
                else:
                    print("No hostname provided, using localhost instead")
                    port = str(port) if port is not None else "27017"
                    uri = f"mongodb://localhost:{port}"
                    # clear port
                    port = None

            self.client = pymongo.MongoClient(
                uri,
                port=int(port) if port is not None else None,
                **kwargs,
            )

        if db_name == "random":
            self.db_name = generate_random_collection_name()
            print(f"🦛 Chonkie created a new MongoDB database: {self.db_name}")
        else:
            self.db_name = db_name
        self.db = self.client[self.db_name]

        if collection_name == "random":
            self.collection_name = generate_random_collection_name()
            print(
                f"🦛 Chonkie created a new MongoDB collection: {self.collection_name}"
            )
        else:
            self.collection_name = collection_name
        self.collection = self.db[self.collection_name]

        # Embedding model
        if isinstance(embedding_model, str):
            self.embedding_model = AutoEmbeddings.get_embeddings(embedding_model)
        elif isinstance(embedding_model, BaseEmbeddings):
            self.embedding_model = embedding_model
        else:
            raise ValueError(f"Invalid embedding model: {embedding_model}")
        self.dimension = self.embedding_model.dimension

    def _is_available(self) -> bool:
        return importutil.find_spec("pymongo") is not None

    def _import_dependencies(self) -> None:
        if self._is_available():
            global pymongo
            import pymongo
        else:
            raise ImportError(
                "pymongo is not installed. Please install it with `pip install chonkie[mongodb]`."
            )

    def _generate_id(self, index: int, chunk: Chunk) -> str:
        return str(
            uuid5(NAMESPACE_OID, f"{self.collection_name}::chunk-{index}:{chunk.text}")
        )

    def _generate_document(
        self, index: int, chunk: Chunk, embedding: List[float]
    ) -> dict:
        return {
            "_id": self._generate_id(index, chunk),
            "text": chunk.text,
            "start_index": chunk.start_index,
            "end_index": chunk.end_index,
            "token_count": chunk.token_count,
            "embedding": embedding,
        }

    def write(self, chunks: Union[Chunk, List[Chunk]]) -> None:
        """Write chunks to the MongoDB collection."""
        if isinstance(chunks, Chunk):
            chunks = [chunks]
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_model.embed_batch(texts)  # type: ignore
        documents = []
        for index, chunk in enumerate(chunks):
            embedding = embeddings[index]
            if hasattr(embedding, "tolist"):
                embedding_list: List[float] = embedding.tolist()
            else:
                embedding_list = embedding  # type: ignore[assignment]
            documents.append(self._generate_document(index, chunk, embedding_list))
        if documents:
            self.collection.insert_many(documents)
            print(
                f"🦛 Chonkie wrote {len(documents)} chunks to MongoDB collection: {self.collection_name}"
            )

    def __repr__(self) -> str:
        """Return a string representation of the MongoDBHandshake instance."""
        return f"MongoDBHandshake(db_name={self.db_name}, collection_name={self.collection_name})"

    def search(
        self,
        query: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks in the MongoDB collection.

        Args:
            query: The query string to search for.
            embedding: The embedding vector to search for.
            limit: The number of top similar chunks to return.

        Returns:
            A list of dictionaries containing the similar chunks and their metadata.

        """
        assert (query is not None or embedding is not None), "Either query or embedding must be provided."
        if query is not None:
            embedding = self.embedding_model.embed(query).tolist()
        # Get all documents with embeddings
        docs = list(
            self.collection.find(
                {},
                {
                    "_id": 1,
                    "text": 1,
                    "embedding": 1,
                    "start_index": 1,
                    "end_index": 1,
                    "token_count": 1,
                },
            )
        )

        def cosine_similarity(a: List[float], b: List[float]) -> float:
            """Compute cosine similarity between two vectors."""
            import math

            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(y * y for y in b))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)

        # Score and sort
        results = []
        for doc in docs:
            emb = doc.get("embedding")
            if emb is not None:
                score = cosine_similarity(embedding, emb) # type: ignore[arg-type]
                result = {
                    "id": doc["_id"],
                    "score": score,
                    "text": doc["text"],
                    "start_index": doc.get("start_index"),
                    "end_index": doc.get("end_index"),
                    "token_count": doc.get("token_count"),
                }
                results.append(result)
        # Sort by score descending and return limit
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]
