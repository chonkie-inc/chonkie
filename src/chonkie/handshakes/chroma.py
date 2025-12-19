"""Chroma Handshake to export Chonkie's Chunks into a Chroma collection."""

import importlib.util as importutil
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Optional,
    Union,
    cast,
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
    import numpy as np


# NOTE: This is a bit of a hack to work with Chroma's EmbeddingFunction interface
# since we can't have a EmbeddingFunction without having ChromaDB in the base install.
# So we create a local class (which we don't pass to our namespace) that mimics the
# interface of chromadb.EmbeddingFunction. It has a __call__ that takes in a input
# and returns a numpy array.

# Since chromadb.Documents and chromadb.Embeddings are just strings and numpy arrays respectively,
# we can just return the numpy array from __call__ and be done with it.


class ChromaEmbeddingFunction:
    """Chroma Embedding Function.

    Embeds the text of the chunks using the embedding model and
    adds the embeddings to the chunks for use in downstream tasks
    like upserting into a vector database.

    Args:
        embedding_model: The embedding model to use.
        **kwargs: Additional keyword arguments.

    """

    def __init__(
        self,
        embedding_model: Union[str, BaseEmbeddings] = "minishlab/potion-retrieval-32M",
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize the ChromaEmbeddingFunction."""
        super().__init__()

        # Check if the model is a string
        if isinstance(embedding_model, str):
            self.embedding_model = AutoEmbeddings.get_embeddings(embedding_model, **kwargs)
            self._model_name = embedding_model  # Store name for ChromaDB compatibility
        elif isinstance(embedding_model, BaseEmbeddings):
            self.embedding_model = embedding_model
            self._model_name = str(embedding_model)  # Store name for ChromaDB compatibility
        else:
            raise ValueError("Model must be a string or a BaseEmbeddings instance.")

    def name(self) -> str:
        """Return the name of the embedding model for ChromaDB compatibility."""
        return self._model_name

    def __call__(self, input: Union[str, list[str]]) -> Union["np.ndarray", list["np.ndarray"]]:
        """Call the ChromaEmbeddingFunction."""
        if isinstance(input, str):
            return self.embedding_model.embed(input)
        elif isinstance(input, list):
            return self.embedding_model.embed_batch(input)
        else:
            raise ValueError("Input must be a string or a list of strings.")


@handshake("chroma")
class ChromaHandshake(BaseHandshake):
    """Chroma Handshake to export Chonkie's Chunks into a Chroma collection.

    This handshake is experimental and may change in the future. Not all Chonkie features are supported yet.

    Args:
        client: The Chroma client to use.
        collection_name: The name of the collection to use.
        embedding_model: The embedding model to use.
        path: The path to the Chroma collection locally. If provided, it will create a Persistent Chroma Client.

    """

    def __init__(
        self,
        client: Optional[Any] = None,  # chromadb.Client
        collection_name: Union[str, Literal["random"]] = "random",
        embedding_model: Union[str, BaseEmbeddings] = "minishlab/potion-retrieval-32M",
        path: Optional[str] = None,
    ) -> None:
        """Initialize the Chroma Handshake.

        Args:
            client: The Chroma client to use.
            collection_name: The name of the collection to use.
            embedding_model: The embedding model to use.
            path: The path to the Chroma collection locally. If provided, it will create a Persistent Chroma Client.

        """
        super().__init__()

        try:
            import chromadb
        except ImportError as ie:
            raise ImportError(
                "ChromaDB is not installed. Please install it with `pip install chonkie[chroma]`.",
            ) from ie

        # Initialize Chroma client
        if client is None and path is None:
            self.client = chromadb.Client()
        elif client is None and path is not None:
            self.client = chromadb.PersistentClient(path=path)
        else:
            self.client = client  # type: ignore[assignment]

        # Initialize the EmbeddingRefinery internally!
        self.embedding_function = ChromaEmbeddingFunction(embedding_model)

        # If the collection name is not random, create the collection
        if collection_name != "random":
            self.collection_name = collection_name
            self.collection = self.client.get_or_create_collection(
                self.collection_name,
                embedding_function=self.embedding_function,  # type: ignore[arg-type]
            )  # type: ignore[arg-type]
        else:
            # Keep generating random collection names until we find one that doesn't exist
            while True:
                self.collection_name = generate_random_collection_name()
                try:
                    self.collection = self.client.create_collection(
                        self.collection_name,
                        embedding_function=self.embedding_function,  # type: ignore[arg-type]
                    )  # type: ignore[arg-type]
                    break
                except Exception:
                    pass
            logger.info(f"Chonkie created a new collection in ChromaDB: {self.collection_name}")

        # Now that we have a collection, we can write the Chunks to it!

    @classmethod
    def _is_available(cls) -> bool:
        """Check if the dependencies are available."""
        return importutil.find_spec("chromadb") is not None

    def _generate_id(self, index: int, chunk: Chunk) -> str:
        """Generate a unique index name for the Chunk."""
        return str(uuid5(NAMESPACE_OID, f"{self.collection_name}::chunk-{index}:{chunk.text}"))

    def _generate_metadata(self, chunk: Chunk) -> dict:
        """Generate the metadata for the Chunk."""
        return {
            "start_index": chunk.start_index,
            "end_index": chunk.end_index,
            "token_count": chunk.token_count,
        }

    def write(self, chunks: Union[Chunk, list[Chunk]]) -> None:
        """Write the Chunks to the Chroma collection."""
        if isinstance(chunks, Chunk):
            chunks = [chunks]

        logger.debug(f"Writing {len(chunks)} chunks to Chroma collection: {self.collection_name}")
        # Generate the ids and metadata
        ids = [self._generate_id(index, chunk) for (index, chunk) in enumerate(chunks)]
        metadata = [self._generate_metadata(chunk) for chunk in chunks]
        texts = [chunk.text for chunk in chunks]

        # Write the Chunks to the Chroma collection
        # Since this uses the `upsert` method, if the same index and same chunk text already exist, it will update the existing Chunk — which would only be the case if the Chunk has a different embedding
        self.collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadata,  # type: ignore
        )

        logger.info(
            f"Chonkie wrote {len(chunks)} chunks to the Chroma collection: {self.collection_name}",
        )

    def __repr__(self) -> str:
        """Return the string representation of the ChromaHandshake."""
        return f"ChromaHandshake(collection_name={self.collection_name})"

    def search(
        self,
        query: Optional[str] = None,
        embedding: Optional[list[float]] = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Search the Chroma collection for similar chunks.

        Args:
            query: The query string to search for. If provided, `embedding` is ignored.
            embedding: The embedding vector to search for.
            limit: The maximum number of results to return.

        Returns:
            list[dict[str, Any]]: A list of dictionaries containing the matching chunks and their metadata.

        """
        logger.debug(f"Searching Chroma collection: {self.collection_name} with limit={limit}")
        if query is None and embedding is None:
            raise ValueError("Either 'query' or 'embedding' must be provided.")

        # Determine the query embeddings based on the input
        if query:
            query_embedding_result = cast("np.ndarray", self.embedding_function(query))
            query_embeddings = [query_embedding_result.tolist()]
        else:
            query_embeddings = [embedding]  # type: ignore[list-item]

        # Perform the query
        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=limit,
            include=["metadatas", "documents", "distances"],
        )

        # Safely extract results, checking for None values
        ids_list = results.get("ids")
        distances_list = results.get("distances")
        metadatas_list = results.get("metadatas")
        documents_list = results.get("documents")

        # Ensure all required result lists are present and not None
        if (
            ids_list is None
            or distances_list is None
            or metadatas_list is None
            or documents_list is None
        ):
            return []

        # We queried with one vector, so we get the first list of results
        ids, distances, metadatas, documents = (
            ids_list[0],
            distances_list[0],
            metadatas_list[0],
            documents_list[0],
        )

        # Process and format the results
        matches = []
        distance_metric = (
            self.collection.metadata.get("hnsw:space", "l2") if self.collection.metadata else "l2"
        )

        for id_val, distance, metadata, document in zip(ids, distances, metadatas, documents):
            similarity = None
            if distance is not None:
                if distance_metric == "cosine":
                    similarity = 1.0 - distance
                elif distance_metric == "l2":
                    similarity = 1.0 - (distance**2 / 2)
                else:  # 'ip' (inner product) is already a similarity score
                    similarity = distance

            match_data = {
                "id": id_val,
                "score": similarity,
                "text": document,
            }
            if metadata:
                match_data.update(metadata)

            matches.append(match_data)

        logger.info(f"Search complete: found {len(matches)} matching chunks")
        return matches
