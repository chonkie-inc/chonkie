"""LanceDB Handshake to export Chonkie's Chunks into a LanceDB table."""

import importlib.util as importutil
import os
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Optional,
    Union,
)

import numpy as np

from chonkie.embeddings import AutoEmbeddings, BaseEmbeddings
from chonkie.logger import get_logger
from chonkie.pipeline import handshake
from chonkie.types import Chunk

from .base import BaseHandshake
from .utils import generate_random_collection_name

logger = get_logger(__name__)

if TYPE_CHECKING:
    import lancedb as lancedb_module


@handshake("lancedb")
class LanceDBHandshake(BaseHandshake):
    """LanceDB Handshake to export Chonkie's Chunks into a LanceDB table.

    Args:
        connection: Optional LanceDB connection. If None, a new connection is created.
        uri: The URI to the LanceDB database. Defaults to "memory://" for an in-memory DB.
        table_name: The name of the table to use, or "random" to auto-generate one.
        embedding_model: The embedding model identifier or instance.
        **kwargs: Additional keyword arguments passed to lancedb.connect().

    """

    def __init__(
        self,
        connection: Optional["lancedb_module.DBConnection"] = None,
        uri: Union[str, os.PathLike] = "memory://",
        table_name: Union[str, Literal["random"]] = "random",
        embedding_model: Union[str, BaseEmbeddings] = "minishlab/potion-retrieval-32M",
        **kwargs: Any,
    ) -> None:
        """Initialize the LanceDB Handshake.

        Args:
            connection: Optional LanceDB connection. If None, a new connection is created using `uri`.
            uri: The URI to the LanceDB database. Use "memory://" for an ephemeral in-memory database.
            table_name: The name of the table to write chunks to, or "random" to auto-generate a name.
            embedding_model: The embedding model identifier string or a BaseEmbeddings instance.
            **kwargs: Additional keyword arguments passed to lancedb.connect().

        """
        super().__init__()

        try:
            import lancedb

        except ImportError as ie:
            raise ImportError(
                "LanceDB is not installed. Please install it with `pip install chonkie[lancedb]`.",
            ) from ie

        # Initialize the connection
        if connection is None:
            self.connection = lancedb.connect(str(uri), **kwargs)
        else:
            self.connection = connection

        # Initialize the embedding model
        if isinstance(embedding_model, str):
            embedding_model = AutoEmbeddings.get_embeddings(embedding_model)
        if not isinstance(embedding_model, BaseEmbeddings):
            raise ValueError(
                "The provided embedding model is not a valid BaseEmbeddings instance."
            )

        self.embedding_model = embedding_model
        self.dimension = self.embedding_model.dimension

        # Initialize the table name
        if table_name == "random":
            self.table_name = generate_random_collection_name()
            logger.info(f"Chonkie created a new LanceDB table: {self.table_name}")
        else:
            self.table_name = table_name

        # Create the table if it doesn't exist
        existing_tables = self.connection.table_names()
        if self.table_name not in existing_tables:
            import pyarrow as pa

            schema = pa.schema([
                pa.field("id", pa.utf8()),
                pa.field("text", pa.utf8()),
                pa.field("start_index", pa.int32()),
                pa.field("end_index", pa.int32()),
                pa.field("token_count", pa.int32()),
                pa.field("vector", pa.list_(pa.float32(), self.dimension)),
            ])
            self.table = self.connection.create_table(self.table_name, schema=schema)
        else:
            self.table = self.connection.open_table(self.table_name)

    @classmethod
    def _is_available(cls) -> bool:
        """Check if the dependencies are installed."""
        return importutil.find_spec("lancedb") is not None

    def _generate_row(self, chunk: Chunk, embedding: list[float]) -> dict:
        """Generate a row dict for the chunk."""
        return {
            "id": self._generate_id(f"{self.table_name}:{chunk.start_index}:{chunk.text}"),
            "text": chunk.text,
            "start_index": chunk.start_index,
            "end_index": chunk.end_index,
            "token_count": chunk.token_count,
            "vector": embedding,
        }

    def write(self, chunks: Union[Chunk, list[Chunk]]) -> None:
        """Write chunks to the LanceDB table."""
        if isinstance(chunks, Chunk):
            chunks = [chunks]

        logger.debug(f"Writing {len(chunks)} chunks to LanceDB table: {self.table_name}")

        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_model.embed_batch(texts)

        rows = []
        for chunk, embedding in zip(chunks, embeddings):
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            rows.append(self._generate_row(chunk, embedding))

        # Upsert: update if id matches, insert otherwise
        (
            self.table
            .merge_insert("id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(rows)
        )

        logger.info(
            f"Chonkie wrote {len(rows)} chunks to LanceDB table: {self.table_name}",
        )

    def __repr__(self) -> str:
        """Return the string representation of the LanceDBHandshake."""
        return f"LanceDBHandshake(table_name={self.table_name})"

    def search(
        self,
        query: Optional[str] = None,
        embedding: Optional[list[float]] = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Retrieve the top-k most similar chunks to the query.

        Args:
            query: The query string to search for.
            embedding: The embedding vector to search for. Only used if `query` is not provided.
            limit: The number of top similar chunks to retrieve.

        Returns:
            list[dict[str, Any]]: The list of most similar chunks with their metadata.

        """
        logger.debug(f"Searching LanceDB table: {self.table_name} with limit={limit}")
        if embedding is None and query is None:
            raise ValueError("Either query or embedding must be provided")
        if query is not None:
            embedding = self.embedding_model.embed(query).tolist()

        query_builder: Any = self.table.search(embedding)
        results = query_builder.metric("cosine").limit(limit).to_list()  # ty: ignore[unresolved-attribute]

        matches = [
            {
                "id": r["id"],
                "score": 1.0 - r["_distance"],  # cosine distance -> similarity
                "text": r["text"],
                "start_index": r["start_index"],
                "end_index": r["end_index"],
                "token_count": r["token_count"],
            }
            for r in results
        ]
        logger.info(f"Search complete: found {len(matches)} matching chunks")
        return matches
