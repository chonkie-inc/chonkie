"""Turbopuffer Handshake to export Chonkie's Chunks into a Turbopuffer database."""
import importlib.util as importutil
import os
import warnings
from typing import TYPE_CHECKING, Any, Literal, Optional, Sequence, Union
from uuid import NAMESPACE_OID, uuid5

from chonkie.embeddings import AutoEmbeddings, BaseEmbeddings
from chonkie.types import Chunk

from .base import BaseHandshake
from .utils import generate_random_collection_name

if TYPE_CHECKING:
    import turbopuffer as tpuf


class TurbopufferHandshake(BaseHandshake):
    """Turbopuffer Handshake to export Chonkie's Chunks into a Turbopuffer database."""

    def __init__(self, 
                 namespace: Optional["tpuf.Namespace"] = None,
                 namespace_name: Union[str, Literal["random"]] = "random", 
                 embedding_model: Union[str, BaseEmbeddings] = "minishlab/potion-retrieval-32M", 
                 api_key: Optional[str] = None,
                 ) -> None:
        """Initialize the Turbopuffer Handshake.
        
        Args:
            namespace: The namespace to use.
            namespace_name: The name of the namespace to use, if the namespace is not provided.
            embedding_model: The embedding model to use.
            api_key: The API key to use.

        """
        # Warn the user that TurbopufferHandshake is experimental
        warnings.warn("Chonkie's TurbopufferHandshake is experimental and may change in the future. Not all Chonkie features are supported yet.", FutureWarning)
        
        super().__init__()

        # Lazy import the dependencies
        self._import_dependencies()

        # Check for the API Key
        api_key = api_key or os.getenv("TURBOPUFFER_API_KEY")
        if not api_key:
            raise ValueError("Turbopuffer API key not found. Please provide an API key or set the TURBOPUFFER_API_KEY environment variable.")

        # Setting the tpuf api key
        tpuf.api_key = api_key

        # Get a list of namespaces
        namespaces = [ns.name for ns in tpuf.namespaces()]

        # If the namespace is not provided, generate a random one
        if namespace is None:
            if namespace_name == "random":
                # Generate a random namespace
                while True:
                    namespace_name = generate_random_collection_name()
                    if namespace_name not in namespaces:
                        break
            self.namespace = tpuf.Namespace(namespace_name)
            print(f"🦛 Chonkie has created a new namespace: {self.namespace.name}")
        else:
            self.namespace = namespace

        # Initialize the embedding model
        self.embedding_model = AutoEmbeddings.get_embeddings(embedding_model)

    def _is_available(self) -> bool:
        """Check if Turbopuffer is available."""
        return importutil.find_spec("turbopuffer") is not None

    def _import_dependencies(self) -> None:
        """Import the dependencies for Turbopuffer."""
        if self._is_available():
            global tpuf
            import turbopuffer as tpuf

    def _generate_id(self, index: int, chunk: Chunk) -> str:
        """Generate a unique ID for the chunk."""
        return str(
            uuid5(
                NAMESPACE_OID, 
                f"{self.namespace.name}::chunk-{index}:{chunk.text}"
            )
        )

    def write(self, chunks: Union[Chunk, Sequence[Chunk]]) -> None:
        """Write the chunks to the Turbopuffer database."""
        if isinstance(chunks, Chunk):
            chunks = [chunks]

        # Embed the chunks
        ids = [self._generate_id(index, chunk) for (index, chunk) in enumerate(chunks)]
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_model.embed_batch(texts)
        start_indices = [chunk.start_index for chunk in chunks]
        end_indices = [chunk.end_index for chunk in chunks]
        token_counts = [chunk.token_count for chunk in chunks]

        # Write the chunks to the database
        self.namespace.write(
            upsert_columns={
                "id": ids,
                "vector": embeddings,
                "text": texts,
                "start_index": start_indices,
                "end_index": end_indices,
                "token_count": token_counts,
            }
        )
        
        print(f"🦛 Chonkie has written {len(chunks)} chunks to the namespace: {self.namespace.name}")

    def __repr__(self) -> str:
        """Return the representation of the Turbopuffer Handshake."""
        return f"TurbopufferHandshake(namespace={self.namespace.name})"
