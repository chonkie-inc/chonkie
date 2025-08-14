"""Pinecone Handshake to export Chonkie's Chunks into a Pinecone index."""

import importlib.util as importutil
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Sequence, Union
from uuid import NAMESPACE_OID, uuid5

from chonkie.embeddings import AutoEmbeddings, BaseEmbeddings
from chonkie.types import Chunk

from .base import BaseHandshake
from .utils import generate_random_collection_name

if TYPE_CHECKING:
    import pinecone

class PineconeHandshake(BaseHandshake):
    """Pinecone Handshake to export Chonkie's Chunks into a Pinecone index.
    
    This handshake is experimental and may change in the future. Not all Chonkie features are supported yet.

    Args:
        client: Optional[pinecone.Pinecone]: The Pinecone client to use. If None, will create a new client.
        api_key: str: The Pinecone API key.
        index_name: Union[str, Literal["random"]]: The name of the index to use.
        embedding_model: Union[str, BaseEmbeddings]: The embedding model to use.
        dimension: Optional[int]: The dimension of the embeddings. If not provided, will infer from embedding_model.
        **kwargs: Additional keyword arguments to pass to the Pinecone client.
    """

    def __init__(self,
                 client: Optional["pinecone.Pinecone"] = None,
                 api_key : Optional[str] = None,
                 index_name: Union[str, Literal["random"]] = "random",
                 embedding_model: Union[str, BaseEmbeddings] = "minishlab/potion-retrieval-32M",
                 dimension: Optional[int] = None,
                 **kwargs: Dict[str, Any]
                 ) -> None:
        """Initialize the Pinecone handshake.

        Args:
            client: Optional[pinecone.Pinecone]: The Pinecone client to use. If None, will create a new client.
            api_key: The Pinecone API key.
            index_name: The name of the index to use, or "random" for auto-generated name.
            embedding_model: The embedding model to use, either as string or BaseEmbeddings instance.
            dimension: The dimension of the embeddings. If not provided, will infer from embedding_model.
            **kwargs: Additional keyword arguments to pass to the Pinecone client.
        
        """
        super().__init__()
        self._import_dependencies()
        
        if client is not None:
            self.client = client
        else : 
            self.client = pinecone.Pinecone(api_key=api_key)

        if isinstance(embedding_model, str):
            self.embedding_model = AutoEmbeddings.get_embeddings(embedding_model)
        else:
            self.embedding_model = embedding_model
        self.dimension = dimension or self.embedding_model.dimension
        
        if index_name == "random":
            while True:
                self.index_name = generate_random_collection_name()
                if not self.client.has_index(self.index_name):
                    break
            print(f"🦛 Chonkie created a new index in Pinecone: {self.index_name}")
        else:
            self.index_name = index_name

        # Create the index if it doesn't exist
        if not self.client.has_index(self.index_name):
            self.client.create_index(name=self.index_name, dimension=self.dimension, **kwargs)
        self.index = self.client.Index(self.index_name)

    def _is_available(self) -> bool:
        return importutil.find_spec("pinecone") is not None

    def _import_dependencies(self) -> None:
        if self._is_available():
            global pinecone
            import pinecone
        else:
            raise ImportError("Pinecone is not installed. Please install it with `pip install chonkie[pinecone]`.")

    def _generate_id(self, index: int, chunk: Chunk) -> str:
        return str(uuid5(NAMESPACE_OID, f"{self.index_name}::chunk-{index}:{chunk.text}"))

    def _generate_metadata(self, chunk: Chunk) -> dict:
        return {
            "text": chunk.text,
            "start_index": chunk.start_index,
            "end_index": chunk.end_index,
            "token_count": chunk.token_count,
        }

    def _get_vectors(self, chunks: Union[Chunk, Sequence[Chunk]]):
        if isinstance(chunks, Chunk):
            chunks = [chunks]
        vectors = []
        for index, chunk in enumerate(chunks):
            vectors.append((
                self._generate_id(index, chunk),
                self.embedding_model.embed(chunk.text).tolist(),
                self._generate_metadata(chunk)
            ))
        return vectors

    def write(self, chunks: Union[Chunk, Sequence[Chunk]]) -> None:
        """Write chunks to the Pinecone index.

        Args:
            chunks: A single Chunk or sequence of Chunks to write to the index.

        Returns:
            None

        """
        if isinstance(chunks, Chunk):
            chunks = [chunks]
        vectors = self._get_vectors(chunks)
        self.index.upsert(vectors)
        print(f"🦛 Chonkie wrote {len(chunks)} chunks to Pinecone index: {self.index_name}")

    def __repr__(self) -> str:
        """Return a string representation of the PineconeHandshake instance.

        Returns:
            str: A string representation containing the index name.
        
        """
        return f"PineconeHandshake(index_name={self.index_name})"
