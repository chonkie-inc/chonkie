"""Module containing the LateChunker class."""

import importlib.util as importutil
from typing import Any, List, Optional, Union

# Get all the Chonkie imports
from chonkie.chunker.recursive import RecursiveChunker
from chonkie.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from chonkie.types import LateChunk, RecursiveRules


class LateChunker(RecursiveChunker):
    """A chunker that chunks texts based on late interaction.

    This class extends the RecursiveChunker class and overrides its chunk method to implement late chunking.

    Args:
        embedding_model: The embedding model to use for chunking.
        chunk_size: The maximum size of each chunk.

    """

    def __init__(
        self,
        embedding_model: Union[
            str, SentenceTransformerEmbeddings, Any
        ] = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 512,
        rules: RecursiveRules = RecursiveRules(),
        min_characters_per_chunk: int = 24,
        **kwargs: Any,
    ) -> None:
        """Initialize the LateChunker.

        Args:
            embedding_model: The embedding model to use for chunking.
            chunk_size: The maximum size of each chunk.
            rules: The rules to use for chunking.
            min_characters_per_chunk: The minimum number of characters per chunk.
            **kwargs: Additional keyword arguments.

        """
        # Lazy import all the dependencies on initialization
        self._import_dependencies()

        # set all the additional attributes
        if isinstance(embedding_model, SentenceTransformerEmbeddings):
            self.embedding_model = embedding_model
        elif isinstance(embedding_model, str):
            self.embedding_model = SentenceTransformerEmbeddings(model=embedding_model, **kwargs)
        else:
            raise ValueError(f"{embedding_model} is not a valid embedding model")

        # Probably the dependency hasn't been installed
        if self.embedding_model is None:
            raise ImportError(
                "Oh! seems like you're missing the proper dependency to run this chunker. Please install it using `pip install chonkie[st]`"
            )

        # Initialize the RecursiveChunker with the embedding_model's tokenizer
        super().__init__(
            tokenizer_or_token_counter=self.embedding_model.get_tokenizer_or_token_counter(),
            chunk_size=chunk_size,
            rules=rules,
            min_characters_per_chunk=min_characters_per_chunk,
            return_type="chunks",
        )

        # Disable multiprocessing for this chunker
        self._use_multiprocessing = False
    
    @classmethod
    def from_recipe(cls, 
                    name: Optional[str] = "default", 
                    lang: Optional[str] = "en", 
                    path: Optional[str] = None, 
                    embedding_model: Union[str, SentenceTransformerEmbeddings] = "sentence-transformers/all-MiniLM-L6-v2",
                    chunk_size: int = 512,
                    min_characters_per_chunk: int = 24,
                    **kwargs: Any) -> "LateChunker": # type: ignore
        """Create a LateChunker from a recipe.

        Args:
            name: The name of the recipe to use.
            lang: The language that the recipe should support.
            path: The path to the recipe to use.
            embedding_model: The embedding model to use.
            chunk_size: The chunk size to use.
            min_characters_per_chunk: The minimum number of characters per chunk.
            **kwargs: Additional keyword arguments.

        Returns:
            LateChunker: The created LateChunker.   

        Raises:
            ValueError: If the recipe is invalid or if the recipe is not found.

        """
        # Create a hubbie instance
        rules = RecursiveRules.from_recipe(name, lang, path)
        return cls(
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            rules=rules,
            min_characters_per_chunk=min_characters_per_chunk,
            **kwargs
        )   
    
    def _get_late_embeddings(
        self, token_embeddings: "np.ndarray", token_counts: List[int]
    ) -> List["np.ndarray"]:
        # Split the token embeddings into chunks based on the token counts
        embs = []
        cum_token_counts = np.cumsum([0] + token_counts)
        for i in range(len(token_counts)):
            embs.append(
                np.mean(
                    token_embeddings[cum_token_counts[i] : cum_token_counts[i + 1]],
                    axis=0,
                )
            )
        return embs

    def chunk(self, text: str) -> List[LateChunk]: # type: ignore
        """Chunk the text via LateChunking."""
        # This would first call upon the _recursive_chunk method
        # and then use the embedding model to get the token token_embeddings
        # Lastly, we would combine the methods together to create the LateChunk objects
        chunks = self._recursive_chunk(text)
        token_embeddings = self.embedding_model.embed_as_tokens(text)

        # Get the token_counts for all the chunks
        token_counts = [c.token_count for c in chunks]

        # Validate the token_counts with the actual count
        if sum(token_counts) > token_embeddings.shape[0]:
            raise ValueError(
                "The sum of token counts exceeds the number of tokens in the text"
            )
        # Diff would always be positive now~ which it should be considering token_counts
        # doesn't have any special tokens added
        if sum(token_counts) < token_embeddings.shape[0]:
            # Use a little trick to ensure that the token counts get properly adjusted
            diff = token_embeddings.shape[0] - sum(token_counts)
            token_counts[0] = token_counts[0] + diff // 2
            token_counts[-1] = token_counts[-1] + (diff - diff // 2)

        if sum(token_counts) != token_embeddings.shape[0]:
            raise ValueError(
                "The sum of token counts does not match the number of tokens in the text",
                f"Expected {token_embeddings.shape[0]}, got {sum(token_counts)}",
            )

        # Split the token embeddings into chunks based on the token counts
        late_embds = self._get_late_embeddings(token_embeddings, token_counts)

        # Wrap it all up in LateChunks
        result = []
        for chunk, token_count, embedding in zip(chunks, token_counts, late_embds):
            result.append(
                LateChunk(
                    text=chunk.text,
                    start_index=chunk.start_index,
                    end_index=chunk.end_index,
                    token_count=token_count,
                    embedding=embedding,
                )
            )
        return result

    def _import_dependencies(self) -> None:
        """Lazy import dependencies for the chunker implementation.

        This method should be implemented by all chunker implementations that require
        additional dependencies. It lazily imports the dependencies only when they are needed.
        """
        if importutil.find_spec("numpy"):
            global np
            import numpy as np
        else:
            raise ImportError(
                "numpy is not available. Please install it via `pip install chonkie[semantic]`"
            )
