"""Module containing the LateChunker class."""

from typing import Any, List, Optional, Union

import numpy as np

# Get all the Chonkie imports
from chonkie.chunker.recursive import RecursiveChunker
from chonkie.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from chonkie.logger import get_logger
from chonkie.pipeline import chunker
from chonkie.types import Chunk, RecursiveRules

logger = get_logger(__name__)


@chunker("late")
class LateChunker(RecursiveChunker):
    """A chunker that chunks texts based on late interaction.

    This class extends the RecursiveChunker class and overrides its chunk method to implement late chunking.

    Args:
        embedding_model: The embedding model to use for chunking.
        chunk_size: The maximum size of each chunk.
        rules: Recursive rules to chunk by
        min_characters_per_chunk: Minimum number of characters in a single chunk

    """

    def __init__(
        self,
        embedding_model: Union[
            str, SentenceTransformerEmbeddings, Any
        ] = "nomic-ai/modernbert-embed-base",
        chunk_size: int = 2048,
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
            tokenizer=self.embedding_model.get_tokenizer(),
            chunk_size=chunk_size,
            rules=rules,
            min_characters_per_chunk=min_characters_per_chunk,
        )

        # Disable multiprocessing for this chunker
        self._use_multiprocessing = False
    
    @classmethod
    def from_recipe(cls,  # type: ignore[override]
                    name: Optional[str] = "default", 
                    lang: Optional[str] = "en", 
                    path: Optional[str] = None, 
                    embedding_model: Union[str, SentenceTransformerEmbeddings] = "sentence-transformers/all-MiniLM-L6-v2",
                    chunk_size: int = 2048,
                    min_characters_per_chunk: int = 24,
                    **kwargs: Any) -> "LateChunker":
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
        logger.info("Loading LateChunker recipe", name=name, lang=lang)
        # Create a hubbie instance
        rules = RecursiveRules.from_recipe(name, lang, path)
        logger.debug(f"Recipe loaded successfully with {len(rules.levels or [])} levels")
        return cls(
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            rules=rules,
            min_characters_per_chunk=min_characters_per_chunk,
            **kwargs
        )   
    
    def _get_late_embeddings(
        self, token_embeddings: np.ndarray, token_counts: List[int]
    ) -> List[np.ndarray]:
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

    def chunk(self, text: str) -> List[Chunk]:
        """Chunk the text via LateChunking."""
        logger.debug(f"Starting late chunking for text of length {len(text)}")
        # This would first call upon the _recursive_chunk method
        # and then use the embedding model to get the token token_embeddings
        # Lastly, we would combine the methods together to create the LateChunk objects
        chunks = self._recursive_chunk(text)
        logger.debug(f"Created {len(chunks)} initial chunks from recursive splitting")
        token_embeddings = self.embedding_model.embed_as_tokens(text)

        # Get the token_counts for all the chunks
        token_counts = [c.token_count for c in chunks]  # type: ignore[union-attr]

        # If fallback was used, token_embeddings may be fewer than sum(token_counts)
        if token_embeddings.shape[0] < sum(token_counts):
            # Fallback: use sentence embeddings for each chunk
            # Re-embed each chunk as a sentence embedding
            token_embeddings = np.array([
                self.embedding_model.embed(c.text) for c in chunks
            ])
            token_counts = [1 for _ in chunks]

        # Validate the token_counts with the actual count
        if sum(token_counts) > token_embeddings.shape[0]:
            raise ValueError(
                "The sum of token counts exceeds the number of tokens in the text"
            )
        if sum(token_counts) < token_embeddings.shape[0]:
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

        # Wrap it all up in Chunks with embeddings
        result = []
        for chunk, token_count, embedding in zip(chunks, token_counts, late_embds):
            # Note: LateChunker always returns chunks, so chunk is always a Chunk
            result.append(
                Chunk(
                    text=chunk.text,  # type: ignore[attr-defined]
                    start_index=chunk.start_index,  # type: ignore[attr-defined]
                    end_index=chunk.end_index,  # type: ignore[attr-defined]
                    token_count=token_count,
                    embedding=embedding,
                )
            )
        logger.info(f"Created {len(result)} chunks with late interaction embeddings")
        return result

