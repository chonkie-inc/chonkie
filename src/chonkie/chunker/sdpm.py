"""Module containing the SDPMChunker class.

This chunker uses the Semantic Double-Pass Merging algorithm to chunk text.

"""

from typing import Any, Dict, List, Literal, Optional, Union

from chonkie.chunker.semantic import SemanticChunker
from chonkie.embeddings import BaseEmbeddings
from chonkie.types import SemanticChunk, Sentence
from chonkie.utils import Hubbie


class SDPMChunker(SemanticChunker):
    """Chunker using the Semantic Double-Pass Merging algorithm.

    This chunker uses the Semantic Double-Pass Merging algorithm to chunk text.

    Args:
        embedding_model: The embedding model to use.
        mode: The mode to use.
        threshold: The threshold to use.
        chunk_size: The chunk size to use.
        similarity_window: The similarity window to use.
        min_sentences: The minimum number of sentences to use.
        min_chunk_size: The minimum chunk size to use.
        min_characters_per_sentence: The minimum number of characters per sentence to use.
        threshold_step: The threshold step to use.
        delim: The delimiters to use.
        include_delim: Whether to include delimiters in chunks.
        skip_window: The skip window to use.
        return_type: The return type to use.
        **kwargs: Additional keyword arguments.

    """

    def __init__(
        self,
        embedding_model: Union[str, BaseEmbeddings] = "minishlab/potion-base-8M",
        chunk_size: int = 512,
        mode: str = "window",
        threshold: Union[str, float, int] = "auto",
        similarity_window: int = 1,
        min_sentences: int = 1,
        min_chunk_size: int = 2,
        min_characters_per_sentence: int = 12,
        threshold_step: float = 0.01,
        delim: Union[str, List[str]] = [".", "!", "?", "\n"],
        include_delim: Optional[Literal["prev", "next"]] = "prev",
        skip_window: int = 1,
        return_type: Literal["chunks", "texts"] = "chunks",
        **kwargs: Dict[str, Any],
    ) -> None:  # type: ignore
        """Initialize the SDPMChunker.

        Args:
            embedding_model: The embedding model to use.
            mode: The mode to use.
            threshold: The threshold to use.
            chunk_size: The chunk size to use.
            similarity_window: The similarity window to use.
            min_sentences: The minimum number of sentences to use.
            min_chunk_size: The minimum chunk size to use.
            min_characters_per_sentence: The minimum number of characters per sentence to use.
            threshold_step: The threshold step to use.
            delim: The delimiters to use.
            include_delim: Whether to include delimiters.
            skip_window: The skip window to use.
            return_type: The return type to use.
            **kwargs: Additional keyword arguments.

        """
        super().__init__(
            embedding_model=embedding_model,
            mode=mode,
            threshold=threshold,
            chunk_size=chunk_size,
            similarity_window=similarity_window,
            min_sentences=min_sentences,
            min_chunk_size=min_chunk_size,
            min_characters_per_sentence=min_characters_per_sentence,
            threshold_step=threshold_step,
            delim=delim,
            include_delim=include_delim,
            return_type=return_type,
            **kwargs,
        )

        self.skip_window = skip_window

        # Disable the multiprocessing flag for this class
        self._use_multiprocessing = False

    
    @classmethod
    def from_recipe(cls,
                    name: str = "default", 
                    lang: Optional[str] = "en", 
                    path: Optional[str] = None, 
                    embedding_model: Union[str, BaseEmbeddings] = "minishlab/potion-base-8M",
                    mode: str = "window",
                    threshold: Union[str, float, int] = "auto",
                    chunk_size: int = 512,
                    similarity_window: int = 1,
                    min_sentences: int = 1,
                    min_chunk_size: int = 2,
                    min_characters_per_sentence: int = 12,
                    threshold_step: float = 0.01,
                    skip_window: int = 1,
                    return_type: Literal["chunks", "texts"] = "chunks",
                    **kwargs: Dict[str, Any]) -> "SDPMChunker":  # type: ignore
        """Create a SDPMChunker from a recipe.

        Args:
            name: The name of the recipe to use.
            lang: The language that the recipe should support.
            path: The path to the recipe to use.
            embedding_model: The embedding model to use.
            mode: The mode to use.
            threshold: The threshold to use.
            chunk_size: The chunk size to use.
            similarity_window: The similarity window to use.
            min_sentences: The minimum number of sentences to use.
            min_chunk_size: The minimum chunk size to use.
            min_characters_per_sentence: The minimum number of characters per sentence to use.
            threshold_step: The threshold step to use.
            skip_window: The skip window to use.
            return_type: The return type to use.
            **kwargs: Additional keyword arguments.

        Returns:
            SDPMChunker: The created SDPMChunker.

        Raises:
            ValueError: If the recipe is invalid or if the recipe is not found.

        """
        # Create a hubbie instance
        hub = Hubbie()
        recipe = hub.get_recipe(name, lang, path)
        return cls(
            embedding_model=embedding_model,
            mode=mode,
            threshold=threshold,
            chunk_size=chunk_size,
            similarity_window=similarity_window,
            min_sentences=min_sentences,
            min_chunk_size=min_chunk_size,
            min_characters_per_sentence=min_characters_per_sentence,
            threshold_step=threshold_step,
            delim=recipe["recipe"]["delimiters"],
            include_delim=recipe["recipe"]["include_delim"],
            skip_window=skip_window,
            return_type=return_type,
            **kwargs,
        )

    def _merge_sentence_groups(self, sentence_groups: List[List[str]]) -> List[str]:
        """Merge sentence groups into a single sentence.

        Args:
            sentence_groups: The sentence groups to merge.

        Returns:
            The merged sentence.

        """
        merged_sentences = []
        for sentence_group in sentence_groups:
            merged_sentences.extend(sentence_group)
        return merged_sentences

    def _skip_and_merge(
        self, groups: List[List[Sentence]], similarity_threshold: float
    ) -> List[List[Sentence]]:
        """Merge similar groups considering skip window."""
        if len(groups) <= 1:
            return groups

        merged_groups = []
        embeddings = [self._compute_group_embedding(group) for group in groups]

        while groups:
            if len(groups) == 1:
                merged_groups.append(groups[0])
                break

            # Calculate skip index ensuring it's valid
            skip_index = min(self.skip_window + 1, len(groups) - 1)

            # Compare current group with skipped group
            similarity = self._get_semantic_similarity(
                embeddings[0], embeddings[skip_index]
            )

            if similarity >= similarity_threshold:
                # Merge groups from 0 to skip_index (inclusive)
                merged = self._merge_sentence_groups(groups[: skip_index + 1])

                # Remove the merged groups
                for _ in range(skip_index + 1):
                    groups.pop(0)
                    embeddings.pop(0)

                # Add merged group back at the start
                groups.insert(0, merged)
                embeddings.insert(0, self._compute_group_embedding(merged))
            else:
                # No merge possible, move first group to results
                merged_groups.append(groups.pop(0))
                embeddings.pop(0)

        return merged_groups

    def chunk(self, text: str) -> List[SemanticChunk]:
        """Split text into semantically coherent chunks using two-pass approach.

        First groups sentences by semantic similarity, then splits groups to respect
        chunk_size while maintaining sentence boundaries.

        Args:
            text: Input text to be chunked

        Returns:
            List of SemanticChunk objects containing the chunked text and metadata

        """
        if not text.strip():
            return []

        # Prepare sentences with precomputed information
        sentences = self._prepare_sentences(text)
        if len(sentences) <= self.min_sentences:
            return [self._create_chunk(sentences)]

        # Calculate similarity threshold
        self.similarity_threshold = self._calculate_similarity_threshold(sentences)

        # First pass: Group sentences by semantic similarity
        sentence_groups = self._group_sentences(sentences)

        # Second pass: Skip and Merge by semantic similarity
        merged_groups = self._skip_and_merge(sentence_groups, self.similarity_threshold)

        # Second pass: Split groups into size-appropriate chunks
        chunks = self._split_chunks(merged_groups)

        return chunks

    def __repr__(self) -> str:
        """Return a string representation of the SDPMChunker object."""
        return (
            f"SDPMChunker(model={self.embedding_model}, "
            f"chunk_size={self.chunk_size}, "
            f"mode={self.mode}, "
            f"threshold={self.threshold}, "
            f"similarity_window={self.similarity_window}, "
            f"min_sentences={self.min_sentences}, "
            f"min_chunk_size={self.min_chunk_size}, "
            f"min_characters_per_sentence={self.min_characters_per_sentence}, "
            f"threshold_step={self.threshold_step}, "
            f"delim={self.delim}, "
            f"include_delim={self.include_delim}, "
            f"skip_window={self.skip_window}, "
            f"return_type={self.return_type})"
        )
