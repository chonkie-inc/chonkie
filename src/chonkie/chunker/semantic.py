"""Semantic chunking using sentence embeddings."""

import importlib.util as importutil
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

from chonkie.chunker.base import BaseChunker
from chonkie.embeddings.base import BaseEmbeddings
from chonkie.types.semantic import SemanticChunk, SemanticSentence, Sentence
from chonkie.utils import Hubbie

if TYPE_CHECKING:
    import numpy as np


class SemanticChunker(BaseChunker):
    """Chunker that splits text into semantically coherent chunks using embeddings.

    Args:
        embedding_model: Embedding model to use for semantic chunking
        mode: Mode for grouping sentences, either "cumulative" or "window"
        threshold: Threshold for semantic similarity (0-1) or percentile (1-100), defaults to "auto"
        chunk_size: Maximum tokens allowed per chunk
        similarity_window: Number of sentences to consider for similarity threshold calculation
        min_sentences: Minimum number of sentences per chunk
        min_characters_per_sentence: Minimum number of characters per sentence
        min_chunk_size: Minimum number of tokens per sentence (defaults to 2)
        threshold_step: Step size for similarity threshold calculation
        delim: Delimiters to split sentences on
        include_delim: Whether to include the delimiters in the sentences
        return_type: Whether to return chunks or texts

    Raises:
        ValueError: If parameters are invalid

    """

    def __init__(
        self,
        embedding_model: Union[str, BaseEmbeddings] = "minishlab/potion-base-8M",
        mode: str = "window",
        threshold: Union[str, float, int] = "auto",
        chunk_size: int = 512,
        similarity_window: int = 1,
        min_sentences: int = 1,
        min_chunk_size: int = 2,
        min_characters_per_sentence: int = 12,
        threshold_step: float = 0.01,
        delim: Union[str, List[str]] = [".", "!", "?", "\n"],
        include_delim: Union[Literal["prev", "next"], None] = "prev",
        return_type: Literal["chunks", "texts"] = "chunks",
        **kwargs: Dict[str, Any],
    ) -> None:  # type: ignore
        """Initialize the SemanticChunker.

        SemanticChunkers split text into semantically coherent chunks using embeddings.

        Args:
            embedding_model: Name of the sentence-transformers model to load
            mode: Mode for grouping sentences, either "cumulative" or "window"
            threshold: Threshold for semantic similarity (0-1) or percentile (1-100), defaults to "auto"
            chunk_size: Maximum tokens allowed per chunk
            similarity_window: Number of sentences to consider for similarity threshold calculation
            min_sentences: Minimum number of sentences per chunk
            min_characters_per_sentence: Minimum number of characters per sentence
            min_chunk_size: Minimum number of tokens per chunk (and sentence, defaults to 2)
            threshold_step: Step size for similarity threshold calculation
            delim: Delimiters to split sentences on
            include_delim: Whether to include the delimiters in the sentences
            return_type: Whether to return chunks or texts
            **kwargs: Additional keyword arguments

        Raises:
            ValueError: If parameters are invalid
            ImportError: If required dependencies aren't installed

        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if min_chunk_size <= 0:
            raise ValueError("min_chunk_size must be positive")
        if min_sentences <= 0:
            raise ValueError("min_sentences must be positive")
        if similarity_window < 0:
            raise ValueError("similarity_window must be non-negative")
        if threshold_step <= 0 or threshold_step >= 1:
            raise ValueError("threshold_step must be between 0 and 1")
        if mode not in ["cumulative", "window"]:
            raise ValueError("mode must be 'cumulative' or 'window'")
        if type(threshold) not in [str, float, int]:
            raise ValueError("threshold must be a string, float, or int")
        if type(delim) not in [str, list]:
            raise ValueError("delim must be a string or list of strings")
        elif type(threshold) == str and threshold not in ["auto"]:
            raise ValueError("threshold must be 'auto', 'smart', or 'percentile'")
        elif type(threshold) == float and (threshold < 0 or threshold > 1):
            raise ValueError("threshold (float) must be between 0 and 1")
        elif type(threshold) == int and (threshold < 1 or threshold > 100):
            raise ValueError("threshold (int) must be between 1 and 100")
        if return_type not in ["chunks", "texts"]:
            raise ValueError("Invalid return_type. Must be either 'chunks' or 'texts'.")

        # Lazy import dependencies to avoid importing them when not needed
        self._import_dependencies()

        self.mode = mode
        self.chunk_size = chunk_size
        self.threshold = threshold
        self.similarity_window = similarity_window if mode == "window" else 1
        self.min_sentences = min_sentences
        self.min_chunk_size = min_chunk_size
        self.min_characters_per_sentence = min_characters_per_sentence
        self.threshold_step = threshold_step
        self.delim = delim
        self.include_delim = include_delim
        self.sep = "✄"
        self.return_type = return_type

        if isinstance(threshold, float):
            self.similarity_threshold = threshold
            self.similarity_percentile = None
        elif isinstance(threshold, int):
            self.similarity_percentile = threshold
            self.similarity_threshold = None
        else:
            self.similarity_threshold = None
            self.similarity_percentile = None

        if isinstance(embedding_model, BaseEmbeddings):
            self.embedding_model = embedding_model
        elif isinstance(embedding_model, str):
            from chonkie.embeddings.auto import AutoEmbeddings

            self.embedding_model = AutoEmbeddings.get_embeddings(
                embedding_model, **kwargs
            )
        else:
            raise ValueError(f"{embedding_model} is not a valid embedding model")

        # Probably the dependency is not installed
        if self.embedding_model is None:
            raise ImportError(
                f"{embedding_model} is not a valid embedding model",
                "Please install the `semantic` extra to use this feature",
            )

        # Keeping the tokenizer the same as the sentence model is important
        # for the group semantic meaning to be calculated properly
        super().__init__(self.embedding_model.get_tokenizer_or_token_counter())

        # Remove the multiprocessing flag from the base class
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
                    return_type: Literal["chunks", "texts"] = "chunks",
                    **kwargs: Dict[str, Any]    
                    ) -> "SemanticChunker":
        """Create a SemanticChunker from a recipe.

        Args:
            name: The name of the recipe to use.
            lang: The language that the recipe should support.
            path: The path to the recipe to use.
            embedding_model: The embedding model to use.
            mode: The mode to use for grouping sentences.
            threshold: The threshold to use for semantic similarity.
            chunk_size: The maximum tokens allowed per chunk.
            similarity_window: The number of sentences to consider for similarity threshold calculation.
            min_sentences: The minimum number of sentences per chunk.
            min_chunk_size: The minimum number of tokens per chunk.
            min_characters_per_sentence: The minimum number of characters per sentence.
            threshold_step: The step size for similarity threshold calculation.
            return_type: Whether to return chunks or texts.
            **kwargs: Additional keyword arguments.

        Returns:
            SemanticChunker: The created SemanticChunker.

        Raises:
            ValueError: If the recipe is invalid.

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
            return_type=return_type,
            **kwargs,
        ) 

    def _split_sentences(
        self,
        text: str,
    ) -> List[str]:
        """Fast sentence splitting while maintaining accuracy.

        This method is faster than using regex for sentence splitting and is more accurate than using the spaCy sentence tokenizer.

        Args:
            text: Input text to be split into sentences
            delim: Delimiters to split sentences on
            sep: Separator to use when splitting sentences

        Returns:
            List of sentences

        """
        t = text
        for c in self.delim:
            if self.include_delim == "prev":
                t = t.replace(c, c + self.sep)
            elif self.include_delim == "next":
                t = t.replace(c, self.sep + c)
            else:
                t = t.replace(c, self.sep)

        # Initial split
        splits = [s for s in t.split(self.sep) if s != ""]

        # Combine short splits with previous sentence
        current = ""
        sentences = []
        for s in splits:
            # If the split is short, add to current and if long add to sentences
            if len(s) < self.min_characters_per_sentence:
                current += s
            elif current:
                current += s
                sentences.append(current)
                current = ""
            else:
                sentences.append(s)

            # At any point if the current sentence is longer than the min_characters_per_sentence,
            # add it to the sentences
            if len(current) >= self.min_characters_per_sentence:
                sentences.append(current)
                current = ""

        # If there is a current split, add it to the sentences
        if current:
            sentences.append(current)

        return sentences

    def _compute_similarity_threshold(self, all_similarities: List[float]) -> float:
        """Compute similarity threshold based on percentile if specified."""
        if self.similarity_threshold is not None:
            return self.similarity_threshold
        else:
            return float(np.percentile(all_similarities, self.similarity_percentile))

    def _prepare_sentences(self, text: str) -> List[SemanticSentence]:
        """Prepare sentences with precomputed information.

        Args:
            text: Input text to be processed

        Returns:
            List of SemanticSentence objects with precomputed token counts and embeddings

        """
        if not text.strip():
            return []

        # Split text into sentences
        raw_sentences = self._split_sentences(text)

        # Get start and end indices for each sentence
        sentence_indices = []
        current_idx = 0
        for sent in raw_sentences:
            start_idx = text.find(sent, current_idx)
            end_idx = start_idx + len(sent)
            sentence_indices.append((start_idx, end_idx))
            current_idx = end_idx

        # Batch compute embeddings for all sentences
        # The embeddings are computed assuming a similarity window is applied
        # There should be len(raw_sentences) number of similarity groups
        sentence_groups = []
        for i in range(len(raw_sentences)):
            group = []
            # similarity window should consider before and after the current sentence
            for j in range(i - self.similarity_window, i + self.similarity_window + 1):
                if j >= 0 and j < len(raw_sentences):
                    group.append(raw_sentences[j])
            sentence_groups.append("".join(group))
        assert len(sentence_groups) == len(raw_sentences), (
            f"Number of sentence groups ({len(sentence_groups)}) does not match number of raw sentences ({len(raw_sentences)})"
        )
        embeddings = self.embedding_model.embed_batch(sentence_groups)

        # Batch compute token counts
        token_counts = self.tokenizer.count_tokens_batch(raw_sentences)

        # Create Sentence objects with all precomputed information
        sentences = [
            SemanticSentence(
                text=sent,
                start_index=start_idx,
                end_index=end_idx,
                token_count=count,
                embedding=embedding,
            )
            for sent, (start_idx, end_idx), count, embedding in zip(
                raw_sentences, sentence_indices, token_counts, embeddings
            )
        ]

        return sentences

    def _get_semantic_similarity(
        self, embedding1: "np.ndarray", embedding2: "np.ndarray"
    ) -> float:
        """Compute cosine similarity between two embeddings."""
        similarity = self.embedding_model.similarity(embedding1, embedding2)
        return similarity

    def _compute_group_embedding(
        self, sentences: List[SemanticSentence]
    ) -> "np.ndarray":
        """Compute mean embedding for a group of sentences."""
        if len(sentences) == 1:
            return sentences[0].embedding
        else:
            # NOTE: There's a known issue, where while calculating the sentence embeddings special tokens are added
            # but when taking the token count the special tokens are not being counted, which causes a mismatch here.
            # This is a known issue and we're working on a fix. At the moment, the error is minimal and doesn't affect the chunking as much.

            # TODO: Account for embedding model truncating to max_seq_length, which causes a mismatch in the token count.
            return np.divide(
                np.sum(
                    [(sent.embedding * sent.token_count) for sent in sentences],
                    axis=0,
                ),
                np.sum([sent.token_count for sent in sentences]),
                dtype=np.float32,
            )

    def _compute_window_similarities(
        self, sentences: List[SemanticSentence]
    ) -> List[float]:
        """Compute all pairwise similarities between sentences."""
        similarities = [1.0]
        current_sentence_window = [sentences[0]]
        window_embedding = sentences[0].embedding
        for i in range(1, len(sentences)):
            similarities.append(
                self._get_semantic_similarity(window_embedding, sentences[i].embedding)
            )

            # Update the window embedding
            if len(current_sentence_window) < self.similarity_window:
                current_sentence_window.append(sentences[i])
                window_embedding = self._compute_group_embedding(
                    current_sentence_window
                )
            else:
                current_sentence_window.pop(0)
                current_sentence_window.append(sentences[i])
                window_embedding = self._compute_group_embedding(
                    current_sentence_window
                )

        return similarities

    def _get_split_indices(
        self, similarities: List[float], threshold: float = None
    ) -> List[int]:
        """Get indices of sentences to split at."""
        if threshold is None:
            threshold = (
                self.similarity_threshold
                if self.similarity_threshold is not None
                else 0.5
            )

        # get the indices of the sentences that are below the threshold
        splits = [
            i + 1
            for i, s in enumerate(similarities)
            if s <= threshold and i + 1 < len(similarities)
        ]
        # add the start and end of the text
        splits = [0] + splits + [len(similarities)]
        # check if the splits are valid (i.e. there are enough sentences between them)
        i = 0
        while i < len(splits) - 1:
            if splits[i + 1] - splits[i] < self.min_sentences:
                splits.pop(i + 1)
            else:
                i += 1
        return splits

    def _calculate_threshold_via_binary_search(
        self, sentences: List[SemanticSentence]
    ) -> float:
        """Calculate similarity threshold via binary search."""
        # Get the token counts and cumulative token counts
        token_counts = [sent.token_count for sent in sentences]
        cumulative_token_counts = np.cumsum([0] + token_counts)

        # Compute all pairwise similarities
        similarities = self._compute_window_similarities(sentences)

        # get the median and the std for the similarities
        median = np.median(similarities)
        std = np.std(similarities)

        # the threshold is set between 1 std of the median
        low = max(median - 1 * std, 0.0)
        high = min(median + 1 * std, 1.0)

        # set iterations
        iterations = 0

        # initialize threshold
        threshold = (low + high) / 2

        while abs(high - low) > self.threshold_step:
            threshold = (low + high) / 2
            # Get the split indices
            split_indices = self._get_split_indices(similarities, threshold)

            # Get the cumulative token count`s at the split indices

            split_token_counts = np.diff(cumulative_token_counts[split_indices])

            # Get the median of the split token counts
            # median_split_token_count = np.median(split_token_counts)
            # Check if the split respects the chunk size
            # if self.min_chunk_size * 1.1 <= median_split_token_count <= 0.95 * self.chunk_size:
            # break
            # elif median_split_token_count > 0.95 * self.chunk_size:
            # The code is calculating the median of a list of token counts stored in the variable
            # `split_token_counts` using the `np.median()` function from the NumPy library in Python.
            #     low = threshold + self.threshold_step
            # else:
            #     high = threshold - self.threshold_step

            # check if all the split token counts are between the min and max chunk size
            if self.min_chunk_size <= all(split_token_counts) <= self.chunk_size:
                break
            # check if any of the split token counts are greater than the max chunk size
            elif any(split_token_counts) > self.chunk_size:
                low = threshold + self.threshold_step
            # check if any of the split token counts are less than the min chunk size
            else:
                high = threshold - self.threshold_step

            iterations += 1
            if iterations > 10:
                warnings.warn(
                    "Too many iterations in threshold calculation, stopping...",
                    stacklevel=2,
                )
                break

        return threshold

    def _calculate_threshold_via_percentile(
        self, sentences: List[SemanticSentence]
    ) -> float:
        """Calculate similarity threshold via percentile."""
        # Compute all pairwise similarities, since the embeddings are already computed
        # The embeddings are computed assuming a similarity window is applied
        all_similarities = self._compute_window_similarities(sentences)
        return float(np.percentile(all_similarities, 100 - self.similarity_percentile))

    def _calculate_similarity_threshold(
        self, sentences: List[SemanticSentence]
    ) -> float:
        """Calculate similarity threshold either through the smart binary search or percentile."""
        if self.similarity_threshold is not None:
            return self.similarity_threshold
        elif self.similarity_percentile is not None:
            return self._calculate_threshold_via_percentile(sentences)
        else:
            return self._calculate_threshold_via_binary_search(sentences)

    def _group_sentences_cumulative(
        self, sentences: List[SemanticSentence]
    ) -> List[List[SemanticSentence]]:
        """Group sentences based on semantic similarity, ignoring token count limits.

        Args:
            sentences: List of SemanticSentence objects with precomputed embeddings

        Returns:
            List of sentence groups, where each group is semantically coherent

        """
        groups = []
        current_group = sentences[: self.min_sentences]
        current_embedding = self._compute_group_embedding(current_group)

        for sentence in sentences[self.min_sentences :]:
            # Compare new sentence against mean embedding of entire current group
            similarity = self._get_semantic_similarity(
                current_embedding, sentence.embedding
            )

            if similarity >= self.similarity_threshold:
                # Add to current group
                current_group.append(sentence)
                # Update mean embedding
                current_embedding = self._compute_group_embedding(current_group)
            else:
                # Start new group
                if current_group:
                    groups.append(current_group)
                current_group = [sentence]
                current_embedding = sentence.embedding

        # Add final group
        if current_group:
            groups.append(current_group)

        return groups

    def _group_sentences_window(
        self, sentences: List[Sentence]
    ) -> List[List[Sentence]]:
        """Group sentences based on semantic similarity, respecting the similarity window."""
        similarities = self._compute_window_similarities(
            sentences
        )  # NOTE: This is calculating pairwise, but not window.
        split_indices = self._get_split_indices(similarities, self.similarity_threshold)
        groups = [
            sentences[split_indices[i] : split_indices[i + 1]]
            for i in range(len(split_indices) - 1)
        ]
        return groups

    def _group_sentences(self, sentences: List[Sentence]) -> List[List[Sentence]]:
        """Group sentences based on semantic similarity, either cumulatively or by window."""
        if self.mode == "cumulative":
            return self._group_sentences_cumulative(sentences)
        else:
            return self._group_sentences_window(sentences)

    def _create_chunk(self, sentences: List[SemanticSentence]) -> SemanticChunk:
        """Create a chunk from a list of sentences."""
        if not sentences:
            raise ValueError("Cannot create chunk from empty sentence list")
        if self.return_type == "chunks":
            # Compute chunk text and token count from sentences
            text = "".join(sent.text for sent in sentences)
            token_count = sum(sent.token_count for sent in sentences)
            return SemanticChunk(
                text=text,
                start_index=sentences[0].start_index,
                end_index=sentences[-1].end_index,
                token_count=token_count,
                sentences=sentences,
            )
        elif self.return_type == "texts":
            return "".join(sent.text for sent in sentences)
        else:
            raise ValueError("Invalid return_type. Must be either 'chunks' or 'texts'.")

    def _split_chunks(
        self, sentence_groups: List[List[SemanticSentence]]
    ) -> List[SemanticChunk]:
        """Split sentence groups into chunks that respect chunk_size.

        Args:
            sentence_groups: List of semantically coherent sentence groups

        Returns:
            List of SemanticChunk objects

        """
        chunks = []

        for group in sentence_groups:
            current_chunk_sentences = []
            current_tokens = 0

            for sentence in group:
                test_tokens = (
                    current_tokens
                    + sentence.token_count
                    + (1 if current_chunk_sentences else 0)
                )

                if test_tokens <= self.chunk_size:
                    # Add to current chunk
                    current_chunk_sentences.append(sentence)
                    current_tokens = test_tokens
                else:
                    # Create chunk if we have sentences
                    if current_chunk_sentences:
                        chunks.append(self._create_chunk(current_chunk_sentences))

                    # Start new chunk with current sentence
                    current_chunk_sentences = [sentence]
                    current_tokens = sentence.token_count

            # Create final chunk for this group
            if current_chunk_sentences:
                chunks.append(self._create_chunk(current_chunk_sentences))

        return chunks

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

        # Second pass: Split groups into size-appropriate chunks
        chunks = self._split_chunks(sentence_groups)

        return chunks

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

    def __repr__(self) -> str:
        """Return a string representation of the SemanticChunker."""
        return (
            f"SemanticChunker(model={self.embedding_model}, "
            f"chunk_size={self.chunk_size}, "
            f"mode={self.mode}, "
            f"threshold={self.threshold}, "
            f"similarity_window={self.similarity_window}, "
            f"min_sentences={self.min_sentences}, "
            f"min_chunk_size={self.min_chunk_size}, "
            f"return_type={self.return_type})"
        )
