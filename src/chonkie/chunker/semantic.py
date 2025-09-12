"""SemanticChunker with advanced peak detection and window embedding calculation.

This chunker uses peak detection to find split points instead of a simple threshold, 
and calculates window embeddings directly rather than approximating them from sentence embeddings.
It uses Savitzky-Golay filtering for smoother boundary detection.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

if TYPE_CHECKING:
    import numpy as np

from chonkie.embeddings import AutoEmbeddings, BaseEmbeddings
from chonkie.types import Chunk, Sentence
from chonkie.utils import Hubbie

from .base import BaseChunker

# Import the unified split function
try:
    from .c_extensions.split import split_text

    SPLIT_AVAILABLE = True
except ImportError:
    SPLIT_AVAILABLE = False

# Import the optimized Savitzky-Golay filter (pure C implementation)
from .c_extensions.savgol import (
    filter_split_indices,
    find_local_minima_interpolated,
    windowed_cross_similarity,
)


class SemanticChunker(BaseChunker):
    """SemanticChunker uses peak detection to find split points and direct window embedding calculation.

    This chunker improves on traditional semantic chunking by using Savitzky-Golay filtering
    for smoother boundary detection and calculating window embeddings directly for more accurate
    semantic similarity computation.
    """

    def __init__(self, 
                 embedding_model: Union[str, BaseEmbeddings] = "minishlab/potion-base-32M",
                 threshold: float = 0.8,
                 chunk_size: int = 2048,
                 similarity_window: int = 3,
                 min_sentences_per_chunk: int = 1,
                 min_characters_per_sentence: int = 24,
                 delim: Union[str, List[str]] = [". ", "! ", "? ", "\n"],
                 include_delim: Optional[Literal["prev", "next"]] = "prev",
                 skip_window: int = 0,
                 filter_window: int = 5,
                 filter_polyorder: int = 3,
                 filter_tolerance: float = 0.2,
                 **kwargs: Dict[str, Any]) -> None:
        """Initialize the SemanticChunker.

        Args:
            embedding_model: Name of the sentence-transformers model to load
            threshold: Threshold for semantic similarity (0-1)
            chunk_size: Maximum tokens allowed per chunk
            similarity_window: Number of sentences to consider for similarity threshold calculation
            min_sentences_per_chunk: Minimum number of sentences per chunk
            min_characters_per_sentence: Minimum number of characters per sentence
            delim: Delimiter to use for sentence splitting
            include_delim: Whether to include the delimiter in the sentence
            skip_window: Number of groups to skip when merging (0=disabled, >0=enabled)
            filter_window: Window length for the Savitzky-Golay filter
            filter_polyorder: Polynomial order for the Savitzky-Golay filter
            filter_tolerance: Tolerance for the Savitzky-Golay filter
            **kwargs: Additional keyword arguments

        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if similarity_window <= 0:
            raise ValueError("similarity_window must be positive")
        if min_sentences_per_chunk <= 0:
            raise ValueError("min_sentences_per_chunk must be positive")
        if skip_window < 0:
            raise ValueError("skip_window must be non-negative")
        if not isinstance(threshold, (int, float)) or threshold <= 0 or threshold >= 1:
            raise ValueError("threshold must be between 0 and 1")
        if type(delim) not in [str, list]:
            raise ValueError("delim must be a string or list of strings")
        if filter_window <= 0:
            raise ValueError("filter_window must be positive")
        if filter_polyorder < 0 or filter_polyorder >= filter_window:
            raise ValueError("filter_polyorder must be non-negative and less than filter_window")
        if filter_tolerance <= 0 or filter_tolerance >= 1:
            raise ValueError("filter_tolerance must be between 0 and 1")
         
        if isinstance(embedding_model, str):
            self.embedding_model = AutoEmbeddings.get_embeddings(embedding_model)
        elif isinstance(embedding_model, BaseEmbeddings):
            self.embedding_model = embedding_model
        else:
            raise ValueError("embedding_model must be a string or a BaseEmbeddings object")

        # Lazy import dependencies
        self._import_dependencies()

        # Initialize the tokenizer and chunker
        tokenizer = self.embedding_model.get_tokenizer_or_token_counter()
        self.tokenizer = tokenizer  # type: ignore[assignment]
        super().__init__(tokenizer)
        
        # Initialize the chunker parameters
        self.threshold = threshold
        self.chunk_size = chunk_size
        self.similarity_window = similarity_window
        self.min_sentences_per_chunk = min_sentences_per_chunk
        self.skip_window = skip_window
        self.delim = delim
        self.include_delim = include_delim
        self.sep = "✄"
        self.min_characters_per_sentence = min_characters_per_sentence
        self.filter_window = filter_window
        self.filter_polyorder = filter_polyorder
        self.filter_tolerance = filter_tolerance

        # Set the multiprocessing flag to False
        self._use_multiprocessing = False
    
    @classmethod
    def from_recipe(cls, 
                   name: str = "default", 
                   lang: Optional[str] = "en", 
                   path: Optional[str] = None,
                   embedding_model: Union[str, BaseEmbeddings] = "minishlab/potion-base-32M",
                   threshold: float = 0.8,
                   chunk_size: int = 2048, 
                   similarity_window: int = 3,
                   min_sentences_per_chunk: int = 1, 
                   min_characters_per_sentence: int = 24,
                   delim: Union[str, List[str]] = [". ", "! ", "? ", "\n"],
                   include_delim: Optional[Literal["prev", "next"]] = "prev",
                   skip_window: int = 0,
                   filter_window: int = 5,
                   filter_polyorder: int = 3,
                   filter_tolerance: float = 0.2,
                   **kwargs: Dict[str, Any]    
                ) -> "SemanticChunker":
        """Create a SemanticChunker from a recipe.
        
        Args:
            name: The name of the recipe to use.
            lang: The language that the recipe should support.
            path: The path to the recipe to use.
            embedding_model: The embedding model to use.
            threshold: The threshold to use for semantic similarity.
            chunk_size: The maximum tokens allowed per chunk.
            similarity_window: The number of sentences to consider for similarity threshold calculation.
            min_sentences_per_chunk: The minimum number of sentences per chunk.
            min_characters_per_sentence: The minimum number of characters per sentence.
            delim: The delimiter to use for sentence splitting.
            include_delim: Whether to include the delimiter in the sentence.
            skip_window: Window size for merging non-consecutive groups (0 to disable)
            filter_window: Window length for the Savitzky-Golay filter
            filter_polyorder: Polynomial order for the Savitzky-Golay filter
            filter_tolerance: Tolerance for the Savitzky-Golay filter
            **kwargs: Additional keyword arguments

        """
        hub = Hubbie()
        recipe = hub.get_recipe(name, lang, path)
        return cls(
            embedding_model=embedding_model,
            threshold=threshold,
            chunk_size=chunk_size,
            similarity_window=similarity_window,
            min_sentences_per_chunk=min_sentences_per_chunk,
            min_characters_per_sentence=min_characters_per_sentence,
            delim=recipe["recipe"]["delimiters"],
            include_delim=recipe["recipe"]["include_delim"],
            skip_window=skip_window,
            filter_window=filter_window,
            filter_polyorder=filter_polyorder,
            filter_tolerance=filter_tolerance,
            **kwargs,
        )
    
    def _split_sentences(self, text: str) -> List[str]:
        """Fast sentence splitting using unified split function when available.

        This method is faster than using regex for sentence splitting and is more accurate than using the spaCy sentence tokenizer.

        Args:
            text: Input text to be split into sentences

        Returns:
            List of sentences

        """
        if SPLIT_AVAILABLE:
            # Use optimized Cython split function
            return list(
                split_text(
                    text=text,
                    delim=self.delim,
                    include_delim=self.include_delim,
                    min_characters_per_segment=self.min_characters_per_sentence,
                    whitespace_mode=False,
                    character_fallback=True,
                )
            )
        else:
            # Fallback to original Python implementation
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

    def _prepare_sentences(self, text: str) -> List[Sentence]:
        """Prepare the sentences for chunking."""
        # Handle empty or whitespace-only text
        if not text or text.isspace():
            return []
            
        sentences = self._split_sentences(text)
        if not sentences:
            return []
            
        token_counts = self.tokenizer.count_tokens_batch(sentences)  # type: ignore[union-attr]
        return [Sentence(text=s, start_index=i, end_index=i + len(s), token_count=tc) for (i, (s, tc)) in enumerate(zip(sentences, token_counts))]
    
    def _get_sentence_embeddings(self, sentences: List[Sentence]) -> List["np.ndarray"]:
        """Get the embeddings for the sentences."""
        return self.embedding_model.embed_batch([s.text for s in sentences[self.similarity_window:]])

    def _get_window_embeddings(self, sentences: List[Sentence]) -> List["np.ndarray"]:
        """Get the embeddings for the window."""
        paragraphs = []
        for i in range(len(sentences) - self.similarity_window):
            paragraphs.append("".join([s.text for s in sentences[i:i + self.similarity_window]]))
        return self.embedding_model.embed_batch(paragraphs)

    def _get_similarity(self, sentences: List[Sentence]) -> List[float]:
        """Get the similarity between the window and the sentence embeddings."""
        window_embeddings = self._get_window_embeddings(sentences)
        sentence_embeddings = self._get_sentence_embeddings(sentences)
        similarities = [float(self.embedding_model.similarity(w, s)) for w, s in zip(window_embeddings, sentence_embeddings)]
        return similarities
    
    def _get_split_indices(self, similarities: Union[List[float], "np.ndarray"]) -> List[int]:
        """Get split indices using optimized Savitzky-Golay filter with interpolation."""
        # Convert to numpy array if needed
        if not isinstance(similarities, np.ndarray):
            similarities = np.asarray(similarities, dtype=np.float64)
        
        # Handle case where data is too small for the filter window
        if len(similarities) == 0:
            return []
        if len(similarities) < self.filter_window:
            # If data is too small for filter, return boundaries only
            return []
        
        # Use optimized Cython implementation with interpolation
        minima_indices, minima_values = find_local_minima_interpolated(
            similarities,
            window_size=self.filter_window,
            poly_order=self.filter_polyorder,
            tolerance=self.filter_tolerance,
            use_float32=False  # Use float64 for consistency with embeddings
        )
        
        # Handle empty case
        if len(minima_indices) == 0:
            return []
        
        # Filter by percentile and minimum distance
        filtered_indices, _ = filter_split_indices(
            minima_indices,
            minima_values,
            self.threshold,
            self.min_sentences_per_chunk
        )
        split_indices_list = filtered_indices  # Already a list from filter_split_indices
            
        # Add boundaries with window offset
        return ([0] + 
                [int(i + self.similarity_window) for i in split_indices_list] + 
                [len(similarities) + self.similarity_window])
    
    def _compute_group_embeddings_batch(self, groups: List[List[Sentence]]) -> List["np.ndarray"]:
        """Compute embeddings for all groups in batch.
        
        Args:
            groups: List of sentence groups
            
        Returns:
            List of embedding vectors, one for each group

        """
        if not groups:
            return []
            
        # Combine sentences in each group into a single text
        group_texts = []
        for group in groups:
            combined_text = "".join([s.text for s in group])
            group_texts.append(combined_text)
        
        # Get embeddings for all groups in batch
        embeddings = self.embedding_model.embed_batch(group_texts)
        return embeddings
    
    def _get_windowed_similarity(self, sentences: List[Sentence]) -> Union[List[float], "np.ndarray"]:
        """Alternative similarity computation using windowed cross-similarity.
        
        This can be more robust than pairwise window-sentence comparison.
        """
        # Get embeddings for all sentences
        embeddings = self.embedding_model.embed_batch([s.text for s in sentences])
        # Convert embeddings to list of lists for the C extension
        embeddings_list = [emb.tolist() if hasattr(emb, 'tolist') else list(emb) for emb in embeddings]
        result = windowed_cross_similarity(embeddings_list, self.similarity_window * 2 + 1)
        return np.asarray(result)
    
    def _skip_and_merge(self, groups: List[List[Sentence]]) -> List[List[Sentence]]:
        """Merge similar groups considering skip window.
        
        Args:
            groups: List of sentence groups to potentially merge
            
        Returns:
            List of merged groups

        """
        if len(groups) <= 1 or self.skip_window == 0:
            return groups
            
        # Get embeddings for all groups in batch for efficiency
        group_texts = ["".join([s.text for s in group]) for group in groups]
        embeddings = self.embedding_model.embed_batch(group_texts)
        
        merged_groups = []
        i = 0
        
        while i < len(groups):
            if i == len(groups) - 1:
                # Last group, can't merge with anything
                merged_groups.append(groups[i])
                break
                
            # Calculate skip index ensuring it's valid
            skip_index = min(i + self.skip_window + 1, len(groups) - 1)
            
            # Find the best merge candidate within the skip window
            best_similarity = -1.0
            best_idx = -1
            
            # Check similarity with all groups within skip window
            for j in range(i + 1, min(skip_index + 1, len(groups))):
                similarity = float(self.embedding_model.similarity(
                    embeddings[i], 
                    embeddings[j]
                ))
                if similarity >= self.threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_idx = j
            
            if best_idx != -1:
                # Merge groups from i to best_idx (inclusive)
                merged = []
                for k in range(i, best_idx + 1):
                    merged.extend(groups[k])
                merged_groups.append(merged)
                i = best_idx + 1  # Skip past merged groups
            else:
                # No merge possible, add current group
                merged_groups.append(groups[i])
                i += 1
                
        return merged_groups
    
    def _group_sentences(self, sentences: List[Sentence], split_indices: List[int]) -> List[List[Sentence]]:
        """Group the sentences based on the split indices.
        
        Simply groups sentences between split points without any size checking.
        Size enforcement happens later in _split_groups.
        """
        groups = []
        
        # Handle empty split_indices
        if not split_indices:
            # Return all sentences as one group if no splits
            if sentences:
                groups.append(sentences)
            return groups
            
        # Create groups from sentences between split indices
        for i in range(len(split_indices) - 1):
            group = sentences[split_indices[i]:split_indices[i + 1]]
            if group:  # Only add non-empty groups
                groups.append(group)

        # Add the last group if there are remaining sentences
        if len(split_indices) > 0 and split_indices[-1] < len(sentences):
            remaining = sentences[split_indices[-1]:]
            if remaining:
                groups.append(remaining)

        return groups

    def _split_groups(self, groups: List[List[Sentence]]) -> List[List[Sentence]]:
        """Split groups that exceed chunk_size into smaller groups.
        
        Args:
            groups: List of sentence groups
            
        Returns:
            List of groups where each respects the chunk_size limit

        """
        final_groups = []
        
        for group in groups:
            token_count = sum([s.token_count for s in group])
            
            if token_count <= self.chunk_size:
                final_groups.append(group)
            else:
                # Split the group into smaller chunks that respect chunk_size
                current_group = []
                current_token_count = 0
                
                for sentence in group:
                    if current_token_count + sentence.token_count <= self.chunk_size:
                        current_group.append(sentence)
                        current_token_count += sentence.token_count
                    else:
                        if current_group:
                            final_groups.append(current_group)
                        current_group = [sentence]
                        current_token_count = sentence.token_count
                
                if current_group:
                    final_groups.append(current_group)
                    
        return final_groups
    
    def _create_chunks(self, sentence_groups: List[List[Sentence]]) -> List[Chunk]:
        """Create a chunk from the sentence groups."""
        chunks = []
        current_index = 0
        for group in sentence_groups:
            text = "".join([s.text for s in group])
            token_count = sum([s.token_count for s in group])
            chunks.append(Chunk(text=text, start_index=current_index, end_index=current_index + len(text), token_count=token_count))
            current_index += len(text)
        return chunks

    def chunk(self, text: str) -> List[Chunk]:
        """Chunk the text into semantic chunks."""
        # Handle empty text
        if not text or text.isspace():
            return []
            
        # Prepare the sentences
        sentences = self._prepare_sentences(text)
        
        # Handle edge cases - too few sentences
        if len(sentences) <= self.similarity_window:
            # If we have any sentences, return them as a single chunk
            if sentences:
                text = "".join([s.text for s in sentences])
                token_count = sum([s.token_count for s in sentences])
                return [Chunk(
                    text=text,
                    start_index=0,
                    end_index=len(text),
                    token_count=token_count
                )]
            else:
                return []

        # Get the similarities
        similarities = self._get_similarity(sentences)

        # Get the split indices
        split_indices = self._get_split_indices(similarities)

        # Group the sentences into chunks based on split indices
        sentence_groups = self._group_sentences(sentences, split_indices)
        
        # Apply skip-and-merge if skip_window > 0
        if self.skip_window > 0:
            sentence_groups = self._skip_and_merge(sentence_groups)
        
        # Split groups that exceed chunk_size
        final_groups = self._split_groups(sentence_groups)

        # Create the chunks
        chunks = self._create_chunks(final_groups)

        # Return the chunks
        return chunks

    def _import_dependencies(self) -> None:
        """Import the dependencies."""
        global np
        
        # Import NumPy (still needed for array operations)
        import numpy as np
    
    def __repr__(self) -> str: 
        """Return a string representation of the SemanticChunker."""
        return (
            f"SemanticChunker(model={self.embedding_model}, "
            f"chunk_size={self.chunk_size}, "
            f"threshold={self.threshold}, "
            f"similarity_window={self.similarity_window}, "
            f"min_sentences_per_chunk={self.min_sentences_per_chunk}, "
            f"skip_window={self.skip_window}, "
            f"filter_window={self.filter_window}, "
            f"filter_polyorder={self.filter_polyorder}, "
            f"filter_tolerance={self.filter_tolerance})"    
        )
