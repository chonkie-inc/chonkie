"""Summary Refinery for enhancing chunks with document summaries.

Provides mechanisms for generating and attaching summaries to chunks
as additional context for improved retrieval and comprehension.
"""

from typing import Any, Callable, Dict, List, Literal, Optional, Union

from chonkie.refinery.base import BaseRefinery
from chonkie.tokenizer import Tokenizer
from chonkie.types import Chunk


class SummaryRefinery(BaseRefinery):
    """Refinery for enhancing chunks with document summaries.
    
    This refinery generates summaries for chunks and attaches them as
    additional context to improve retrieval and understanding.
    
    Args:
        method (str): Summarization method:
            - "extractive": Extract key sentences (faster, limited quality)
            - "abstractive": Generate new summary text (higher quality, slower)
            - "hybrid": Combine extractive and abstractive approaches
        model_name (str): Model to use for abstractive summarization
        max_summary_length (int): Target maximum length for summaries in characters
        min_length (int): Minimum summary length in characters
        summary_location (str): Where to store/attach the summary:
            - "context": Store in chunk.context metadata only
            - "prepend": Add to beginning of chunk text
            - "append": Add to end of chunk text
        summary_separator (str): Text to separate summary from content
        inplace (bool): Whether to modify chunks in place or return new ones
        tokenizer_or_token_counter: Tokenizer or token counter function
    """

    def __init__(
        self,
        method: Literal["extractive", "abstractive", "hybrid"] = "extractive",
        model_name: str = "t5-small",
        max_summary_length: int = 100,
        min_length: int = 30,
        summary_location: Literal["context", "prepend", "append"] = "context",
        summary_separator: str = " | ",
        inplace: bool = True,
        tokenizer_or_token_counter: Union[str, Callable, Any] = "character",
    ) -> None:
        """Initialize the SummaryRefinery."""
        self.method = method
        self.model_name = model_name
        self.max_summary_length = max_summary_length
        self.min_length = min_length
        self.summary_location = summary_location
        self.summary_separator = summary_separator
        self.inplace = inplace
        self.tokenizer = Tokenizer(tokenizer_or_token_counter)
        
        # Lazy load dependencies
        self._nlp = None
        self._summarizer = None
        self._transformer_model = None
        self._transformer_tokenizer = None

    def is_available(self) -> bool:
        """Check if the refinery is available."""
        try:
            if self.method == "extractive":
                import spacy
                import numpy as np
                return True
            elif self.method == "abstractive":
                import transformers
                return True
            elif self.method == "hybrid":
                import spacy
                import transformers
                import numpy as np
                return True
            else:
                return False
        except ImportError:
            return False

    def _import_dependencies(self) -> None:
        """Lazy import the dependencies."""
        if self.method in ["extractive", "hybrid"]:
            try:
                import spacy
                import numpy as np
                
                # Load spacy for text processing
                try:
                    self._nlp = spacy.load("en_core_web_sm")
                except OSError:
                    # If the model is not available, download it
                    import subprocess
                    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
                    self._nlp = spacy.load("en_core_web_sm")
            except ImportError:
                raise ImportError(
                    "Extractive summarization requires spaCy. "
                    "Please install it with:\n"
                    "`pip install spacy`\n"
                    "and download the English model with:\n"
                    "`python -m spacy download en_core_web_sm`"
                )
                
        if self.method in ["abstractive", "hybrid"]:
            try:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
                
                # Load transformer model and tokenizer for summarization
                self._transformer_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._transformer_model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            except ImportError:
                raise ImportError(
                    "Abstractive summarization requires transformers. "
                    "Please install it with:\n"
                    "`pip install transformers`"
                )

    def _summarize_extractive(self, text: str) -> str:
        """Generate an extractive summary of the text.
        
        Uses a simple algorithm to extract key sentences based on
        term frequency and sentence position.
        
        Args:
            text: The text to summarize
            
        Returns:
            The extractive summary
        """
        if self._nlp is None:
            self._import_dependencies()
            
        # If text is too short, return it as is
        if len(text) <= self.max_summary_length:
            return text
            
        # Process the text with spaCy
        doc = self._nlp(text)
        
        # Get all sentences
        sentences = list(doc.sents)
        
        # If only 1 sentence, return it
        if len(sentences) <= 1:
            return text
            
        # Calculate sentence scores
        scores = {}
        
        # Extract all tokens that are not stopwords or punctuation
        words = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]
        
        # Calculate word frequencies
        from collections import Counter
        word_freq = Counter(words)
        
        # Score each sentence based on word frequency and position
        for i, sentence in enumerate(sentences):
            # Words in this sentence
            sentence_words = [token.text.lower() for token in sentence if not token.is_stop and not token.is_punct]
            
            # Calculate frequency score
            frequency_score = sum(word_freq[word] for word in sentence_words) / len(sentence_words) if sentence_words else 0
            
            # Position score - sentences at beginning and end are more important
            position_score = 0
            if i < len(sentences) / 3:  # First third
                position_score = 0.3
            elif i > 2 * len(sentences) / 3:  # Last third
                position_score = 0.2
                
            # Length score - penalize very short or very long sentences
            length = len(sentence.text)
            length_score = 0.1
            if length < 10:
                length_score = 0
            elif length > 200:
                length_score = 0
                
            # Final score is a weighted combination
            scores[i] = 0.6 * frequency_score + 0.3 * position_score + 0.1 * length_score
            
        # Get top sentences (about 30% of original text)
        num_sentences = max(1, len(sentences) // 3)
        top_sentences = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)[:num_sentences]
        
        # Sort by original position
        top_sentences.sort()
        
        # Construct the summary
        summary = " ".join(sentences[i].text for i in top_sentences)
        
        # Truncate if still too long
        if len(summary) > self.max_summary_length:
            summary = summary[:self.max_summary_length].rsplit(" ", 1)[0] + "..."
            
        return summary

    def _summarize_abstractive(self, text: str) -> str:
        """Generate an abstractive summary of the text using transformers.
        
        Args:
            text: The text to summarize
            
        Returns:
            The abstractive summary
        """
        if self._transformer_model is None or self._transformer_tokenizer is None:
            self._import_dependencies()
            
        # If text is too short, return it as is
        if len(text) <= self.max_summary_length:
            return text
            
        # Prepare inputs for transformer
        # Handle long inputs - truncate to max input length of the model
        max_length = self._transformer_tokenizer.model_max_length
        encoded_input = self._transformer_tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
        
        # Generate summary
        summary_ids = self._transformer_model.generate(
            encoded_input["input_ids"],
            max_length=self.max_summary_length // 4,  # Characters to tokens approximation
            min_length=self.min_length // 4,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True,
        )
        
        # Decode summary
        summary = self._transformer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        return summary

    def _summarize_hybrid(self, text: str) -> str:
        """Generate a summary using both extractive and abstractive methods.
        
        First extracts key sentences, then uses a transformer to refine them.
        
        Args:
            text: The text to summarize
            
        Returns:
            The hybrid summary
        """
        # First get extractive summary
        extractive_summary = self._summarize_extractive(text)
        
        # Then refine with abstractive method
        abstractive_summary = self._summarize_abstractive(extractive_summary)
        
        return abstractive_summary

    def _generate_summary(self, text: str) -> str:
        """Generate a summary of the text using the specified method.
        
        Args:
            text: The text to summarize
            
        Returns:
            The generated summary
        """
        if self.method == "extractive":
            return self._summarize_extractive(text)
        elif self.method == "abstractive":
            return self._summarize_abstractive(text)
        elif self.method == "hybrid":
            return self._summarize_hybrid(text)
        else:
            raise ValueError(f"Unknown summarization method: {self.method}")

    def _apply_summary(self, chunk: Chunk, summary: str) -> Chunk:
        """Apply the summary to the chunk based on configuration.
        
        Args:
            chunk: The chunk to apply summary to
            summary: The generated summary
            
        Returns:
            The chunk with summary applied
        """
        # Create a copy if not modifying in place
        if not self.inplace:
            chunk = chunk.copy()
            
        # Store the summary in context metadata
        if not hasattr(chunk, "context") or chunk.context is None:
            chunk.context = {}
            
        chunk.context["summary"] = summary
        
        # Apply summary to text if specified
        if self.summary_location == "prepend":
            # Make sure we don't add the summary twice if it's already there
            if not chunk.text.startswith(summary):
                new_text = summary + self.summary_separator + chunk.text
                chunk.text = new_text
                # Update token count
                chunk.token_count = self.tokenizer.count_tokens(new_text)
            # No need to update indices as we're not changing position in the document
            
        elif self.summary_location == "append":
            # Make sure we don't add the summary twice if it's already there
            if not chunk.text.endswith(summary):
                new_text = chunk.text + self.summary_separator + summary
                chunk.text = new_text
                # Update token count
                chunk.token_count = self.tokenizer.count_tokens(new_text)
            # No need to update indices as we're not changing position in the document
            
        return chunk
            
    def refine(self, chunks: List[Chunk]) -> List[Chunk]:
        """Refine chunks by adding summaries as context.
        
        Args:
            chunks: List of chunks to refine
            
        Returns:
            List of refined chunks with summaries
        """
        if not chunks:
            return []
            
        # Make copies if not modifying in place
        if not self.inplace:
            chunks = [chunk.copy() for chunk in chunks]
            
        # Generate and apply summaries for each chunk
        refined_chunks = []
        
        for chunk in chunks:
            # Generate summary
            summary = self._generate_summary(chunk.text)
            
            # Apply summary to chunk
            refined_chunk = self._apply_summary(chunk, summary)
            
            refined_chunks.append(refined_chunk)
            
        return refined_chunks
    
    def __repr__(self) -> str:
        """Return a string representation of the refinery."""
        return (
            f"SummaryRefinery(method={self.method}, "
            f"model_name={self.model_name}, "
            f"max_summary_length={self.max_summary_length}, "
            f"min_length={self.min_length}, "
            f"summary_location={self.summary_location}, "
            f"inplace={self.inplace})"
        ) 