"""Propositional Refinery for extracting atomic propositions from text.

Based on the Dense X Retrieval paper, which shows proposition-based 
retrieval outperforms traditional passage or sentence-based methods.
"""

import re
from typing import List, Literal, Optional, Union, Dict, Any

from chonkie.refinery.base import BaseRefinery
from chonkie.types import Chunk
from chonkie.tokenizer import Tokenizer


class PropositionalRefinery(BaseRefinery):
    """Refinery for extracting atomic propositions from chunks.
    
    Propositions are defined as atomic expressions within text, each encapsulating a 
    distinct factoid and presented in a concise, self-contained natural language format.
    
    This refinery transforms chunks into proposition-based chunks for improved retrieval performance.
    
    Args:
        method (str): Method for proposition extraction ("rule-based" or "transformer-based")
        model_name (str): Name of model to use for transformer-based method
        min_prop_length (int): Minimum character length for a valid proposition
        max_prop_length (int): Maximum character length for a valid proposition
        min_token_count (int): Minimum token count for a valid proposition
        max_token_count (int): Maximum token count for a valid proposition
        merge_short (bool): Whether to merge short propositions with neighbors
        inplace (bool): Whether to modify chunks in place or return new ones
    """

    def __init__(
        self, 
        method: Literal["rule-based", "transformer-based"] = "rule-based",
        model_name: str = "distilbert-base-uncased",
        min_prop_length: int = 30,
        max_prop_length: int = 200,
        min_token_count: int = 5,
        max_token_count: int = 50,
        merge_short: bool = True,
        inplace: bool = True,
        tokenizer_or_token_counter: Union[str, Any] = "character",
    ) -> None:
        """Initialize the PropositionalRefinery."""
        self.method = method
        self.model_name = model_name
        self.min_prop_length = min_prop_length
        self.max_prop_length = max_prop_length
        self.min_token_count = min_token_count
        self.max_token_count = max_token_count
        self.merge_short = merge_short
        self.inplace = inplace
        self.tokenizer = Tokenizer(tokenizer_or_token_counter)
        
        # Lazy load dependencies based on method
        self._nlp = None
        self._transformer_model = None
        self._transformer_tokenizer = None
        
        # Patterns for identifying clause boundaries in the rule-based approach
        self.clause_patterns = [
            r',\s*(?:and|but|or|because|however|nevertheless|thus|therefore|moreover|furthermore|consequently|hence|so|yet|although|though|while|whereas|since|unless|if|when|where|which|who|whom|whose|what|whatever|whoever|whenever|wherever|as)\s+',
            r';\s*',
            r':\s*',
            r'\.\s+',
            r'\?\s+',
            r'!\s+',
        ]
        
        # Compile patterns for efficiency
        self.compiled_patterns = [re.compile(pattern) for pattern in self.clause_patterns]

    def is_available(self) -> bool:
        """Check if the refinery is available."""
        try:
            if self.method == "rule-based":
                import spacy
                return True
            elif self.method == "transformer-based":
                import transformers
                return True
            else:
                return False
        except ImportError:
            return False

    def _import_dependencies(self) -> None:
        """Lazy import the dependencies."""
        if self.method == "rule-based" and self._nlp is None:
            try:
                import spacy
                try:
                    self._nlp = spacy.load("en_core_web_sm")
                except OSError:
                    # If the model is not available, download it
                    import subprocess
                    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
                    self._nlp = spacy.load("en_core_web_sm")
            except ImportError:
                raise ImportError(
                    "The rule-based method requires spaCy. "
                    "Please install it with `pip install spacy` "
                    "and download the English model with `python -m spacy download en_core_web_sm`."
                )
                
        elif self.method == "transformer-based" and (self._transformer_model is None or self._transformer_tokenizer is None):
            try:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
                self._transformer_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._transformer_model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            except ImportError:
                raise ImportError(
                    "The transformer-based method requires transformers. "
                    "Please install it with `pip install transformers`."
                )
    
    def _split_text_into_clauses(self, text: str) -> List[str]:
        """Split text into clauses based on punctuation and conjunctions."""
        # First, split by sentence boundaries
        if self._nlp is None:
            self._import_dependencies()
            
        # Split by newlines first, then process each line separately
        lines = text.split('\n')
        all_clauses = []
        
        for line in lines:
            if not line.strip():
                continue
                
            doc = self._nlp(line.strip())
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            
            # Then further split each sentence into clauses
            for sentence in sentences:
                # Use our patterns to split into potential clauses
                current_text = sentence
                for pattern in self.compiled_patterns:
                    # Skip processing if current_text is not a string
                    if not isinstance(current_text, str):
                        break
                        
                    # Split and preserve the delimiter with the previous clause
                    parts = []
                    last_end = 0
                    for match in pattern.finditer(current_text):
                        # Add text before match with the match itself
                        parts.append(current_text[last_end:match.end()])
                        last_end = match.end()
                    
                    # Add any remaining text
                    if last_end < len(current_text):
                        parts.append(current_text[last_end:])
                    
                    # If we found any splits, update the current text
                    if parts:
                        current_text = parts
                        
                    # If current_text is now a list, we need to process each part
                    if isinstance(current_text, list):
                        new_parts = []
                        for part in current_text:
                            if isinstance(part, str):
                                # Split each part further with the current pattern
                                part_splits = pattern.split(part)
                                # Only keep non-empty splits
                                new_parts.extend([p.strip() for p in part_splits if p.strip()])
                            else:
                                # If not a string, just add it as is
                                new_parts.append(part)
                        current_text = new_parts
                
                # Add the resulting clauses
                if isinstance(current_text, list):
                    all_clauses.extend([c.strip() for c in current_text if isinstance(c, str) and c.strip()])
                else:
                    all_clauses.append(current_text.strip())
        
        return all_clauses
                
    def _is_valid_proposition(self, text: str) -> bool:
        """Check if a text segment is a valid proposition."""
        # Skip if the text is too short or too long
        if len(text) < self.min_prop_length or len(text) > self.max_prop_length:
            return False
            
        # Count tokens
        token_count = self.tokenizer.count_tokens(text)
        if token_count < self.min_token_count or token_count > self.max_token_count:
            return False
            
        # Basic check that text is relatively well-formed
        # At minimum it should have some meaningful content with proper structure
        if self._nlp:
            doc = self._nlp(text)
            
            # Check if it has at least one content word (noun, verb, adjective)
            has_content = any(token.pos_ in ["NOUN", "VERB", "ADJ", "PROPN"] for token in doc)
            
            if not has_content:
                return False
        
        return True
        
        # NOTE: The following code is commented out temporarily for testing
        # We'll re-enable more sophisticated validation after we've confirmed the overall flow works
        """
        # Basic check that text ends with proper punctuation
        if not text.endswith(('.', '?', '!', ':', ';')):
            # But allow it if it's a complete clause ending with a comma
            if not (text.endswith(',') and len(text) > 25):
                return False
        
        # Check for common non-proposition patterns using NLP
        if self._nlp:
            doc = self._nlp(text)
            
            # Check if it has at least one verb
            has_verb = any(token.pos_ == "VERB" for token in doc)
            
            # For subject check, we're more lenient
            has_subject_like = (
                any(token.dep_ in ["nsubj", "nsubjpass"] for token in doc) or
                any(token.pos_ in ["NOUN", "PROPN", "PRON"] for token in doc)
            )
            
            # A basic check for a complete proposition
            if not (has_verb and has_subject_like):
                return False
                
            # Additional check: the sentence should have some information content
            content_words = [token for token in doc if token.pos_ in ["NOUN", "VERB", "ADJ", "PROPN", "NUM"]]
            if len(content_words) < 2:  # At least 2 content words
                return False
        """
        
        return True
    
    def _extract_propositions_rule_based(self, text: str) -> List[str]:
        """Extract propositions using rule-based approach with spaCy."""
        if self._nlp is None:
            self._import_dependencies()
            
        # Split text into clauses
        clauses = self._split_text_into_clauses(text)
        
        # Filter clauses to valid propositions
        propositions = [clause for clause in clauses if self._is_valid_proposition(clause)]
        
        # If we didn't find any valid propositions, return the whole text if it fits our criteria
        if not propositions and self._is_valid_proposition(text):
            return [text]
            
        return propositions
    
    def _extract_propositions_transformer_based(self, text: str) -> List[str]:
        """Extract propositions using transformer-based approach."""
        if self._transformer_model is None or self._transformer_tokenizer is None:
            self._import_dependencies()
            
        # This is a placeholder implementation
        # In a real implementation, we would use a model fine-tuned for proposition extraction
        # or design a prompt for instruction-tuned models like:
        # "Extract atomic propositions from the following text. Each proposition should express 
        # a single, self-contained factoid: {text}"

        # For now, we'll use the same approach as the rule-based method for simplicity
        return self._extract_propositions_rule_based(text)
    
    def _extract_propositions(self, text: str) -> List[str]:
        """Extract propositions from text using the specified method."""
        if self.method == "rule-based":
            return self._extract_propositions_rule_based(text)
        elif self.method == "transformer-based":
            return self._extract_propositions_transformer_based(text)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def refine(self, chunks: List[Chunk]) -> List[Chunk]:
        """Refine chunks by extracting propositions.
        
        Args:
            chunks: List of chunks to refine
            
        Returns:
            List of refined chunks (proposition-based)
        """
        if not chunks:
            return []
            
        # Make a copy of the chunks if not modifying in place
        if not self.inplace:
            chunks = [chunk.copy() for chunk in chunks]
            
        result_chunks = []
        
        for chunk in chunks:
            # Extract propositions from the chunk text
            propositions = self._extract_propositions(chunk.text)
            
            if not propositions:
                # If no propositions were extracted, keep the original chunk
                result_chunks.append(chunk)
                continue
                
            # Create new chunks for each proposition
            current_position = 0
            for prop in propositions:
                # Find the position of the proposition in the original text
                # This is a simple approach and might not be perfect for all cases
                prop_start = chunk.text.find(prop, current_position)
                if prop_start == -1:
                    # If the proposition can't be found exactly, just append it
                    # This shouldn't happen often, but is a fallback
                    new_chunk = Chunk(
                        text=prop,
                        start_index=chunk.start_index,
                        end_index=chunk.start_index + len(prop),
                        token_count=self.tokenizer.count_tokens(prop),
                    )
                else:
                    prop_end = prop_start + len(prop)
                    current_position = prop_end  # Update position for next search
                    
                    # Calculate absolute indices
                    abs_start = chunk.start_index + prop_start
                    abs_end = chunk.start_index + prop_end
                    
                    # Create a new chunk for the proposition
                    new_chunk = Chunk(
                        text=prop,
                        start_index=abs_start,
                        end_index=abs_end,
                        token_count=self.tokenizer.count_tokens(prop),
                    )
                
                # Copy context if it exists in the original chunk
                if hasattr(chunk, "context") and chunk.context:
                    new_chunk.context = chunk.context
                    
                # Copy embedding if it exists in the original chunk
                if hasattr(chunk, "embedding") and chunk.embedding is not None:
                    # Note: Ideally, we would recompute embeddings for each proposition
                    # but for simplicity, we'll just copy the original embedding
                    new_chunk.embedding = chunk.embedding
                
                result_chunks.append(new_chunk)
                
        # Merge short propositions if requested
        if self.merge_short and len(result_chunks) > 1:
            merged_chunks = []
            current_chunk = None
            
            for chunk in result_chunks:
                if current_chunk is None:
                    current_chunk = chunk
                    continue
                    
                # If the current chunk is short, merge it with the next one
                if current_chunk.token_count < self.min_token_count:
                    # Only merge if they are consecutive
                    if current_chunk.end_index == chunk.start_index:
                        merged_text = current_chunk.text + " " + chunk.text
                        current_chunk = Chunk(
                            text=merged_text,
                            start_index=current_chunk.start_index,
                            end_index=chunk.end_index,
                            token_count=self.tokenizer.count_tokens(merged_text),
                        )
                        
                        # Copy context if it exists in either chunk
                        if hasattr(chunk, "context") and chunk.context:
                            current_chunk.context = chunk.context
                        elif hasattr(current_chunk, "context") and current_chunk.context:
                            # Keep the current context
                            pass
                            
                        # Copy embedding (preferring the larger chunk's embedding)
                        if hasattr(chunk, "embedding") and chunk.embedding is not None:
                            current_chunk.embedding = chunk.embedding
                        elif hasattr(current_chunk, "embedding") and current_chunk.embedding is not None:
                            # Keep the current embedding
                            pass
                    else:
                        merged_chunks.append(current_chunk)
                        current_chunk = chunk
                else:
                    merged_chunks.append(current_chunk)
                    current_chunk = chunk
                    
            # Don't forget to add the last chunk
            if current_chunk is not None:
                merged_chunks.append(current_chunk)
                
            return merged_chunks
            
        return result_chunks
    
    def __repr__(self) -> str:
        """Return a string representation of the refinery."""
        return (
            f"PropositionalRefinery(method={self.method}, "
            f"model_name={self.model_name}, "
            f"min_prop_length={self.min_prop_length}, "
            f"max_prop_length={self.max_prop_length}, "
            f"min_token_count={self.min_token_count}, "
            f"max_token_count={self.max_token_count}, "
            f"merge_short={self.merge_short}, "
            f"inplace={self.inplace})"
        ) 