import copy
from typing import Any, Callable, List, Literal, Optional, Tuple, Union

from chonkie.chunker.recursive import RecursiveChunker
from chonkie.chunker.sentence import SentenceChunker
from chonkie.chunker.token import TokenChunker
from chonkie.types.recursive import RecursiveChunk
from chonkie.tokenizer import Tokenizer


class OverlapRefinery:
    def __init__(
        self,
        context_size: int,
        mode: Literal["token", "sentence", "recursive"],
        method: Literal["suffix", "prefix"],
        inplace: bool,
        tokenizer: Union[str, Callable, Any] = Tokenizer("gpt2"),
        merge: bool = True,
        chunk_size: Optional[int] = 512,
    ) -> None:
        """
        Initialize the OverlapRefinery.

        Args:
            context_size (int): Number of tokens to include as overlapping context.
            mode (Literal["token", "sentence", "recursive"]): Chunking mode.
            method (Literal["suffix", "prefix"]): Whether to add the context as a prefix or suffix.
            inplace (bool): If True, modify the existing chunk objects. Otherwise, work on copies.
            tokenizer (Union[str, Callable, Any]): Tokenizer instance or identifier (default: GPT-2 tokenizer).
            merge (bool): If True, merge overlapping context into the chunk text. Otherwise, store separately.
            chunk_size (Optional[int]): Maximum size of a chunk.
        """

        self.context_size = context_size
        self.mode = mode
        self.method = method
        self.inplace = inplace
        self.merge = merge
        self.chunk_size = chunk_size
        self.tokenizer = tokenizer

        if self.context_size < 0:
            raise ValueError("context_size must be a non-negative integer.")
        if self.context_size >= chunk_size:
            raise ValueError("context_size must be less than chunk_size.")
        if chunk_size is None or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        

        if mode == "token":
            self.chunker = TokenChunker(
                tokenizer=self.tokenizer, chunk_size=self.chunk_size
            )
        elif mode == "sentence":
            self.chunker = SentenceChunker(
                tokenizer_or_token_counter=self.tokenizer, chunk_size=self.chunk_size
            )
        elif mode == "recursive":
            self.chunker = RecursiveChunker(
                tokenizer_or_token_counter=self.tokenizer, chunk_size=self.chunk_size
            )
        else:
            raise ValueError(f"Invalid mode: {mode}. Choose 'token', 'sentence', or 'recursive'.")


    def refine_chunks(self, chunks: List[RecursiveChunk], full_text: str) -> List[RecursiveChunk]:
        """
        Refine chunks by adding overlapping context.

        Args:
            chunks (List[RecursiveChunk]): List of chunks to refine.
            full_text (str): Original text from which chunks were derived.

        Returns:
            List[RecursiveChunk]: Refined chunks with added context.
        """
        refined_chunks = chunks if self.inplace else [copy.copy(chunk) for chunk in chunks]

        for chunk in refined_chunks:
            if self.method == "prefix":
                context_text, context_start, _ = self._get_prefix_context(full_text, chunk.start_index)
                if self.merge:
                    chunk.text = context_text + chunk.text
                    chunk.start_index = context_start
                    chunk.token_count += len(self.tokenizer.encode(context_text))
                else:
                    setattr(chunk, "context", context_text)
            elif self.method == "suffix":
                context_text, _, context_end = self._get_suffix_context(full_text, chunk.end_index)
                if self.merge:
                    chunk.text = chunk.text + context_text
                    chunk.end_index = context_end
                    chunk.token_count += len(self.tokenizer.encode(context_text))
                else:
                    setattr(chunk, "context", context_text)

        return refined_chunks


    def _get_prefix_context(self, text: str, start_index: int) -> Tuple[str, int, int]:
        """
        Extract prefix context of context_size tokens before the start index.

        Args:
            text (str): Original text.
            start_index (int): Starting index of the chunk.

        Returns:
            Tuple[str, int, int]: (Context text, new start index, original start index)
        """
        # No prefix if we are at the start of the text.
        if start_index <= 0:
            return "", 0, start_index

        prefix_text = text[:start_index]
        encoding = self.tokenizer.encode(prefix_text)

        # If the encoding object has offsets and ids use them.
        if hasattr(encoding, "offsets") and hasattr(encoding, "ids"):
            token_ids = encoding.ids
            offsets = encoding.offsets
            if len(token_ids) <= self.context_size:
                return prefix_text, 0, start_index
            else:
                context_start = offsets[len(token_ids) - self.context_size][0]
                return prefix_text[context_start:], context_start, start_index
        else:
            # Fallback: assume encoding returns a list of tokens.
            tokens = encoding 
            if len(tokens) <= self.context_size:
                return prefix_text, 0, start_index
            # locating start of the token at position (len(tokens) - context_size).
            candidate_token = tokens[len(tokens) - self.context_size]
            # If a decode function exists, convert the token to a string; otherwise, use str().
            candidate_token_decoded = (
                self.tokenizer.decode([candidate_token])
                if hasattr(self.tokenizer, "decode")
                else str(candidate_token)
            )
            context_start = prefix_text.rfind(candidate_token_decoded)
            if context_start == -1:
                context_start = 0
            return prefix_text[context_start:], context_start, start_index


    def _get_suffix_context(self, text: str, end_index: int) -> Tuple[str, int, int]:
        """
        Extract suffix context of context_size tokens after the end index.

        Args:
            text (str): Original text.
            end_index (int): Ending index of the chunk.

        Returns:
            Tuple[str, int, int]: (Context text, original end index, new end index)
        """
        # No suffix if end_index is at or past end of the text.
        if end_index >= len(text):
            return "", end_index, len(text)

        suffix_text = text[end_index:]
        encoding = self.tokenizer.encode(suffix_text)

        if hasattr(encoding, "offsets") and hasattr(encoding, "ids"):
            token_ids = encoding.ids
            offsets = encoding.offsets
            if len(token_ids) <= self.context_size:
                return suffix_text, end_index, len(text)
            else:
                context_end_offset = offsets[self.context_size - 1][1]
                return suffix_text[:context_end_offset], end_index, end_index + context_end_offset
        else:
            tokens = encoding 
            if len(tokens) <= self.context_size:
                return suffix_text, end_index, len(text)
            # Fallback: decode the first context_size tokens and join their strings.
            if hasattr(self.tokenizer, "decode"):
                selected_tokens = tokens[:self.context_size]
                # Join decoded tokens to form the context.
                context_text = "".join(self.tokenizer.batch_decode(selected_tokens))
            else:
                # If no decode is available, simply join the token representations.
                selected_tokens = tokens[:self.context_size]
                context_text = "".join(str(token) for token in selected_tokens)
            return context_text, end_index, end_index + len(context_text)


    def chunk_and_refine(self, text: str) -> List[RecursiveChunk]:
        """
        Chunk the text and refine chunks with overlapping context in one step.

        Args:
            text (str): Text to process.

        Returns:
            List[RecursiveChunk]: Chunks with added overlapping context.
        """
       
        initial_chunks = self.chunker.chunk(text)
        return self.refine_chunks(initial_chunks, text)