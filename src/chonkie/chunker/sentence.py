from itertools import accumulate
from typing import List, Union, Optional, Literal
import warnings

# ------------------------------
# Minimal types for demonstration
# ------------------------------

class Sentence:
    def __init__(self, text: str, start_index: int, end_index: int, token_count: int):
        self.text = text
        self.start_index = start_index
        self.end_index = end_index
        self.token_count = token_count

class Chunk:
    def __init__(self, text: str, start_index: int, end_index: int, token_count: int):
        self.text = text
        self.start_index = start_index
        self.end_index = end_index
        self.token_count = token_count

# ------------------------------
# Dummy tokenizer
# ------------------------------
class DummyTokenizer:
    def count_tokens(self, text: str) -> int:
        # Simple word count
        return len(text.split())

    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        return [self.count_tokens(t) for t in texts]

# ------------------------------
# SentenceChunker
# ------------------------------

class SentenceChunker:
    def __init__(
        self,
        tokenizer_or_token_counter: Union[str, DummyTokenizer] = "character",
        chunk_size: int = 50,
        chunk_overlap: int = 0,
        min_sentences_per_chunk: int = 1,
        min_characters_per_sentence: int = 12,
        delim: Union[str, List[str]] = [". ", "! ", "? ", "\n"],
        include_delim: Optional[Literal["prev", "next"]] = "prev",
    ):
        self.tokenizer = tokenizer_or_token_counter if isinstance(tokenizer_or_token_counter, DummyTokenizer) else DummyTokenizer()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_sentences_per_chunk = min_sentences_per_chunk
        self.min_characters_per_sentence = min_characters_per_sentence
        self.delim = delim
        self.include_delim = include_delim
        self.sep = "✄"

    # --- Sentence splitting ---
    def _split_text(self, text: str) -> List[str]:
        t = text
        for c in self.delim:
            if self.include_delim == "prev":
                t = t.replace(c, c + self.sep)
            elif self.include_delim == "next":
                t = t.replace(c, self.sep + c)
            else:
                t = t.replace(c, self.sep)
        splits = [s for s in t.split(self.sep) if s != ""]
        # Merge very short sentences
        current = ""
        sentences = []
        for s in splits:
            if len(s) < self.min_characters_per_sentence:
                current += s
            elif current:
                current += s
                sentences.append(current)
                current = ""
            else:
                sentences.append(s)
            if len(current) >= self.min_characters_per_sentence:
                sentences.append(current)
                current = ""
        if current:
            sentences.append(current)
        return sentences

    def _prepare_sentences(self, text: str) -> List[Sentence]:
        sentence_texts = self._split_text(text)
        if not sentence_texts:
            return []
        positions = []
        current_pos = 0
        for sent in sentence_texts:
            positions.append(current_pos)
            current_pos += len(sent)
        token_counts = self.tokenizer.count_tokens_batch(sentence_texts)
        return [Sentence(sent, pos, pos+len(sent), count)
                for sent, pos, count in zip(sentence_texts, positions, token_counts)]

    def _create_chunk(self, sentences: List[Sentence]) -> Chunk:
        chunk_text = "".join([s.text for s in sentences])
        token_count = self.tokenizer.count_tokens(chunk_text)
        return Chunk(chunk_text, sentences[0].start_index, sentences[-1].end_index, token_count)

    def chunk(self, text: str) -> List[Chunk]:
        if not text.strip():
            return []
        sentences = self._prepare_sentences(text)
        if not sentences:
            return []
        token_sums = list(accumulate([s.token_count for s in sentences], lambda a,b: a+b, initial=0))
        chunks = []
        pos = 0
        while pos < len(sentences):
            target_tokens = token_sums[pos] + self.chunk_size
            split_idx = pos + 1
            for i in range(pos, len(sentences)):
                if token_sums[i+1] - token_sums[pos] > self.chunk_size:
                    break
                split_idx = i + 1
            if split_idx - pos < self.min_sentences_per_chunk:
                split_idx = min(pos + self.min_sentences_per_chunk, len(sentences))
            chunk_sentences = sentences[pos:split_idx]
            chunks.append(self._create_chunk(chunk_sentences))
            # Handle overlap
            if self.chunk_overlap > 0 and split_idx < len(sentences):
                overlap_tokens = 0
                overlap_idx = split_idx - 1
                while overlap_idx > pos and overlap_tokens < self.chunk_overlap:
                    next_tokens = overlap_tokens + sentences[overlap_idx].token_count
                    if next_tokens > self.chunk_overlap:
                        break
                    overlap_tokens = next_tokens
                    overlap_idx -= 1
                pos = overlap_idx + 1
            else:
                pos = split_idx
        return chunks

    def __repr__(self):
        return f"SentenceChunker(chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap})"

# ------------------------------
# TEST
# ------------------------------
if __name__ == "__main__":
    text = ("Hello world! This is a test of the SentenceChunker. "
            "We want to see how it splits sentences. "
            "It should respect chunk_size and overlap. "
            "Let's check if it works properly.")
    
    chunker = SentenceChunker(chunk_size=10, chunk_overlap=5)
    chunks = chunker.chunk(text)
    
    print("Total Chunks:", len(chunks))
    for i, c in enumerate(chunks):
        print(f"Chunk {i+1}: '{c.text}' (Tokens: {c.token_count})")
