from bisect import bisect_left
from itertools import accumulate
from typing import Any, Callable, List, Optional, Union

from chonkie.genie import BaseGenie, GeminiGenie
from chonkie.types import Chunk, RecursiveRules

from .base import BaseChunker

PROMPT_TEMPLATE = """<task> You are given a set of texts between the starting tag <passages> and ending tag </passages>. Each text is labeled as 'ID `N`' where 'N' is the passage number. Your task is to find the first passage where the content clearly separates from the previous passages in topic and/or semantics. </task>

<rules>
Follow the following rules while finding the splitting passage:
- Always return the answer as a JSON parsable object with the 'split_index' key having a value of the first passage where the topic changes.
- Avoid very long groups of paragraphs. Aim for a good balance between identifying content shifts and keeping groups manageable.
- If no clear `split_index` is found, return N + 1, where N is the index of the last passage. 
</rules>

<passages>
{passages}
</passages>
"""

class SlumberChunker(BaseChunker):
    """Enhanced SlumberChunker with better readability, logging, and error handling."""

    def __init__(
        self,
        genie: Optional[BaseGenie] = None,
        tokenizer_or_token_counter: Union[str, Callable, Any] = "character",
        chunk_size: int = 2048,
        rules: RecursiveRules = RecursiveRules(),
        candidate_size: int = 128,
        min_characters_per_chunk: int = 24,
        verbose: bool = True,
        prompt_template: str = PROMPT_TEMPLATE,
        allow_overlap: bool = False,
    ) -> None:
        super().__init__(tokenizer_or_token_counter)
        self._import_dependencies()

        self.genie: BaseGenie = genie or GeminiGenie()
        self.chunk_size: int = chunk_size
        self.candidate_size: int = candidate_size
        self.rules: RecursiveRules = rules
        self.min_characters_per_chunk: int = min_characters_per_chunk
        self.verbose: bool = verbose
        self.template: str = prompt_template
        self.sep: str = "✄"
        self._use_multiprocessing: bool = False
        self.allow_overlap: bool = allow_overlap

    # -------------------------
    # Recursive splitting
    # -------------------------
    def _recursive_split(self, text: str, level: int = 0, offset: int = 0) -> List[Chunk]:
        """Recursively split text based on token/candidate rules."""
        if not self.rules.levels or level >= len(self.rules.levels):
            return [Chunk(
                text=text,
                start_index=offset,
                end_index=offset + len(text),
                token_count=self.tokenizer.count_tokens(text)
            )]

        splits: List[str] = self._split_text(text, self.rules.levels[level])
        token_counts: List[int] = self.tokenizer.count_tokens_batch(splits)

        chunks: List[Chunk] = []
        current_offset: int = offset
        for split_text, token_count in zip(splits, token_counts):
            if token_count > self.candidate_size:
                chunks.extend(self._recursive_split(split_text, level + 1, current_offset))
            else:
                chunks.append(Chunk(
                    text=split_text,
                    start_index=current_offset,
                    end_index=current_offset + len(split_text),
                    token_count=token_count
                ))
            current_offset += len(split_text)
        return chunks

    # -------------------------
    # Chunk assembly
    # -------------------------
    def chunk(self, text: str) -> List[Chunk]:
        """Main chunking logic with genie-based split decisions."""
        splits: List[Chunk] = self._recursive_split(text)
        prepared: List[str] = [f"ID {i}: {s.text.strip()}" for i, s in enumerate(splits)]
        cumulative_tokens: List[int] = [0] + list(accumulate(s.token_count for s in splits))

        chunks: List[Chunk] = []
        current_index: int = 0
        token_counter: int = 0

        while current_index < len(splits):
            group_end = min(
                bisect_left(cumulative_tokens, token_counter + self.chunk_size),
                len(splits)
            )
            if group_end <= current_index:
                group_end = current_index + 1

            # Prepare genie prompt
            prompt_text = "\n".join(prepared[current_index:group_end])
            try:
                response_index = int(self.genie.generate_json(prompt_text, self.Split)['split_index'])
                response_index = max(response_index, current_index + 1)
            except Exception:
                response_index = group_end

            chunk_text = "".join(s.text for s in splits[current_index:response_index])
            chunk = Chunk(
                text=chunk_text,
                start_index=splits[current_index].start_index,
                end_index=splits[response_index - 1].end_index,
                token_count=sum(s.token_count for s in splits[current_index:response_index])
            )
            chunks.append(chunk)

            current_index = response_index
            token_counter = cumulative_tokens[current_index]

        return chunks

    # -------------------------
    # Helper: Logging & Imports
    # -------------------------
    def _import_dependencies(self) -> None:
        """Import dependencies required for the SlumberChunker."""
        try:
            from pydantic import BaseModel

            class Split(BaseModel):
                split_index: int

            self.Split = Split

        except ImportError:
            raise ImportError(
                "The SlumberChunker requires pydantic. Install via `pip install chonkie[genie]`."
            )

    def __repr__(self) -> str:
        return (f"SlumberChunker(genie={self.genie}, chunk_size={self.chunk_size}, "
                f"candidate_size={self.candidate_size}, min_characters={self.min_characters_per_chunk})")
