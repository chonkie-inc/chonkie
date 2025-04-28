"""Module containing the SlumberChunker."""

from bisect import bisect_left
from itertools import accumulate
from typing import Any, Callable, List, Literal, Optional, Union

from tqdm import tqdm

from chonkie.genie import BaseGenie, GeminiGenie
from chonkie.types import Chunk, RecursiveLevel, RecursiveRules

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
    """SlumberChunker is a chunker based on the LumberChunker — but slightly different."""

    def __init__(self,
                 genie: Optional[BaseGenie] = None, 
                 tokenizer_or_token_counter: Union[str, Callable, Any] = "gpt2",
                 chunk_size: int = 1024,
                 rules: RecursiveRules = RecursiveRules(),
                 candidate_size: int = 128,
                 min_characters_per_chunk: int = 24,
                 return_type: Literal["chunks", "texts"] = "chunks", 
                 verbose: bool = True):
        """Initialize the SlumberChunker.

        Args:
            genie (Optional[BaseGenie]): The genie to use.
            tokenizer_or_token_counter (Union[str, Callable, Any]): The tokenizer or token counter to use.
            chunk_size (int): The size of the chunks to create.
            rules (RecursiveRules): The rules to use to split the candidate chunks.
            candidate_size (int): The size of the candidate splits that the chunker will consider.
            min_characters_per_chunk (int): The minimum number of characters per chunk.
            return_type (Literal["chunks", "texts"]): The type of output to return.
            verbose (bool): Whether to print verbose output.

        """
        # Since the BaseChunker sets and defines the tokenizer for us, we don't have to worry.
        super().__init__(tokenizer_or_token_counter)

        # Lazily import the dependencies
        self._import_dependencies()

        # If the genie is not provided, use the default GeminiGenie
        if genie is None:
            genie = GeminiGenie()

        # Set the parameters for the SlumberChunker
        self.genie = genie
        self.chunk_size = chunk_size
        self.candidate_size = candidate_size
        self.rules = rules
        self.min_characters_per_chunk = min_characters_per_chunk
        self.return_type = return_type
        self.verbose = verbose

        # Set the parameters for the defualt prompt template
        self.template = PROMPT_TEMPLATE
        self.sep = "✄"
        self._CHARS_PER_TOKEN = 6.5

        # Set the _use_multiprocessing to False, since we don't know the 
        # behaviour of the Genie under multiprocessing conditions
        self._use_multiprocessing = False

    #TODO: There's an issue where if the self.candidate_size is too small, the text becomes un-reconstructable. I believe this has something to do with the implementation in this function...
    #TODO: The same error above also causes the start_index and end_index for a couple of splits to be incorrect.
    def _split_text(self, text: str, recursive_level: RecursiveLevel) -> List[str]:
        """Split the text into chunks using the delimiters."""
        # At every delimiter, replace it with the sep
        if recursive_level.whitespace:
            candidate_splits = text.split(" ")

            # Add whitespace back; assumes that the whitespace is uniform across the text
            # if the whitespace is not uniform, the split will not be reconstructable.
            if recursive_level.include_delim == "prev":
                splits = [" "  + split for (i, split) in enumerate(candidate_splits) if i > 0]
            elif recursive_level.include_delim == "next":
                splits = [split + " " for (i, split) in enumerate(candidate_splits) if i < len(candidate_splits) - 1]
            else:
                splits = candidate_splits

        elif recursive_level.delimiters:
            if recursive_level.include_delim == "prev":
                for delimiter in recursive_level.delimiters:
                    text = text.replace(delimiter, delimiter + self.sep)
            elif recursive_level.include_delim == "next":
                for delimiter in recursive_level.delimiters:
                    text = text.replace(delimiter, self.sep + delimiter)
            else:
                for delimiter in recursive_level.delimiters:
                    text = text.replace(delimiter, self.sep)

            splits = [split for split in text.split(self.sep) if split != ""]
        else:
            # Encode, Split, and Decode
            encoded = self.tokenizer.encode(text)
            token_splits = [ encoded[i : i + self.chunk_size]
                for i in range(0, len(encoded), self.chunk_size)
            ]
            splits = self.tokenizer.decode_batch(token_splits)
        
        # Merge short splits
        current = ""
        merged = []
        for split in splits:
            if len(split) < self.min_characters_per_chunk:
                current += split
            elif current:
                current += split
                merged.append(current)
                current = ""
            else:
                merged.append(split)

            if len(current) >= self.min_characters_per_chunk:
                merged.append(current)
                current = ""

        if current:
            merged.append(current)

        splits = merged

        # Some splits may not be meaningful yet.
        # This will be handled during chunk creation.
        return splits

    def _recursive_split(self, text: str, level: int = 0, offset: int=0) -> List[Chunk]:
        """Recursively split the text into chunks."""
        if level >= len(self.rules.levels):
            return [Chunk(text,
                         offset,
                         offset + len(text),
                         self.tokenizer.count_tokens(text))]
        
        # Do the first split based on the level provided
        splits = self._split_text(text, self.rules.levels[level])

        # Calculate the token_count of each of the splits
        token_counts = self.tokenizer.count_tokens_batch(splits)

        # Loop throught the splits to see if any split 
        chunks = []
        current_offset = offset
        for split, token_count in zip(splits, token_counts):
            # If the token_count is more than the self.candidate_size, 
            # then call the recursive_split function on it with a higher level
            if token_count > self.candidate_size:
                child_chunks = self._recursive_split(split, level + 1, current_offset)
                chunks.extend(child_chunks)
            else:
                chunks.append(Chunk(split,
                                    current_offset,
                                    current_offset + len(split),
                                    token_count))
            
            # Add the offset as the length of the split
            current_offset += len(split)

        return chunks
      
    def _prepare_splits(self, splits: List[Chunk]) -> List[str]:
        """Prepare the splits for the chunker."""
        return [f"ID {i}: " + split.text.replace('\n', '').strip() for (i, split) in enumerate(splits)]
    
    def _get_cumulative_token_counts(self, splits: List[Chunk]) -> List[int]:
        """Get the cumulative token counts for the splits."""
        return list(accumulate([0] + [split.token_count for split in splits]))

    # TODO: Fix the type error later
    def chunk(self, text: str) -> List[Chunk]: # type: ignore
        """Chunk the text with the SlumberChunker."""
        splits = self._recursive_split(text, level=0, offset=0)

        # Add the IDS to the splits
        prepared_split_texts = self._prepare_splits(splits)

        # Calculate the cumulative token counts for each split
        cumulative_token_counts = self._get_cumulative_token_counts(splits)
        
        # If self.verbose has been set to True, show a TQDM progress bar for the text
        if self.verbose:
            progress_bar = tqdm(
                total=len(splits),
                desc="🦛",
                unit="split",
                bar_format="{desc} ch{bar:20}nk {percentage:3.0f}% • {n_fmt}/{total_fmt} splits processed [{elapsed}<{remaining}, {rate_fmt}] 🌱",
                ascii=" o",
            ) 

        # Pass the self.chunk_size amount of context through the Genie, 
        # so we can contol how much context the Genie gets as well.
        # This is especially useful for models that don't have long context
        # or exhibit weakend reasoning ability over longer texts. 
        chunks = []
        current_pos = 0
        current_token_count = 0
        while(current_pos < len(splits)):
            # bisect_left can return 0? No because input_size > 0 and first value is 0
            group_end_index = min(bisect_left(cumulative_token_counts, current_token_count + self.chunk_size) - 1, len(splits))

            if group_end_index == current_pos:
                group_end_index += 1

            prompt = self.template.format(passages="\n".join(prepared_split_texts[current_pos:group_end_index]))
            response = int(self.genie.generate_json(prompt, Split)['split_index'])

            # Make sure that the response doesn't bug out and return a index smaller 
            # than the current position
            if current_pos >= response:
                response = current_pos + 1

            chunks.append(Chunk(
                text="".join([split.text for split in splits[current_pos: response]]),
                start_index=splits[current_pos].start_index,
                end_index=splits[response - 1].end_index,
                token_count = sum([split.token_count for split in splits[current_pos: response]])
            ))

            current_token_count = cumulative_token_counts[response]
            current_pos = response

            if self.verbose:
                progress_bar.update(current_pos - progress_bar.n)

        return chunks

    def _import_dependencies(self) -> None:
        """Import the dependencies for the SlumberChunker."""
        try: 
            global BaseModel, Split
            from pydantic import BaseModel

            class Split(BaseModel): # type: ignore
                split_index: int
    
        except ImportError:
            raise ImportError("The SlumberChunker requires the pydantic library to be installed. Please install it using `pip install chonkie[genie]`.")

    def __repr__(self) -> str:
        """Return a string representation of the SlumberChunker."""
        return (f"SlumberChunker(genie={self.genie}," +
                f"tokenizer_or_token_counter={self.tokenizer}, " +
                f"chunk_size={self.chunk_size}, " +
                f"candidate_size={self.candidate_size}, " +
                f"min_characters_per_chunk={self.min_characters_per_chunk}, " +
                f"return_type={self.return_type})" # type: ignore
            )