"""Table chunker for processing markdown tables."""

from typing import Any, Callable, List, Union

from chonkie.chunker.base import BaseChunker
from chonkie.types import Chunk


class TableChunker(BaseChunker):
    """Chunker that chunks tables based on character count on each row."""

    def __init__(
        self,
        tokenizer: Union[str, Callable[[str], int], Any] = "minishlab/potion-base-32M",
        chunk_size: int = 2048,
    ) -> None:
        """Initialize the TableChunker with configuration parameters.

        Args:
            tokenizer: The tokenizer to use for chunking.
            chunk_size: The maximum size of each chunk.

        """
        if isinstance(tokenizer, str):
            super().__init__(tokenizer)
        self.chunk_size = chunk_size

    def chunk(self, table: str) -> List[Chunk]:
        """Chunk the table into smaller tables based on the chunk size.

        Args:
            table: The input markdown table as a string.

        Returns:
            List[MarkdownTable]: A list of MarkdownTable chunks.

        """
        rows = table.strip().split("\n")
        if len(rows) < 2:
            raise ValueError("Table must have at least a header and one row.")

        header = "\n".join(rows[0:2])  # header and separators
        data_rows = rows[2:]

        chunks = []
        current_chunk = [header]
        current_size :int = self.tokenizer.count_tokens(header)
        newline_size = self.tokenizer.count_tokens("\n")

        # split data rows into chunks
        for row in data_rows:
            row_size = self.tokenizer.count_tokens(row)
            # if adding this row exceeds chunk size
            if current_size + newline_size + row_size >= self.chunk_size:
                # only create a new chunk if the current chunk has more than just the header
                if len(current_chunk) > 2:
                    chunks.append("\n".join(current_chunk))
                    current_chunk = [header, row]
                    current_size = self.tokenizer.count_tokens(header) + row_size
                # if the current chunk only has the header, we need to add the row anyway
                else:
                    current_chunk.append(row)
                    current_size += row_size
            # if the current chunk is full, we need to create a new chunk
            else:
                current_chunk.append(row)
                current_size += row_size

        # final chunk
        chunks.append("\n".join(current_chunk))

        return [Chunk(text=chunk) for chunk in chunks]
