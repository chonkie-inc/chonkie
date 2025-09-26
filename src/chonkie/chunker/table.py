"""Table chunker for processing markdown tables."""

from typing import List, Union

from chonkie.chunker.base import BaseChunker
from chonkie.embeddings import BaseEmbeddings
from chonkie.types import Chunk


class TableChunker(BaseChunker):
    """Chunker that chunks tables based on character count on each row."""

    def __init__(
        self,
        embedding_model: Union[str, BaseEmbeddings] = "minishlab/potion-base-32M",
        chunk_size: int = 2048,
    ) -> None:
        """Initialize the TableChunker with configuration parameters.

        Args:
            embedding_model: The embedding model to use for chunking.
            chunk_size: The maximum size of each chunk.

        """
        if isinstance(embedding_model, str):
            super().__init__(embedding_model)
        self.embedding_model = embedding_model
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

        header = rows[0]
        separator = rows[1]
        data_rows = rows[2:]

        chunks = []
        current_chunk = [header, separator]
        current_size = len(header) + len(separator) + 2  # +2 for newlines

        for row in data_rows:
            row_size = len(row) + 1  # +1 for newline
            if current_size + row_size > self.chunk_size and len(current_chunk) > 2:
                chunks.append("\n".join(current_chunk))
                current_chunk = [header, separator, row]
                current_size = len(header) + len(separator) + row_size + 2
            else:
                current_chunk.append(row)
                current_size += row_size

        # final chunk if it exists
        if len(current_chunk) > 2:
            chunks.append("\n".join(current_chunk))

        return [Chunk(text=chunk) for chunk in chunks]
