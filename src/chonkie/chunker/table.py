"""Table chunker for processing markdown tables."""

import re
from typing import Any, Callable, List, Union

from typing_extensions import Tuple

from chonkie.chunker.base import BaseChunker
from chonkie.types import Chunk, Document, MarkdownDocument


class TableChunker(BaseChunker):
    """Chunker that chunks tables based on character count on each row."""

    def __init__(
        self,
        tokenizer: Union[str, Callable[[str], int], Any] = "character",
        chunk_size: int = 2048,
    ) -> None:
        """Initialize the TableChunker with configuration parameters.

        Args:
            tokenizer: The tokenizer to use for chunking.
            chunk_size: The maximum size of each chunk.

        """
        if isinstance(tokenizer, str):
            super().__init__(tokenizer)
        
        if chunk_size <= 0:
            raise ValueError("Chunk size must be greater than 0.")
        
        self.chunk_size = chunk_size
        self.newline_pattern = re.compile(r"\n(?=\|)")
        self.sep = "✄"
    
    def _split_table(self, table: str) -> Tuple[str, List[str]]:
        # insert separator right after the newline that precedes a pipe
        raw = self.newline_pattern.sub(rf"\n{self.sep}", table)
        chunks = [c for c in raw.split(self.sep) if c]   # keep empty strings away
        header = "".join(chunks[:2])      # header line + separator line
        rows   = chunks[2:]                # data rows still contain their trailing \n
        return header, rows

    def chunk(self, table: str) -> List[Chunk]:
        """Chunk the table into smaller tables based on the chunk size.

        Args:
            table: The input markdown table as a string.

        Returns:
            List[MarkdownTable]: A list of MarkdownTable chunks.

        """
        # Basic validation
        if not table.strip():
            raise ValueError("Table must have at least a header and one row.")

        rows = table.strip().split("\n")
        if len(rows) < 3:  # Need header, separator, and at least one data row
            raise ValueError("Table must have at least a header and one row.")

        # Check if the table size is smaller than the chunk size
        table_token_count = self.tokenizer.count_tokens(table.strip())
        if table_token_count <= self.chunk_size:
            return [Chunk(text=table, token_count=table_token_count, start_index=0, end_index=len(table))]

        header, data_rows = self._split_table(table)

        chunks: List[Chunk] = []
        header_token_count = self.tokenizer.count_tokens(header)
        current_token_count = header_token_count
        current_index = 0
        current_chunk = [header]

        # split data rows into chunks
        for row in data_rows:
            row_size = self.tokenizer.count_tokens(row)
            # if adding this row exceeds chunk size
            if current_token_count + row_size >= self.chunk_size and len(current_chunk) > 1:
                # only create a new chunk if the current chunk has more than just the header
                # if the current chunk only has the header, we need to add the row anyway
                if chunks == []:
                    chunk  = Chunk(
                        text="".join(current_chunk),
                        start_index=current_index,
                        end_index=current_index + len("".join(current_chunk)),
                        token_count=current_token_count
                    )
                    chunks.append(chunk)
                    current_index = chunk.end_index
                else:   
                    chunk_len = len("".join(current_chunk)) - len(header)
                    chunk = Chunk(
                        text="".join(current_chunk),
                        start_index=current_index,
                        end_index=current_index + chunk_len,
                        token_count=current_token_count
                    )
                    chunks.append(chunk)
                    current_index = chunk.end_index
                current_chunk = [header, row]
                current_token_count = header_token_count + row_size
            # if the current chunk is not full, we need to add the row to the current chunk
            else:
                current_chunk.append(row)
                current_token_count += row_size
        
        # if the current chunk is not full, we need to add the row to the current chunk
        if len(current_chunk) > 1:
            chunk_len = len("".join(current_chunk)) - len(header) if chunks != [] else len("".join(current_chunk))
            chunk = Chunk(
                text="".join(current_chunk),
                start_index=current_index,
                end_index=current_index + chunk_len,
                token_count=current_token_count
            )
            chunks.append(chunk)
        
        return chunks
    
    def chunk_document(self, document: Document) -> Document:
        """Chunk a document."""
        if isinstance(document, MarkdownDocument) and document.tables:
            for table in document.tables:
                chunks = self.chunk(table.content)
                for chunk in chunks:
                    chunk.start_index = table.start_index + chunk.start_index
                    chunk.end_index = table.start_index + chunk.end_index
                document.chunks.extend(chunks)
            document.chunks.sort(key=lambda x: x.start_index)
        else: 
            document.chunks = self.chunk(document.content)
        return document
    
    def __repr__(self) -> str:
        """Return a string representation of the TableChunker."""
        return f"TableChunker(tokenizer={self.tokenizer}, chunk_size={self.chunk_size})"