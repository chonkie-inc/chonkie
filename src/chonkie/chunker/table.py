import re
import warnings
from typing import Any, Callable, List, Union

from chonkie.chunker.base import BaseChunker
from chonkie.types import Chunk, Document, MarkdownDocument


class TableChunker(BaseChunker):
    """Chunker that splits markdown tables into smaller chunks based on token limits per row."""

    def __init__(
        self,
        tokenizer: Union[str, Callable[[str], int], Any] = "character",
        chunk_size: int = 2048,
        min_rows_per_chunk: int = 1,
        verbose: bool = False,
    ) -> None:
        super().__init__(tokenizer) if isinstance(tokenizer, str) else super().__init__(tokenizer)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0.")

        self.chunk_size = chunk_size
        self.min_rows_per_chunk = min_rows_per_chunk
        self.verbose = verbose
        self.newline_pattern = re.compile(r"\n(?=\|)")
        self.sep = "✄"

    # -------------------------
    # Helper methods
    # -------------------------
    def _split_table_rows(self, table: str) -> tuple[str, List[str]]:
        """Split table into header and data rows."""
        # Insert separator after newline that precedes a pipe
        raw = self.newline_pattern.sub(f"\n{self.sep}", table)
        rows = [r for r in raw.split(self.sep) if r]

        if len(rows) < 3:
            raise ValueError("Table must have at least header, separator, and one data row.")

        header = "".join(rows[:2])
        data_rows = rows[2:]
        return header, data_rows

    def _create_chunk(self, rows: List[str], start_index: int, token_count: int) -> Chunk:
        """Create a Chunk object from list of rows."""
        text = "".join(rows)
        return Chunk(
            text=text,
            start_index=start_index,
            end_index=start_index + len(text),
            token_count=token_count,
        )

    # -------------------------
    # Main chunking method
    # -------------------------
    def chunk(self, table: str) -> List[Chunk]:
        """Chunk the markdown table into smaller tables based on token limits."""
        table = table.strip()
        if not table:
            warnings.warn("Empty table received. Skipping chunking.")
            return []

        table_token_count = self.tokenizer.count_tokens(table)
        if table_token_count <= self.chunk_size:
            return [Chunk(text=table, token_count=table_token_count, start_index=0, end_index=len(table))]

        header, data_rows = self._split_table_rows(table)
        header_token_count = self.tokenizer.count_tokens(header)

        chunks: List[Chunk] = []
        rows_buffer = [header]
        buffer_token_count = header_token_count
        global_start_index = 0

        # Precompute token counts for each row
        row_token_counts = [self.tokenizer.count_tokens(row) for row in data_rows]

        for row, row_tokens in zip(data_rows, row_token_counts):
            # Check if adding the row exceeds chunk size
            if buffer_token_count + row_tokens > self.chunk_size and len(rows_buffer) > self.min_rows_per_chunk:
                chunks.append(self._create_chunk(rows_buffer, global_start_index, buffer_token_count))
                global_start_index += len("".join(rows_buffer))
                rows_buffer = [header, row]
                buffer_token_count = header_token_count + row_tokens
            else:
                rows_buffer.append(row)
                buffer_token_count += row_tokens

        # Add remaining rows
        if len(rows_buffer) > 1:
            chunks.append(self._create_chunk(rows_buffer, global_start_index, buffer_token_count))

        if self.verbose:
            print(f"Table split into {len(chunks)} chunks.")

        return chunks

    # -------------------------
    # Document chunking
    # -------------------------
    def chunk_document(self, document: Document) -> Document:
        """Chunk all tables in a document."""
        if isinstance(document, MarkdownDocument):
            for table in getattr(document, "tables", []):
                chunks = self.chunk(table.content)
                for chunk in chunks:
                    chunk.start_index += table.start_index
                    chunk.end_index += table.start_index
                document.chunks.extend(chunks)
            document.chunks.sort(key=lambda x: x.start_index)
        else:
            document.chunks.extend(self.chunk(getattr(document, "content", "")))
            document.chunks.sort(key=lambda x: x.start_index)
        return document

    def __repr__(self) -> str:
        return f"TableChunker(tokenizer={self.tokenizer}, chunk_size={self.chunk_size})"
