"""Pydantic request/response schemas for the Chonkie OSS API.

All chunker endpoints accept a ``text`` field (string or list of strings) plus
chunker-specific configuration parameters.  The response is always a list of
chunk dicts (or a list-of-lists when multiple texts are supplied).
"""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class PipelineStepRequest(BaseModel):
    """A single step in a pipeline."""

    type: Literal["chunk", "refine"]
    chunker: Optional[str] = None
    refinery: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)


class PipelineCreateRequest(BaseModel):
    """Request to create a pipeline."""

    name: str = Field(..., description="Unique pipeline name")
    description: Optional[str] = Field(None, description="Pipeline description")
    steps: List[PipelineStepRequest] = Field(..., description="Pipeline steps")


class PipelineUpdateRequest(BaseModel):
    """Request to partially update a pipeline's name, description, or steps."""

    name: Optional[str] = None
    description: Optional[str] = None
    steps: Optional[List[PipelineStepRequest]] = None


class PipelineExecuteRequest(BaseModel):
    """Request to execute a pipeline on one or more texts."""

    text: Union[str, List[str]] = Field(..., description="Text to process")


class PipelineResponse(BaseModel):
    """Pipeline metadata returned by create, get, update, and list endpoints."""

    id: str
    name: str
    description: Optional[str]
    config: Dict[str, Any]
    created_at: str
    updated_at: str


class ChunkResponse(BaseModel):
    """A single chunk returned by any chunker endpoint."""

    text: str = Field(..., description="Chunk text content")
    start_index: int = Field(..., description="Start character index in the original text")
    end_index: int = Field(..., description="End character index in the original text")
    token_count: int = Field(..., description="Number of tokens in this chunk")


# Flat list for single-text input; list-of-lists for batch input.
ChunkingResponse = Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]


class TokenChunkerRequest(BaseModel):
    """Request model for ``POST /v1/chunk/token``."""

    text: Union[str, List[str]] = Field(..., description="Text or list of texts to chunk")
    tokenizer: str = Field(
        default="character",
        description="Tokenizer to use (e.g. 'character', 'gpt2', 'cl100k_base')",
    )
    chunk_size: int = Field(default=512, ge=1, description="Maximum tokens per chunk")
    chunk_overlap: int = Field(default=0, ge=0, description="Token overlap between chunks")


class SentenceChunkerRequest(BaseModel):
    """Request model for ``POST /v1/chunk/sentence``."""

    text: Union[str, List[str]] = Field(..., description="Text or list of texts to chunk")
    tokenizer: str = Field(
        default="character",
        description="Tokenizer to use",
    )
    chunk_size: int = Field(default=512, ge=1, description="Maximum tokens per chunk")
    chunk_overlap: int = Field(default=0, ge=0, description="Token overlap between chunks")
    min_sentences_per_chunk: int = Field(
        default=1, ge=1, description="Minimum number of sentences per chunk"
    )
    min_characters_per_sentence: int = Field(
        default=12, ge=1, description="Minimum characters required to count as a sentence"
    )
    approximate: bool = Field(
        default=False, description="Use approximate token counting for speed"
    )
    delim: Union[str, List[str]] = Field(
        default=["\n", ". ", "! ", "? "],
        description="Sentence delimiter(s)",
    )
    include_delim: Optional[Literal["prev", "next"]] = Field(
        default="prev",
        description="Attach the delimiter to the previous ('prev') or next ('next') sentence",
    )


class RecursiveChunkerRequest(BaseModel):
    """Request model for ``POST /v1/chunk/recursive``."""

    text: Union[str, List[str]] = Field(..., description="Text or list of texts to chunk")
    tokenizer: str = Field(
        default="character",
        description="Tokenizer to use",
    )
    chunk_size: int = Field(default=512, ge=1, description="Maximum tokens per chunk")
    recipe: str = Field(
        default="default",
        description="Named splitting recipe (e.g. 'default', 'markdown', 'python')",
    )
    lang: str = Field(default="en", description="Language hint for the recipe")
    min_characters_per_chunk: int = Field(
        default=24, ge=1, description="Minimum characters to include in a chunk"
    )


class SemanticChunkerRequest(BaseModel):
    """Request model for ``POST /v1/chunk/semantic``."""

    text: Union[str, List[str]] = Field(..., description="Text or list of texts to chunk")
    embedding_model: str = Field(
        default="minishlab/potion-base-8M",
        description="Sentence-embedding model used to compute semantic similarity",
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Cosine-similarity threshold for splitting",
    )
    chunk_size: int = Field(default=512, ge=1, description="Maximum tokens per chunk")
    similarity_window: int = Field(
        default=3,
        ge=1,
        description="Number of surrounding sentences to consider for similarity",
    )
    min_sentences_per_chunk: int = Field(
        default=1, ge=1, description="Minimum sentences per chunk"
    )
    min_characters_per_sentence: int = Field(
        default=12, ge=1, description="Minimum characters per sentence"
    )
    delim: Union[str, List[str]] = Field(
        default=["\n", ". ", "! ", "? "],
        description="Sentence delimiter(s)",
    )
    include_delim: Optional[Literal["prev", "next"]] = Field(
        default="prev",
        description="Attach delimiter to previous or next sentence",
    )
    skip_window: int = Field(default=0, ge=0, description="Skip window for similarity")
    filter_window: int = Field(default=5, ge=1, description="Savitzky-Golay filter window size")
    filter_polyorder: int = Field(default=3, ge=1, description="Savitzky-Golay polynomial order")
    filter_tolerance: float = Field(
        default=0.2,
        ge=0.0,
        description="Tolerance for detecting peaks in the similarity signal",
    )


class CodeChunkerRequest(BaseModel):
    """Request model for ``POST /v1/chunk/code``."""

    text: Union[str, List[str]] = Field(
        ..., description="Source code or list of source code snippets to chunk"
    )
    tokenizer: str = Field(
        default="character",
        description="Tokenizer to use",
    )
    chunk_size: int = Field(default=512, ge=1, description="Maximum tokens per chunk")
    language: str = Field(
        default="python",
        description="Programming language (e.g. 'python', 'javascript', 'java')",
    )
    include_nodes: bool = Field(
        default=False,
        description="Include AST node metadata in chunk output",
    )


class BaseRefineryRequest(BaseModel):
    """Shared base for refinery requests."""

    chunks: List[Dict[str, Any]] = Field(
        ...,
        description=(
            "List of chunk dicts.  Each dict must contain at least "
            "'text', 'start_index', 'end_index', and 'token_count'."
        ),
    )


class EmbeddingsRefineryRequest(BaseRefineryRequest):
    """Request model for ``POST /v1/refine/embeddings``."""

    embedding_model: str = Field(
        default="text-embedding-3-small",
        description=(
            "Embedding model to use via Catsu.  "
            "Supports any provider auto-detected from the model name "
            "(e.g. 'text-embedding-3-small' for OpenAI, 'embed-english-v3.0' "
            "for Cohere, 'voyage-large-2' for Voyage AI).  "
            "Set the appropriate API key environment variable for your provider."
        ),
    )


class OverlapRefineryRequest(BaseRefineryRequest):
    """Request model for ``POST /v1/refine/overlap``."""

    tokenizer: str = Field(default="character", description="Tokenizer to use")
    context_size: Union[float, int] = Field(
        default=0.25,
        description=(
            "Size of the overlap context.  A float between 0 and 1 is "
            "treated as a fraction of ``chunk_size``; an integer is an "
            "absolute token count."
        ),
    )
    mode: Literal["token", "recursive"] = Field(
        default="token", description="Strategy used to create the overlap window"
    )
    method: Literal["suffix", "prefix"] = Field(
        default="suffix",
        description=(
            "'suffix' appends context from the *previous* chunk; "
            "'prefix' prepends context from the *next* chunk."
        ),
    )
    merge: bool = Field(
        default=True,
        description="Merge the overlap context into the chunk text",
    )
