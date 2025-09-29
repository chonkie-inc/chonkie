"""Main package for Chonkie."""

# ruff: noqa: F401
# Imports are intentionally unused to expose the package's public API.

from .chef import (
    BaseChef,
    MarkdownChef,
    TextChef,
)
from .chunker import (
    BaseChunker,
    CodeChunker,
    LateChunker,
    NeuralChunker,
    RecursiveChunker,
    SemanticChunker,
    SentenceChunker,
    SlumberChunker,
    TableChunker,
    TokenChunker,
)
from .cloud import (
    chunker,
    refineries,
)
from .embeddings import (
    AutoEmbeddings,
    BaseEmbeddings,
    CohereEmbeddings,
    GeminiEmbeddings,
    JinaEmbeddings,
    Model2VecEmbeddings,
    OpenAIEmbeddings,
    SentenceTransformerEmbeddings,
    VoyageAIEmbeddings,
)
from .fetcher import (
    BaseFetcher,
    FileFetcher,
)
from .genie import (
    BaseGenie,
    GeminiGenie,
    OpenAIGenie,
)
from .handshakes import (
    BaseHandshake,
    ChromaHandshake,
    MongoDBHandshake,
    PgvectorHandshake,
    PineconeHandshake,
    QdrantHandshake,
    TurbopufferHandshake,
    WeaviateHandshake,
)
from .porters import (
    BasePorter,
    DatasetsPorter,
    JSONPorter,
)
from .refinery import (
    BaseRefinery,
    EmbeddingsRefinery,
    OverlapRefinery,
)
from .tokenizer import (
    CharacterTokenizer,
    Tokenizer,
    WordTokenizer,
)
from .types import (
    Chunk,
    Document,
    LanguageConfig,
    MarkdownCode,
    MarkdownDocument,
    MarkdownTable,
    MergeRule,
    RecursiveLevel,
    RecursiveRules,
    Sentence,
    SplitRule,
)
from .utils import (
    Hubbie,
    Visualizer,
)

# This hippo grows with every release 🦛✨~
__version__ = "1.3.1"
__name__ = "chonkie"
__author__ = "🦛 Chonkie Inc"
