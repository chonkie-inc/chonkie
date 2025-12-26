"""Main package for Chonkie."""

# ruff: noqa: F401
# Imports are intentionally unused to expose the package's public API.

from .chef import (
    BaseChef,
    MarkdownChef,
    TableChef,
    TextChef,
)
from .chunker import (
    BaseChunker,
    CodeChunker,
    MarkdownChunker,
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
    AzureOpenAIEmbeddings,
    BaseEmbeddings,
    CohereEmbeddings,
    GeminiEmbeddings,
    JinaEmbeddings,
    LiteLLMEmbeddings,
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
    AzureOpenAIGenie,
    BaseGenie,
    GeminiGenie,
    OpenAIGenie,
)
from .handshakes import (
    BaseHandshake,
    ChromaHandshake,
    ElasticHandshake,
    MilvusHandshake,
    MongoDBHandshake,
    PgvectorHandshake,
    PineconeHandshake,
    QdrantHandshake,
    TurbopufferHandshake,
    WeaviateHandshake,
)
from .pipeline import Pipeline
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
    AutoTokenizer,
    ByteTokenizer,
    CharacterTokenizer,
    Tokenizer,
    TokenizerProtocol,
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
__version__ = "1.5.1"
__name__ = "chonkie"
__author__ = "🦛 Chonkie Inc"


# Add basic package metadata to __all__
__all__ = [
    "__name__",
    "__version__",
    "__author__",
]

# Add all data classes to __all__
__all__ += [
    "Context",
    "Chunk",
    "RecursiveChunk",
    "RecursiveLevel",
    "RecursiveRules",
    "SentenceChunk",
    "SemanticChunk",
    "Sentence",
    "SemanticSentence",
    "LateChunk",
    "CodeChunk",
    "LanguageConfig",
    "MergeRule",
    "SplitRule",
]

# Add all tokenizer classes to __all__
__all__ += [
    "Tokenizer",
    "CharacterTokenizer",
    "WordTokenizer",
]

# Add all chunker classes to __all__
__all__ += [
    "BaseChunker",
    "TokenChunker",
    "SentenceChunker",
    "SemanticChunker",
    "RecursiveChunker",
    "LateChunker",
    "CodeChunker",
    "SlumberChunker",
    "NeuralChunker",
    "MarkdownChunker",
]

# Add all cloud classes to __all__
__all__ += [
    "auth",
    "chunker",
    "refineries",
]

# Add all embeddings classes to __all__
__all__ += [
    "BaseEmbeddings",
    "Model2VecEmbeddings",
    "SentenceTransformerEmbeddings",
    "OpenAIEmbeddings",
    "CohereEmbeddings",
    "GeminiEmbeddings",
    "AutoEmbeddings",
    "JinaEmbeddings",
    "VoyageAIEmbeddings",
]

# Add all refinery classes to __all__
__all__ += [
    "BaseRefinery",
    "OverlapRefinery",
    "EmbeddingsRefinery",
]

# Add all utils classes to __all__
__all__ += [
    "Hubbie",
    "Visualizer",
]

# Add all genie classes to __all__
__all__ += [
    "BaseGenie",
    "GeminiGenie",
    "OpenAIGenie",
]

# Add all friends classes to __all__
__all__ += [
    "BasePorter",
    "BaseHandshake",
    "JSONPorter",
    "ChromaHandshake",
    "MongoDBHandshake",
    "PgvectorHandshake",
    "PineconeHandshake",
    "QdrantHandshake",
    "WeaviateHandshake",
    "TurbopufferHandshake",
]

# Add all the chefs to __all__
__all__ += [
    "BaseChef",
    "TextChef",
]

# Add all the fetchers to __all__
__all__ += [
    "BaseFetcher",
    "FileFetcher",
]
