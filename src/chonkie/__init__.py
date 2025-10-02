"""Main package for Chonkie."""
# ruff: noqa: F401

from .chef import (
    BaseChef,
    TextChef,
)
from .chunker import (
    BaseChunker,
    CodeChunker,
    LateChunker,
    NeuralChunker,
    RecursiveChunker,
    SDPMChunker,
    SemanticChunker,
    SentenceChunker,
    SlumberChunker,
    TokenChunker,
)
from .cloud import (
    auth,
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
from .friends import (
    BaseHandshake,
    BasePorter,
    ChromaHandshake,
    JSONPorter,
    PgvectorHandshake,
    QdrantHandshake,
    TurbopufferHandshake,
)
from .genie import (
    BaseGenie,
    GeminiGenie,
    OpenAIGenie,
)
from .pipeline import (
    ComponentRegistry,
    ComponentType,
    Pipeline,
    chef,
    chunker,
    fetcher,
    handshake,
    pipeline_component,
    porter,
    refinery,
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
    CodeChunk,
    Context,
    LanguageConfig,
    LateChunk,
    MergeRule,
    RecursiveChunk,
    RecursiveLevel,
    RecursiveRules,
    SemanticChunk,
    SemanticSentence,
    Sentence,
    SentenceChunk,
    SplitRule,
)
from .utils import (
    Hubbie,
    Visualizer,
)

# This hippo grows with every release 🦛✨~
__version__ = "1.1.1"
__name__ = "chonkie"
__author__ = "🦛 Chonkie Inc"
