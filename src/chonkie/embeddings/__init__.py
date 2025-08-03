"""Embeddings classes for text embedding."""

from .auto import AutoEmbeddings
from .base import BaseEmbeddings
from .cohere import CohereEmbeddings
from .gemini import GeminiEmbeddings
from .jina import JinaEmbeddings
from .model2vec import Model2VecEmbeddings
from .openai import OpenAIEmbeddings
from .registry import EmbeddingsRegistry
from .sentence_transformer import SentenceTransformerEmbeddings
from .voyageai import VoyageAIEmbeddings

# Add all embeddings classes to __all__
__all__ = [
    "BaseEmbeddings",
    "Model2VecEmbeddings",
    "SentenceTransformerEmbeddings",
    "OpenAIEmbeddings",
    "CohereEmbeddings",
    "GeminiEmbeddings",
    "AutoEmbeddings",
    "JinaEmbeddings",
    "VoyageAIEmbeddings",
    "EmbeddingsRegistry",
]
