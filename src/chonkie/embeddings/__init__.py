"""Embeddings classes for text embedding."""

from .auto import AutoEmbeddings
from .azure_openai import AzureOpenAIEmbeddings
from .base import BaseEmbeddings
from .catsu import CatsuEmbeddings
from .cloudflare import CloudflareEmbeddings
from .cohere import CohereEmbeddings
from .deepinfra import DeepInfraEmbeddings
from .gemini import GeminiEmbeddings
from .jina import JinaEmbeddings
from .litellm import LiteLLMEmbeddings
from .mistral import MistralEmbeddings
from .mixedbread import MixedbreadEmbeddings
from .model2vec import Model2VecEmbeddings
from .nomic import NomicEmbeddings
from .openai import OpenAIEmbeddings
from .registry import EmbeddingsRegistry
from .sentence_transformer import SentenceTransformerEmbeddings
from .together import TogetherEmbeddings
from .voyageai import VoyageAIEmbeddings

# Add all embeddings classes to __all__
__all__ = [
    "BaseEmbeddings",
    "Model2VecEmbeddings",
    "SentenceTransformerEmbeddings",
    "OpenAIEmbeddings",
    "AzureOpenAIEmbeddings",
    "CohereEmbeddings",
    "GeminiEmbeddings",
    "AutoEmbeddings",
    "JinaEmbeddings",
    "VoyageAIEmbeddings",
    "CatsuEmbeddings",
    "LiteLLMEmbeddings",
    "EmbeddingsRegistry",
    "MistralEmbeddings",
    "TogetherEmbeddings",
    "MixedbreadEmbeddings",
    "NomicEmbeddings",
    "DeepInfraEmbeddings",
    "CloudflareEmbeddings",
]
