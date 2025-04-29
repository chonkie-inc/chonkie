"""Module containing Chonkie's Genies — Generative Inference Engine."""

from .base import BaseGenie
from .gemini import GeminiGenie

# Add all genie classes to __all__
__all__ = [
    "BaseGenie",
    "GeminiGenie",
]