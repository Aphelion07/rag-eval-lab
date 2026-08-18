"""Embedding backends."""

from .base import Embedder, HashingEmbedder
from .ollama import OllamaEmbedder

__all__ = ["Embedder", "HashingEmbedder", "OllamaEmbedder"]
