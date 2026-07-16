"""Thin wrapper over the external retrieval service's POST API."""

from .client import Chunk, HttpRetrievalClient

__all__ = ["Chunk", "HttpRetrievalClient"]
