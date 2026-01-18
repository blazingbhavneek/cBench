from abc import ABC, abstractmethod
from typing import Any, Dict, List


class RAGAgent(ABC):
    """Base class for RAG functionality for language/library documentation"""

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documentation chunks

        Args:
            query: Query string
            top_k: Number of chunks to retrieve

        Returns:
            List of relevant documentation chunks
        """

    @abstractmethod
    async def get_context(self, query: str, language: str = None) -> str:
        """
        Get formatted documentation context

        Args:
            query: Query describing what documentation is needed
            language: Programming language filter (e.g., "C", "Python")

        Returns:
            Formatted documentation context string to add to prompt
        """

    @abstractmethod
    async def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add new documentation to the RAG store

        Args:
            documents: List of documents with 'content', 'source', 'language' fields
        """


class NoOpRAGAgent(RAGAgent):
    """Empty implementation - does nothing"""

    async def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return []

    async def get_context(self, query: str, language: str = None) -> str:
        return ""

    async def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        pass
