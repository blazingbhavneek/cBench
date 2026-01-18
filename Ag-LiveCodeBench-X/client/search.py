from abc import ABC, abstractmethod
from typing import Any, Dict, List


# Abstract base classes for optional components
class WebSearchAgent(ABC):
    """Base class for web search functionality"""

    @abstractmethod
    async def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search the web for relevant information

        Args:
            query: Search query string
            num_results: Number of results to return

        Returns:
            List of search results with title, url, and snippet
        """

    @abstractmethod
    async def get_context(self, query: str) -> str:
        """
        Get formatted context from search results

        Args:
            query: Search query string

        Returns:
            Formatted context string to add to prompt
        """


# Default empty implementations for optional agents
class NoOpWebSearchAgent(WebSearchAgent):
    """Empty implementation - does nothing"""

    async def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        return []

    async def get_context(self, query: str) -> str:
        return ""
