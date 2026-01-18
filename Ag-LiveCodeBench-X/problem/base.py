from typing import Any, Dict, Optional

from client.search import WebSearchAgent
from client.rag import RAGAgent
from client.agent import AgenticLLMClient

class BaseProblemWrapper:
    """Base class with shared LLM calling logic"""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = None,
        model: str = "nemotron-nano",
        use_thinking_budget: bool = False,
        tokenizer_name_or_path: str = None,
        max_thinking_budget: int = 512,
        max_tokens: int = 2048,
        web_search_agent: Optional[WebSearchAgent] = None,
        rag_agent: Optional[RAGAgent] = None,
        critical_coding_requirements=None
    ):
        self.CRITICAL_CODING_REQUIREMENTS = critical_coding_requirements
        self.llm_client = AgenticLLMClient(
            base_url=base_url,
            api_key=api_key,
            model=model,
            use_thinking_budget=use_thinking_budget,
            tokenizer_name_or_path=tokenizer_name_or_path,
            max_thinking_budget=max_thinking_budget,
            max_tokens=max_tokens,
            web_search_agent=web_search_agent,
            rag_agent=rag_agent,
        )

    def _build_system_prompt(self) -> str:
        """To be implemented by subclasses"""
        raise NotImplementedError

    def _build_user_prompt(self, **kwargs) -> str:
        """To be implemented by subclasses"""
        raise NotImplementedError

    def _parse_response(self, response: Dict[str, Any], **kwargs) -> dict:
        """To be implemented by subclasses"""
        raise NotImplementedError
