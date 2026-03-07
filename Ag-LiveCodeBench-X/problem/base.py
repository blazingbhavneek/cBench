from typing import Any, Dict, List, Optional

from client.search import WebSearchAgent
from client.rag import RAGAgent
from client.mcp import MCPAgent
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
        temperature: float = 0.6,
        top_p: float = 0.95,
        reasoning_effort: str = "medium",
        max_retries: int = 3,
        use_rag: bool = False,
        rag_data_dir: str = None,
        rag_embedding_base_url: str = "http://localhost:8000/v1",
        rag_embedding_api_key: str = None,
        rag_embedding_model: str = "bge-m3",
        use_mcp: bool = False,
        mcp_servers: Optional[List[Dict[str, Any]]] = None,
        mcp_config_path: str = None,
        mcp_timeout: int = 30,
        web_search_agent: Optional[WebSearchAgent] = None,
        rag_agent: Optional[RAGAgent] = None,
        mcp_agent: Optional[MCPAgent] = None,
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
            temperature=temperature,
            top_p=top_p,
            reasoning_effort=reasoning_effort,
            max_retries=max_retries,
            use_rag=use_rag,
            rag_data_dir=rag_data_dir,
            rag_embedding_base_url=rag_embedding_base_url,
            rag_embedding_api_key=rag_embedding_api_key,
            rag_embedding_model=rag_embedding_model,
            use_mcp=use_mcp,
            mcp_servers=mcp_servers,
            mcp_config_path=mcp_config_path,
            mcp_timeout=mcp_timeout,
            web_search_agent=web_search_agent,
            rag_agent=rag_agent,
            mcp_agent=mcp_agent,
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
