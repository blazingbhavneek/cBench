import json
import openai
import asyncio
import logging
from typing import Any, Dict, List, Literal, Optional

from client.rag import RAGAgent, NoOpRAGAgent, ChromaRAGAgent
from client.mcp import MCPAgent, NoOpMCPAgent, MCPClientAgent
from client.search import WebSearchAgent, NoOpWebSearchAgent
from client.thinking import ThinkingBudgetClient


logger = logging.getLogger(__name__)

REASONING_EFFORT_ORDER = ["high", "medium", "low"]


def _get_degraded_reasoning_effort(current_effort: str) -> Optional[str]:
    """Get the next lower reasoning effort level, or None if already at lowest."""
    try:
        idx = REASONING_EFFORT_ORDER.index(current_effort)
        if idx < len(REASONING_EFFORT_ORDER) - 1:
            return REASONING_EFFORT_ORDER[idx + 1]
        return None
    except ValueError:
        return None


def _is_retryable_error(error: Exception) -> bool:
    """Check if an error is retryable (connection/timeout issues)."""
    if isinstance(error, openai.APIConnectionError):
        return True
    if isinstance(error, openai.APITimeoutError):
        return True
    if isinstance(error, openai.RateLimitError):
        return True
    if isinstance(error, openai.InternalServerError):
        return True
    # Check for common connection error strings
    error_str = str(error).lower()
    if any(x in error_str for x in ["connection", "timeout", "network", "unreachable"]):
        return True
    return False


def _is_token_limit_error(error: Exception, content: str = None) -> bool:
    """Check if error is due to token limit or parsing failure."""
    error_str = str(error).lower()
    if any(x in error_str for x in ["token", "context_length", "max_tokens", "too long"]):
        return True
    # Check for JSON parsing failures (often due to truncated output)
    if isinstance(error, json.JSONDecodeError):
        return True
    # Check if content is empty or None (could be truncated)
    if content is not None and (not content or content.strip() == ""):
        return True
    return False


class AgenticLLMClient:
    """
    Agentic LLM client that can iteratively gather information from RAG/Web Search
    before making the final LLM call. Can also act as a simple LLM client.
    """

    DEFAULT_SUMMARIZER_PROMPT = """You are a context summarizer. Your job is to take retrieved information and summarize it concisely while preserving all critical technical details.

Retrieved Information:
{context}

Provide a concise summary that keeps all important technical details, code examples, and specific information while removing redundancy."""

    DEFAULT_AGENT_DECISION_PROMPT = """Based on the current task and information gathered so far, decide if you need more information.

Task: {task}

Information gathered so far:
{gathered_info}

Respond with JSON:
{{
    "needs_more_info": true/false,
    "reason": "why you need more info or why current info is sufficient",
    "search_query": "what to search for (only if needs_more_info is true)",
    "search_type": "rag" or "web" (only if needs_more_info is true)
}}"""

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
        summarizer_prompt: str = None,
        agent_decision_prompt: str = None,
    ):
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1")
        
        self.model = model
        self.use_thinking_budget = use_thinking_budget
        self.max_thinking_budget = max_thinking_budget
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.reasoning_effort = reasoning_effort
        self.max_retries = max_retries
        self.summarizer_prompt = summarizer_prompt or self.DEFAULT_SUMMARIZER_PROMPT
        self.agent_decision_prompt = agent_decision_prompt or self.DEFAULT_AGENT_DECISION_PROMPT
        self.web_search_agent = (
            web_search_agent if web_search_agent else NoOpWebSearchAgent()
        )
        
        # Initialize RAG agent if requested
        if use_rag:
            if rag_agent:
                # Use provided RAG agent
                self.rag_agent = rag_agent
            elif rag_data_dir:
                # Create Chroma-based RAG agent
                self.rag_agent = ChromaRAGAgent(
                    data_dir=rag_data_dir,
                    embedding_base_url=rag_embedding_base_url,
                    embedding_api_key=rag_embedding_api_key,
                    embedding_model=rag_embedding_model,
                )
                logger.info(f"Initialized ChromaRAGAgent with data dir: {rag_data_dir}")
            else:
                logger.warning("RAG requested but no data_dir provided, using NoOpRAGAgent")
                self.rag_agent = NoOpRAGAgent()
        else:
            # Use NoOp RAG agent (disabled)
            self.rag_agent = rag_agent if rag_agent else NoOpRAGAgent()

        # Initialize MCP agent if requested
        if use_mcp:
            if mcp_agent:
                # Use provided MCP agent
                self.mcp_agent = mcp_agent
            elif mcp_config_path:
                # Load from config file
                from client.mcp import create_mcp_agent_from_config
                self.mcp_agent = create_mcp_agent_from_config(mcp_config_path)
                logger.info(f"Initialized MCPAgent from config: {mcp_config_path}")
            elif mcp_servers:
                # Create from server list
                self.mcp_agent = MCPClientAgent(
                    servers=mcp_servers,
                    timeout=mcp_timeout,
                )
                logger.info(f"Initialized MCPAgent with {len(mcp_servers)} servers")
            else:
                logger.warning("MCP requested but no servers provided, using NoOpMCPAgent")
                self.mcp_agent = NoOpMCPAgent()
        else:
            # Use NoOp MCP agent (disabled)
            self.mcp_agent = mcp_agent if mcp_agent else NoOpMCPAgent()

        if use_thinking_budget:
            assert (
                tokenizer_name_or_path
            ), "tokenizer_name_or_path required for thinking budget"
            self.client = ThinkingBudgetClient(
                base_url,
                tokenizer_name_or_path,
                api_key,
                max_retries=max_retries,
            )
        else:
            self.client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def _retryable_chat_completion(self, messages: List[Dict[str, Any]], **kwargs):
        """
        Internal method to make a chat completion call with retry logic.
        Returns (success, result, current_effort) tuple.
        """
        # Determine reasoning effort levels to try
        if self.reasoning_effort == "high":
            effort_levels = ["high", "medium", "low"]
        else:
            effort_levels = [self.reasoning_effort]

        last_error = None
        for attempt in range(self.max_retries):
            current_effort = effort_levels[min(attempt, len(effort_levels) - 1)]
            
            try:
                # Add reasoning_effort to API call if supported
                api_kwargs = kwargs.copy()
                if current_effort != "medium":  # Don't send default
                    api_kwargs["reasoning_effort"] = current_effort
                
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **api_kwargs
                )
                
                logger.debug(f"LLM call succeeded on attempt {attempt + 1}/{self.max_retries}")
                return True, response, current_effort
                
            except Exception as e:
                last_error = e
                
                if attempt < self.max_retries - 1:
                    if _is_retryable_error(e):
                        logger.warning(
                            f"LLM call failed (attempt {attempt + 1}/{self.max_retries}): "
                            f"{type(e).__name__}: {e}. Retrying with backoff..."
                        )
                        await asyncio.sleep(0.5 * (attempt + 1))
                        continue
                    elif _is_token_limit_error(e):
                        next_effort = _get_degraded_reasoning_effort(current_effort)
                        if next_effort:
                            logger.warning(
                                f"LLM call failed (token limit, attempt {attempt + 1}/{self.max_retries}). "
                                f"Degrading reasoning effort: {current_effort} -> {next_effort}"
                            )
                            continue
                        else:
                            logger.error(
                                f"LLM call failed (token limit, attempt {attempt + 1}/{self.max_retries}): {e}. "
                                f"No lower effort level available."
                            )
                    else:
                        logger.warning(
                            f"LLM call failed (attempt {attempt + 1}/{self.max_retries}): "
                            f"{type(e).__name__}: {e}. Retrying..."
                        )
                        await asyncio.sleep(0.5 * (attempt + 1))
                        continue
                else:
                    logger.error(
                        f"LLM call failed (final attempt {attempt + 1}/{self.max_retries}): "
                        f"{type(e).__name__}: {e}"
                    )
                    break

        return False, None, last_error

    async def _summarize_context(self, context: str) -> str:
        """Summarize retrieved context to reduce token usage"""
        messages = [
            {
                "role": "user",
                "content": self.summarizer_prompt.format(context=context),
            }
        ]

        for attempt in range(self.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=1024,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"Summarization failed (attempt {attempt + 1}/{self.max_retries}): {e}. Retrying..."
                    )
                    await asyncio.sleep(0.5 * (attempt + 1))
                else:
                    logger.error(
                        f"Summarization failed after {self.max_retries} attempts: {e}. "
                        f"Returning original context."
                    )
                    return context
        
        return context

    async def _should_gather_more_info(
        self,
        task_description: str,
        gathered_info: str,
        iteration: int,
        max_iterations: int,
    ) -> Dict[str, Any]:
        """Decide if more information gathering is needed"""
        if iteration >= max_iterations:
            return {"needs_more_info": False, "reason": "Max iterations reached"}

        for attempt in range(self.max_retries):
            try:
                messages = [
                    {
                        "role": "user",
                        "content": self.agent_decision_prompt.format(
                            task=task_description,
                            gathered_info=(
                                gathered_info
                                if gathered_info
                                else "No information gathered yet"
                            ),
                        ),
                    }
                ]

                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=512,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    response_format={"type": "json_object"},
                )

                decision = json.loads(response.choices[0].message.content)
                return decision
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"Agent decision failed (attempt {attempt + 1}/{self.max_retries}): {e}. Retrying..."
                    )
                    await asyncio.sleep(0.5 * (attempt + 1))
                else:
                    logger.error(
                        f"Agent decision failed after {self.max_retries} attempts: {e}. "
                        f"Stopping information gathering."
                    )
                    return {"needs_more_info": False, "reason": f"Error: {e}"}
        
        return {"needs_more_info": False, "reason": "Error in decision making"}

    async def _gather_information(
        self, search_type: Literal["rag", "web"], query: str, language: str = None
    ) -> str:
        """Gather information from RAG or Web Search"""
        try:
            if search_type == "rag":
                context = await self.rag_agent.get_context(
                    query=query, language=language
                )
            else:  # web
                context = await self.web_search_agent.get_context(query=query)

            return context if context else ""
        except Exception as e:
            logger.warning(f"Information gathering failed for {search_type}: {e}")
            return ""

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_agent_iterations: int = 0,
        summarize_context: bool = False,
        language: str = None,
    ) -> Dict[str, Any]:
        """
        Generate a response with optional agentic information gathering and retry logic.

        Args:
            system_prompt: System prompt for the LLM
            user_prompt: User prompt/task description
            max_agent_iterations: Maximum number of RAG/Web search iterations (0 = no agent behavior)
            summarize_context: Whether to summarize gathered context before adding to prompt
            language: Programming language (used for RAG filtering)

        Returns:
            Dict with 'reasoning', 'content', and optional 'agent_logs'.
            On failure, returns {'reasoning': '', 'content': '', 'error': '...'}
        """
        agent_logs = []
        gathered_contexts = []

        # Agentic information gathering loop
        if max_agent_iterations > 0:
            for iteration in range(max_agent_iterations):
                # Decide if we need more info
                gathered_info_summary = (
                    "\n\n".join(gathered_contexts) if gathered_contexts else ""
                )
                decision = await self._should_gather_more_info(
                    task_description=user_prompt,
                    gathered_info=gathered_info_summary,
                    iteration=iteration,
                    max_iterations=max_agent_iterations,
                )

                agent_logs.append({"iteration": iteration, "decision": decision})

                if not decision.get("needs_more_info", False):
                    break

                # Gather information based on decision
                search_query = decision.get("search_query", "")
                search_type = decision.get("search_type", "rag")

                if search_query:
                    context = await self._gather_information(
                        search_type=search_type, query=search_query, language=language
                    )

                    if context:
                        # Optionally summarize to save tokens
                        if summarize_context:
                            context = await self._summarize_context(context)

                        gathered_contexts.append(
                            f"[{search_type.upper()} - {search_query}]\n{context}"
                        )
                        agent_logs[-1]["retrieved_context_length"] = len(context)

        # Build final prompt with gathered context
        final_user_prompt = user_prompt
        if gathered_contexts:
            context_section = "\n\n---\n\n".join(gathered_contexts)
            final_user_prompt = f"""Retrieved Information:
{context_section}

---

{user_prompt}"""

        # Make the final LLM call with retry logic
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": final_user_prompt},
        ]

        logger.debug(f"Making LLM call with {len(messages)} messages")

        if self.use_thinking_budget:
            # ThinkingBudgetClient has its own retry logic (if added)
            try:
                response = await self.client.chat_completion(
                    model=self.model,
                    messages=messages,
                    max_thinking_budget=self.max_thinking_budget,
                    max_tokens=self.max_tokens,
                )
                reasoning = response["reasoning_content"]
                content = response["content"]
            except Exception as e:
                logger.error(f"ThinkingBudgetClient failed: {e}")
                return {
                    "reasoning": "",
                    "content": "",
                    "error": f"ThinkingBudgetClient failed: {str(e)}",
                }
        else:
            success, response, effort_or_error = await self._retryable_chat_completion(
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                response_format={"type": "json_object"},
            )

            if not success:
                return {
                    "reasoning": "",
                    "content": "",
                    "error": f"All {self.max_retries} attempts failed. Last error: {str(effort_or_error)}",
                }

            content = response.choices[0].message.content

            # Parse JSON response
            try:
                parsed = json.loads(content)
                reasoning = parsed.get("reasoning", "")
                content = parsed.get(
                    "solution", parsed.get("refined_solution", content)
                )
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.debug(f"Raw response: {content[:500]}...")
                # Return error - don't retry here as _retryable_chat_completion already did
                return {
                    "reasoning": "",
                    "content": "",
                    "error": f"Failed to parse JSON response: {str(e)}",
                }

        result = {
            "reasoning": reasoning,
            "content": content,
        }

        if agent_logs:
            result["agent_logs"] = agent_logs

        return result
