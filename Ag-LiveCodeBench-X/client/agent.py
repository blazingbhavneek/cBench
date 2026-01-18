import json
import openai
from typing import Any, Dict, List, Literal, Optional

from client.rag import RAGAgent, NoOpRAGAgent
from client.search import WebSearchAgent, NoOpWebSearchAgent
from client.thinking import ThinkingBudgetClient

class AgenticLLMClient:
    """
    Agentic LLM client that can iteratively gather information from RAG/Web Search
    before making the final LLM call. Can also act as a simple LLM client.
    """

    SUMMARIZER_PROMPT = """You are a context summarizer. Your job is to take retrieved information and summarize it concisely while preserving all critical technical details.

Retrieved Information:
{context}

Provide a concise summary that keeps all important technical details, code examples, and specific information while removing redundancy."""

    AGENT_DECISION_PROMPT = """Based on the current task and information gathered so far, decide if you need more information.

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
        web_search_agent: Optional[WebSearchAgent] = None,
        rag_agent: Optional[RAGAgent] = None,
    ):
        self.model = model
        self.use_thinking_budget = use_thinking_budget
        self.max_thinking_budget = max_thinking_budget
        self.max_tokens = max_tokens
        self.web_search_agent = (
            web_search_agent if web_search_agent else NoOpWebSearchAgent()
        )
        self.rag_agent = rag_agent if rag_agent else NoOpRAGAgent()

        if use_thinking_budget:
            assert (
                tokenizer_name_or_path
            ), "tokenizer_name_or_path required for thinking budget"
            self.client = ThinkingBudgetClient(
                base_url,
                tokenizer_name_or_path,
                api_key,
            )
        else:
            self.client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def _summarize_context(self, context: str) -> str:
        """Summarize retrieved context to reduce token usage"""
        try:
            messages = [
                {
                    "role": "user",
                    "content": self.SUMMARIZER_PROMPT.format(context=context),
                }
            ]

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1024,
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Summarization failed: {e}, returning original context")
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

        try:
            messages = [
                {
                    "role": "user",
                    "content": self.AGENT_DECISION_PROMPT.format(
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
                response_format={"type": "json_object"},
            )

            decision = json.loads(response.choices[0].message.content)
            return decision
        except Exception as e:
            print(f"Agent decision failed: {e}, stopping information gathering")
            return {"needs_more_info": False, "reason": f"Error: {e}"}

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
            print(f"Information gathering failed for {search_type}: {e}")
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
        Generate a response with optional agentic information gathering.

        Args:
            system_prompt: System prompt for the LLM
            user_prompt: User prompt/task description
            max_agent_iterations: Maximum number of RAG/Web search iterations (0 = no agent behavior)
            summarize_context: Whether to summarize gathered context before adding to prompt
            language: Programming language (used for RAG filtering)

        Returns:
            Dict with 'reasoning', 'content', and optional 'agent_logs'
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

        # Make the final LLM call
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": final_user_prompt},
        ]

        if self.use_thinking_budget:
            response = await self.client.chat_completion(
                model=self.model,
                messages=messages,
                max_thinking_budget=self.max_thinking_budget,
                max_tokens=self.max_tokens,
            )
            reasoning = response["reasoning_content"]
            content = response["content"]
        else:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content

            # Parse JSON response
            try:
                parsed = json.loads(content)
                reasoning = parsed.get("reasoning", "")
                content = parsed.get(
                    "solution", parsed.get("refined_solution", content)
                )
            except json.JSONDecodeError:
                reasoning = ""

        result = {
            "reasoning": reasoning,
            "content": content,
        }

        if agent_logs:
            result["agent_logs"] = agent_logs

        return result
