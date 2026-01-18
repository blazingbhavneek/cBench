from typing import Any, Dict

from problem.base import BaseProblemWrapper
from utils.utils import extract_code_from_markdown

class SolveProblemWrapper(BaseProblemWrapper):
    """Wrapper that solves programming problems using agentic LLM client"""

    def _build_system_prompt(self) -> str:
        return f"""/no_think You are an expert programmer. Solve the following programming problem.

{self.CRITICAL_CODING_REQUIREMENTS}

Return your response in JSON format with:
- reasoning: Your step-by-step thought process
- solution: The complete code solution in markdown format
"""

    def _build_user_prompt(self, language: str, question_content: str, **kwargs) -> str:
        return f"""Programming Language: {language}

Problem Statement:
{question_content}

Provide a complete solution with reasoning."""

    def _parse_response(
        self, response: Dict[str, Any], question_id: str, **kwargs
    ) -> dict:
        solution = extract_code_from_markdown(response["content"])
        return {
            "solution": solution,
            "reasoning": response["reasoning"],
            "question_id": question_id,
        }

    async def aforward(
        self,
        language: str,
        question_content: str,
        question_id: str,
        private_test_cases=None,
        max_agent_iterations: int = 0,
        summarize_context: bool = False,
    ) -> dict:
        """
        Solve a programming problem.

        Args:
            language: Programming language
            question_content: Problem statement
            question_id: Unique identifier for the problem
            private_test_cases: Optional test cases (unused)
            max_agent_iterations: Max RAG/Web search iterations (0 = no agent behavior)
            summarize_context: Whether to summarize gathered information
        """
        try:
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(
                language=language, question_content=question_content
            )

            response = await self.llm_client.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_agent_iterations=max_agent_iterations,
                summarize_context=summarize_context,
                language=language,
            )

            result = self._parse_response(response, question_id=question_id)

            # Optionally include agent logs
            if "agent_logs" in response:
                result["agent_logs"] = response["agent_logs"]

            return result

        except Exception as e:
            return {
                "solution": None,
                "reasoning": f"Error: {str(e)}",
                "question_id": question_id,
            }
