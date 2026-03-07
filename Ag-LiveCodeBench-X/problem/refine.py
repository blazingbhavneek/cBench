import json
import logging
from typing import Any, Dict

from problem.base import BaseProblemWrapper
from utils.utils import extract_code_from_markdown

logger = logging.getLogger(__name__)

class RefineProblemWrapper(BaseProblemWrapper):
    """Wrapper that refines failed code solutions using agentic LLM client"""

    def _build_system_prompt(self) -> str:
        return f"""/no_think You are an expert programmer specializing in debugging and code refinement.

{self.CRITICAL_CODING_REQUIREMENTS}

Analyze the error and provide:
1. Step-by-step reasoning about what went wrong
2. A complete corrected solution

Return your response in JSON format with:
- reasoning: Step-by-step analysis of the error and your fix
- refined_solution: The complete corrected code in markdown format
"""

    def _build_user_prompt(
        self,
        language: str,
        problem_statement: str,
        original_code: str,
        error_feedback: dict,
        **kwargs,
    ) -> str:
        error_str = json.dumps(error_feedback, indent=2)
        return f"""Programming Language: {language}

Problem Statement:
{problem_statement}

Original Code:
```{language.lower()}
{original_code}
```

Error Details:
{error_str}

Analyze the error and provide a corrected solution."""

    def _parse_response(
        self,
        response: Dict[str, Any],
        question_id: str,
        original_code: str,
        error_feedback: dict,
        language: str,
        problem_statement: str,
        **kwargs,
    ) -> dict:
        # Check for error from retry exhaustion
        if "error" in response:
            logger.error(f"LLM failed for {question_id}: {response['error']}")
            return {
                "refined_code": None,
                "reasoning": f"LLM failed: {response.get('error', 'Unknown error')}",
                "question_id": question_id,
                "original_code": original_code,
                "error_feedback": error_feedback,
                "language": language,
                "problem_statement": problem_statement,
            }
        
        refined_code = extract_code_from_markdown(response["content"])
        return {
            "refined_code": refined_code,
            "reasoning": response["reasoning"],
            "question_id": question_id,
            "original_code": original_code,
            "error_feedback": error_feedback,
            "language": language,
            "problem_statement": problem_statement,
        }

    async def aforward(
        self,
        language: str,
        problem_statement: str,
        original_code: str,
        error_feedback: dict,
        question_id: str,
        max_agent_iterations: int = 0,
        summarize_context: bool = False,
    ) -> dict:
        """
        Refine a failed code solution.

        Args:
            language: Programming language
            problem_statement: Original problem statement
            original_code: Code that failed
            error_feedback: Error details
            question_id: Unique identifier
            max_agent_iterations: Max RAG/Web search iterations (0 = no agent behavior)
            summarize_context: Whether to summarize gathered information
        """
        try:
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(
                language=language,
                problem_statement=problem_statement,
                original_code=original_code,
                error_feedback=error_feedback,
            )

            response = await self.llm_client.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_agent_iterations=max_agent_iterations,
                summarize_context=summarize_context,
                language=language,
            )

            result = self._parse_response(
                response,
                question_id=question_id,
                original_code=original_code,
                error_feedback=error_feedback,
                language=language,
                problem_statement=problem_statement,
            )

            # Optionally include agent logs
            if "agent_logs" in response:
                result["agent_logs"] = response["agent_logs"]

            return result

        except Exception as e:
            logger.exception(f"Unexpected error in aforward for {question_id}: {e}")
            return {
                "refined_code": None,
                "reasoning": f"Error: {str(e)}",
                "question_id": question_id,
                "original_code": original_code,
                "error_feedback": error_feedback,
                "language": language,
                "problem_statement": problem_statement,
            }
