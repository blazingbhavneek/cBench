from typing import Any, Dict, List

import openai
from transformers import AutoTokenizer


class ThinkingBudgetClient:
    """OpenAI client wrapper with thinking budget support"""

    def __init__(
        self,
        base_url: str,
        tokenizer_name_or_path: str,
        api_key: str = "sk-dummy-key",
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self.client = openai.AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)

    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        max_thinking_budget: int = 512,
        max_tokens: int = 1024,
        **kwargs,
    ) -> Dict[str, Any]:
        assert (
            max_tokens > max_thinking_budget
        ), f"thinking budget must be smaller than maximum new tokens. Given {max_tokens=} and {max_thinking_budget=}"

        # 1. first call chat completion to get reasoning content
        response = await self.client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_thinking_budget, **kwargs
        )
        content = response.choices[0].message.content
        reasoning_content = content if content else ""

        if not "</think>" in reasoning_content:
            # reasoning content is too long, closed with a period (.)
            reasoning_content = f"{reasoning_content}.\n</think>\n\n"

        reasoning_tokens_len = len(
            self.tokenizer.encode(reasoning_content, add_special_tokens=False)
        )
        remaining_tokens = max_tokens - reasoning_tokens_len

        assert (
            remaining_tokens > 0
        ), f"remaining tokens must be positive. Given {remaining_tokens=}. Increase the max_tokens or lower the max_thinking_budget."

        # 2. append reasoning content to messages and call completion
        messages.append({"role": "assistant", "content": reasoning_content})
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            continue_final_message=True,
        )
        response = self.client.completions.create(
            model=model, prompt=prompt, max_tokens=remaining_tokens, **kwargs
        )

        response_data = {
            "reasoning_content": reasoning_content.strip().strip("</think>").strip(),
            "content": response.choices[0].text,
            "finish_reason": response.choices[0].finish_reason,
        }
        return response_data
