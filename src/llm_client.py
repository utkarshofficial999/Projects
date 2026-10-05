"""Robust OpenAI-compatible LLM client supporting Groq, Ollama, DeepSeek, and OpenAI."""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional
from openai import OpenAI
from src.config import AppConfig

logger = logging.getLogger("GitAgentic.LLM")

class LLMClient:
    """Wrapper around OpenAI-compatible API endpoints."""

    def __init__(self, config: AppConfig):
        self.config = config
        api_key = config.llm_api_key or "ollama"
        self.client = OpenAI(
            base_url=config.llm_base_url,
            api_key=api_key,
        )
        self.model = config.llm_model

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 8192,
        retries: int = 3,
    ) -> str:
        """Call the LLM with retry mechanism."""
        last_error = None
        for attempt in range(1, retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                last_error = e
                logger.warning(
                    f"LLM call attempt {attempt}/{retries} failed: {e}. Retrying in {attempt * 2}s..."
                )
                time.sleep(attempt * 2)

        raise RuntimeError(f"All {retries} LLM attempts failed. Last error: {last_error}")

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        """Generate and parse structured JSON response from the LLM."""
        messages = [
            {
                "role": "system",
                "content": f"{system_prompt}\n\nIMPORTANT: You must return ONLY raw valid JSON. Do not wrap in markdown codeblocks (no ```json or ```). Do not include any explanations outside the JSON."
            },
            {"role": "user", "content": user_prompt}
        ]

        raw = self.chat(messages, temperature=temperature)
        cleaned = self._clean_json_string(raw)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as err:
            logger.error(f"Failed to parse JSON. Raw output was:\n{raw}")
            # Fallback regex extraction if there was surrounding text
            match = re.search(r"(\{.*\}|\[.*\])", raw, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except Exception:
                    pass
            raise ValueError(f"Could not parse valid JSON from LLM: {err}")

    @staticmethod
    def _clean_json_string(s: str) -> str:
        """Strip markdown fences, leading/trailing whitespace."""
        s = s.strip()
        # Remove ```json ... ``` or ``` ... ```
        if s.startswith("```"):
            lines = s.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            s = "\n".join(lines).strip()
        return s
