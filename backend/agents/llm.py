"""LiteLLM wrapper using ``config`` — model chain is local (Ollama) or cloud (Groq → Gemini)."""

from __future__ import annotations

import json
import logging
from typing import Optional

import litellm

from config import get_settings

logger = logging.getLogger(__name__)

litellm.suppress_debug_info = True


def _api_base_for_model(model: str, ollama_base: Optional[str]) -> Optional[str]:
    if model.startswith("ollama"):
        return ollama_base or get_settings().ollama_base_url
    return None


def llm_completion(
    messages: list[dict],
    max_tokens: int = 1024,
    temperature: float = 0.1,
) -> str:
    """Call LLM with automatic fallback through the configured model chain."""
    last_error: Exception | None = None
    for model, ollama_base in get_settings().llm_model_chain:
        try:
            logger.debug("Trying model: %s", model)
            response = litellm.completion(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                api_base=_api_base_for_model(model, ollama_base),
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning("Model %s failed: %s", model, exc)
            last_error = exc

    raise RuntimeError(f"All LLM models failed. Last error: {last_error}")


def llm_completion_json(
    messages: list[dict],
    max_tokens: int = 1024,
    temperature: float = 0.1,
) -> str:
    """Call LLM requesting JSON output. Returns raw JSON string."""
    last_error: Exception | None = None
    for model, ollama_base in get_settings().llm_model_chain:
        try:
            logger.debug("Trying model (JSON mode): %s", model)
            kwargs = dict(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                api_base=_api_base_for_model(model, ollama_base),
            )
            try:
                kwargs["response_format"] = {"type": "json_object"}
                response = litellm.completion(**kwargs)
            except Exception:
                kwargs.pop("response_format", None)
                response = litellm.completion(**kwargs)

            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning("Model %s failed (JSON): %s", model, exc)
            last_error = exc

    raise RuntimeError(f"All LLM models failed (JSON mode). Last error: {last_error}")


def parse_llm_json(raw: str, messages: list[dict] | None = None) -> dict:
    """Parse JSON from LLM output. Retries once on failure if messages provided."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        if "```" in raw:
            for block in raw.split("```"):
                cleaned = block.strip()
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:].strip()
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    continue

    if messages:
        retry_messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": "Your response was not valid JSON. Return ONLY valid JSON with no extra text."},
        ]
        try:
            retry_raw = llm_completion_json(retry_messages)
            return json.loads(retry_raw)
        except (json.JSONDecodeError, RuntimeError):
            pass

    raise ValueError(f"Failed to parse LLM JSON output: {raw[:200]}")
