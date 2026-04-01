"""LiteLLM wrapper with Ollama → Groq → OpenAI fallback chain."""

from __future__ import annotations

import json
import logging
import os

import litellm

logger = logging.getLogger(__name__)

# ── Model configuration (override via env vars) ──────────────────────────

# Choose your active AI provider here. 
# Options: "ollama", "openai", "anthropic", "groq", "gemini"
ACTIVE_PROVIDER = os.getenv("ACTIVE_PROVIDER", "ollama")

# Define the default model for each provider
PROVIDER_MODELS = {
    "ollama": "ollama/llama3.2",
    "openai": "openai/gpt-4o",
    "anthropic": "anthropic/claude-3-5-sonnet-latest",
    "groq": "groq/llama-3.1-70b-versatile",
    "gemini": "gemini/gemini-1.5-pro",
}

PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", PROVIDER_MODELS.get(ACTIVE_PROVIDER, "ollama/llama3.2"))
OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")

# Suppress litellm's verbose logging
litellm.suppress_debug_info = True


def _get_api_base(model: str) -> str | None:
    """Return api_base for Ollama models, None otherwise."""
    if model.startswith("ollama"):
        return OLLAMA_API_BASE
    return None


def llm_completion(
    messages: list[dict],
    max_tokens: int = 1024,
    temperature: float = 0.1,
) -> str:
    """Call LLM with automatic fallback. Returns the response text."""
    models = [PRIMARY_MODEL]

    last_error = None
    for model in models:
        try:
            logger.debug("Trying model: %s", model)
            response = litellm.completion(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                api_base=_get_api_base(model),
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
    models = [PRIMARY_MODEL]

    last_error = None
    for model in models:
        try:
            logger.debug("Trying model (JSON mode): %s", model)
            kwargs = dict(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                api_base=_get_api_base(model),
            )
            # JSON mode — not all providers support response_format
            try:
                kwargs["response_format"] = {"type": "json_object"}
                response = litellm.completion(**kwargs)
            except Exception:
                # Retry without response_format for providers that don't support it
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
        # Try to extract JSON from markdown code blocks
        if "```" in raw:
            for block in raw.split("```"):
                cleaned = block.strip()
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:].strip()
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    continue

    # Retry once with explicit instruction
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
