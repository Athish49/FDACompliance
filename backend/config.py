"""
Application configuration loaded from `backend/.env`.

Set ``ENVIRONMENT`` to ``local`` (default) or ``cloud`` to select URL defaults,
LLM chain, and Qdrant target. Values are read from the environment with
`python-dotenv` (`.env` next to this file) and are available via `get_settings()`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(_env_path)

EnvironmentName = Literal["local", "cloud"]


def _norm_env() -> EnvironmentName:
    raw = (os.getenv("ENVIRONMENT") or "local").strip().lower()
    if raw == "cloud":
        return "cloud"
    return "local"


@dataclass(frozen=True)
class Settings:
    """Resolved settings for the current ``ENVIRONMENT``."""

    environment: EnvironmentName
    frontend_url: str
    backend_url: str
    qdrant_url: str
    qdrant_api_key: str | None
    qdrant_collection: str = "cfr_chunks"
    # LLM: each entry is (litellm model id, ollama_api_base or None)
    llm_model_chain: tuple[tuple[str, str | None], ...] = field(default_factory=tuple)
    ollama_base_url: str = "http://localhost:11434"
    # Extra CORS origins (comma-separated in env) in addition to frontend_url
    cors_origins: tuple[str, ...] = ()

    @staticmethod
    def from_env() -> "Settings":
        env = _norm_env()
        if env == "local":
            return Settings._from_local()
        return Settings._from_cloud()

    @staticmethod
    def _from_local() -> "Settings":
        frontend = (os.getenv("FRONTEND_URL") or "http://localhost:3000").strip().rstrip("/")
        backend = (os.getenv("BACKEND_URL") or "http://localhost:8000").strip().rstrip("/")
        ollama_base = (os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").strip().rstrip("/")
        model = (os.getenv("OLLAMA_MODEL") or "llama3.2:3b").strip()
        if not model.startswith("ollama/"):
            model = f"ollama/{model}"
        qdrant_url = (os.getenv("QDRANT_URL") or "http://localhost:6333").strip().rstrip("/")
        qdrant_key = (os.getenv("QDRANT_API_KEY") or "").strip() or None
        qdrant_collection = (os.getenv("QDRANT_COLLECTION") or "cfr_chunks").strip()
        extras = _split_origins(os.getenv("CORS_EXTRA_ORIGINS"))
        origins = _unique_strs((frontend, *extras))
        return Settings(
            environment="local",
            frontend_url=frontend,
            backend_url=backend,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_key,
            qdrant_collection=qdrant_collection,
            llm_model_chain=((model, ollama_base),),
            ollama_base_url=ollama_base,
            cors_origins=origins,
        )

    @staticmethod
    def _from_cloud() -> "Settings":
        frontend = (os.getenv("FRONTEND_URL") or "").strip().rstrip("/")
        backend = (os.getenv("BACKEND_URL") or "").strip().rstrip("/")
        if not frontend or not backend:
            raise ValueError(
                "Cloud mode requires FRONTEND_URL and BACKEND_URL (e.g. Vercel and Render bases)."
            )
        qdrant_url = (os.getenv("QDRANT_URL") or "").strip().rstrip("/")
        if not qdrant_url:
            raise ValueError("Cloud mode requires QDRANT_URL (Qdrant Cloud cluster URL).")
        qdrant_key = (os.getenv("QDRANT_API_KEY") or "").strip() or None
        if not qdrant_key:
            raise ValueError("Cloud mode requires QDRANT_API_KEY for Qdrant Cloud.")
        qdrant_collection = (os.getenv("QDRANT_COLLECTION") or "cfr_chunks").strip()

        groq_model = (os.getenv("GROQ_MODEL") or "groq/llama-3.1-8b-instant").strip()
        if not groq_model.startswith("groq/"):
            groq_model = f"groq/{groq_model}"
        gemini_model = (os.getenv("GEMINI_MODEL") or "gemini/gemini-2.0-flash").strip()
        if not gemini_model.startswith("gemini/"):
            gemini_model = f"gemini/{gemini_model}"

        # LiteLLM reads these from the process environment for Groq / Gemini
        gk = (os.getenv("GROQ_API_KEY") or "").strip()
        gem = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
        if gk:
            os.environ.setdefault("GROQ_API_KEY", gk)
        if gem:
            os.environ.setdefault("GEMINI_API_KEY", gem)
            os.environ.setdefault("GOOGLE_API_KEY", gem)

        ollama_base = (os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").strip().rstrip("/")
        extras = _split_origins(os.getenv("CORS_EXTRA_ORIGINS"))
        origins = _unique_strs((frontend, *extras))
        return Settings(
            environment="cloud",
            frontend_url=frontend,
            backend_url=backend,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_key,
            qdrant_collection=qdrant_collection,
            llm_model_chain=((groq_model, None), (gemini_model, None)),
            ollama_base_url=ollama_base,
            cors_origins=origins,
        )


def _split_origins(raw: str | None) -> tuple[str, ...]:
    if not raw or not str(raw).strip():
        return ()
    return tuple(s.strip().rstrip("/") for s in str(raw).split(",") if s.strip())


def _unique_strs(items: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for s in items:
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return tuple(out)


@lru_cache
def get_settings() -> Settings:
    return Settings.from_env()


def reload_settings() -> Settings:
    """Clear cache (e.g. after tests). Not used in production."""
    get_settings.cache_clear()
    return get_settings()
