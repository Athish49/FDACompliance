"""
Application configuration — presence-based, no ENVIRONMENT switch.

Decision logic (driven entirely by which values are set/uncommented in .env):

  URLs
    FRONTEND_URL / BACKEND_URL  — used as-is; localhost → local dev,
                                  Vercel/Render URL → production
  Qdrant
    Whichever QDRANT_URL is uncommented is used.
    QDRANT_API_KEY is optional (required only by Qdrant Cloud).

  LLM chain (tried left-to-right by LiteLLM)
    GROQ_API_KEY present   → groq/<GROQ_MODEL> added first
    GEMINI_API_KEY present → gemini/<GEMINI_MODEL> added next
    No cloud keys present  → Ollama only (OLLAMA_MODEL @ OLLAMA_BASE_URL)
    Ollama is always added as final fallback when at least one cloud key exists.

The ENVIRONMENT variable is accepted but ignored — kept only for backwards
compatibility with existing .env files.
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


def _is_local_url(url: str) -> bool:
    return "localhost" in url or "127.0.0.1" in url or "0.0.0.0" in url


@dataclass(frozen=True)
class Settings:
    """Resolved settings derived purely from which .env values are present."""

    environment: EnvironmentName        # informational only (not used for branching)
    frontend_url: str
    backend_url: str
    qdrant_url: str
    qdrant_api_key: str | None
    qdrant_collection: str = "FDAComplianceAI"
    # Each entry: (litellm model id, ollama_api_base or None)
    llm_model_chain: tuple[tuple[str, str | None], ...] = field(default_factory=tuple)
    ollama_base_url: str = "http://localhost:11434"
    cors_origins: tuple[str, ...] = ()

    @staticmethod
    def from_env() -> "Settings":
        # ── URLs ─────────────────────────────────────────────────────────────
        frontend = (os.getenv("FRONTEND_URL") or "http://localhost:3000").strip().rstrip("/")
        backend  = (os.getenv("BACKEND_URL")  or "http://localhost:8000").strip().rstrip("/")

        # ── Qdrant ───────────────────────────────────────────────────────────
        # Whichever QDRANT_URL is uncommented wins; fallback to local.
        qdrant_url = (os.getenv("QDRANT_URL") or "http://localhost:6333").strip().rstrip("/")
        qdrant_key = (os.getenv("QDRANT_API_KEY") or "").strip() or None
        qdrant_collection = (os.getenv("QDRANT_COLLECTION") or "FDAComplianceAI").strip()

        # ── Ollama (always configured; used as fallback) ──────────────────────
        ollama_base = (os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").strip().rstrip("/")
        ollama_raw  = (os.getenv("OLLAMA_MODEL") or "llama3.2:3b").strip()
        ollama_model = ollama_raw if ollama_raw.startswith("ollama/") else f"ollama/{ollama_raw}"

        # ── LLM chain: add cloud providers whose keys are present ─────────────
        chain: list[tuple[str, str | None]] = []

        groq_key = (os.getenv("GROQ_API_KEY") or "").strip()
        if groq_key:
            os.environ["GROQ_API_KEY"] = groq_key
            groq_raw = (os.getenv("GROQ_MODEL") or "llama-3.1-8b-instant").strip()
            groq_model = groq_raw if groq_raw.startswith("groq/") else f"groq/{groq_raw}"
            chain.append((groq_model, None))

        gemini_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
        if gemini_key:
            os.environ["GEMINI_API_KEY"] = gemini_key
            os.environ["GOOGLE_API_KEY"] = gemini_key
            gemini_raw = (os.getenv("GEMINI_MODEL") or "gemini-2.0-flash").strip()
            gemini_model = gemini_raw if gemini_raw.startswith("gemini/") else f"gemini/{gemini_raw}"
            chain.append((gemini_model, None))

        # Ollama: sole provider when no cloud keys, otherwise final fallback
        chain.append((ollama_model, ollama_base))

        # ── CORS ─────────────────────────────────────────────────────────────
        extras  = _split_origins(os.getenv("CORS_EXTRA_ORIGINS"))
        origins = _unique_strs((frontend, *extras))

        # Derive informational label (not used for any branching)
        env_label: EnvironmentName = "local" if _is_local_url(frontend) else "cloud"

        return Settings(
            environment=env_label,
            frontend_url=frontend,
            backend_url=backend,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_key,
            qdrant_collection=qdrant_collection,
            llm_model_chain=tuple(chain),
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
    """Clear cache (e.g. after tests)."""
    get_settings.cache_clear()
    return get_settings()
