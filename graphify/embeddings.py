"""Embedding backend registry + live base-URL routing (R3, C3.1).

RT1/RT4: ``EMBEDDING_BACKENDS`` is independent from the chat ``BACKENDS``
registry (C5), and base URLs resolve LIVE at call time via
``resolve_embedding_backend`` — never captured at import, so a gateway env var
change is honored by the running stack.
"""
from __future__ import annotations

import os

from graphify.llm import _resolve_ollama_base_url, _validate_ollama_base_url


def _ollama_base_url() -> str:
    """Route ``ollama`` through the configured gateway (``OPENAI_BASE_URL``,
    the stack's Bifrost/llama.cpp/LM Studio route) when one is set — RT4 arm 1.
    Otherwise resolve the direct local Ollama ``/v1`` endpoint (arm 2) and
    hard-block link-local/metadata hosts (F3). Never a paid cloud default."""
    gateway = os.environ.get("OPENAI_BASE_URL")
    if gateway:
        return gateway
    url = _resolve_ollama_base_url("http://localhost:11434/v1")
    _validate_ollama_base_url(url)
    return url


def _openai_base_url() -> str:
    """``openai`` follows the same gateway env var, else the OpenAI cloud."""
    return os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"


# Independent from the chat ``BACKENDS`` registry — embedding backends only (C5).
# ``base_url`` is a call-time function of env so resolution is LIVE (RT4).
EMBEDDING_BACKENDS: dict[str, dict] = {
    "ollama": {
        "default_model": "nomic-embed-text",
        "base_url": _ollama_base_url,
        "env_key": "OLLAMA_API_KEY",
    },
    "openai": {
        "default_model": "text-embedding-3-small",
        "base_url": _openai_base_url,
        "env_key": "OPENAI_API_KEY",
    },
}


def resolve_embedding_backend(backend: str) -> str:
    """Resolve a backend's base URL live at call time (RT4)."""
    return EMBEDDING_BACKENDS[backend]["base_url"]()
