"""R3 C3.1 — embedding backend registry, live base-URL routing, and
excluded-backend errors (RT1, RT2, RT4).

RT1: ``EMBEDDING_BACKENDS`` exposes exactly ``{"ollama", "openai"}`` (C5).
RT2: ``_call_embeddings`` fails fast for excluded backends
(``claude-cli``/``bedrock``/``claude``) with a ``ValueError`` naming the
supported set — no route through, no fall-through 404.
RT4: ``resolve_embedding_backend("ollama")`` resolves the configured local
gateway when ``OPENAI_BASE_URL`` is set (the Bifrost route the stack uses),
else the direct local Ollama ``http://localhost:11434/v1`` — never a paid
``https://api.*`` cloud endpoint in either arm, resolved LIVE at call time.
"""
from __future__ import annotations

import pytest

from graphify.embed import _call_embeddings
from graphify.embeddings import EMBEDDING_BACKENDS, resolve_embedding_backend


def test_embedding_backends_exposes_exactly_ollama_and_openai():
    """RT1 — the embedding registry is exactly ``{"ollama", "openai"}``,
    independent of the chat ``BACKENDS`` shape (C5)."""
    assert set(EMBEDDING_BACKENDS) == {"ollama", "openai"}


def test_excluded_backends_raise_valueerror_naming_supported_set():
    """RT2 — ``claude-cli``/``bedrock``/``claude`` are excluded from the
    embedding seam and must fail fast with a ``ValueError`` naming the
    supported set (both discriminators appear: the excluded backend's
    "claude" and the supported "openai"), never touching the network."""
    for backend in ("claude-cli", "bedrock", "claude"):
        with pytest.raises(ValueError) as excinfo:
            _call_embeddings(backend, "nomic-embed-text", ["some text"])
        message = str(excinfo.value)
        assert "claude" in message
        assert "openai" in message


def test_ollama_resolves_to_openai_base_url_gateway_when_set(monkeypatch):
    """RT4 arm 1 — with ``OPENAI_BASE_URL`` set, ``ollama`` resolves to it
    verbatim (the Bifrost gateway route the stack already uses)."""
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:9765/openai/v1")
    assert resolve_embedding_backend("ollama") == "http://localhost:9765/openai/v1"


def test_ollama_resolves_local_when_no_gateway_env_set(monkeypatch):
    """RT4 arm 2 — with ``OPENAI_BASE_URL``, ``OLLAMA_BASE_URL`` and
    ``OLLAMA_HOST`` all unset, ``ollama`` resolves to the direct local
    ``http://localhost:11434/v1`` (never a cloud endpoint)."""
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert resolve_embedding_backend("ollama") == "http://localhost:11434/v1"
