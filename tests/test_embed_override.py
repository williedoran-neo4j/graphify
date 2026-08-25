"""Per-space embed backend/model env override (C1+C2).

A new ``_embed_space_backend_model`` helper resolves ``(backend, model)`` per
space from ``GRAPHIFY_EMBED_*_BACKEND`` / ``GRAPHIFY_EMBED_*_MODEL`` env vars,
fallbacking to the existing hardcoded defaults. ``_embed_group`` uses the helper
so the chosen values reach BOTH the sidecar ``meta`` and the ``_call_embeddings``
seam.
"""
from __future__ import annotations

import json

import networkx as nx
import numpy as np
import pytest

from graphify.embed import _STUB_DIM, enrich_embeddings


def _mixed_graph() -> nx.Graph:
    """One code node and one text node so both spaces are exercised."""
    g = nx.Graph()
    g.add_nodes_from(
        [
            (
                "n-code-01",
                {
                    "id": "n-code-01",
                    "label": "Parser",
                    "source_file": "src/parser.py",
                    "file_type": "code",
                },
            ),
            (
                "n-text-01",
                {
                    "id": "n-text-01",
                    "label": "API Tokens",
                    "source_file": "docs/security.md",
                    "file_type": "document",
                },
            ),
        ]
    )
    return g


# Default arms: no env vars set -> both spaces use ollama defaults.


def test_embed_override_defaults_preserved(tmp_path, monkeypatch):
    """Without env vars the text and code spaces still use ollama defaults."""
    for v in (
        "GRAPHIFY_EMBED_TEXT_BACKEND",
        "GRAPHIFY_EMBED_TEXT_MODEL",
        "GRAPHIFY_EMBED_CODE_BACKEND",
        "GRAPHIFY_EMBED_CODE_MODEL",
    ):
        monkeypatch.delenv(v, raising=False)

    calls: list[tuple[str, str, list[str]]] = []

    def recording_seam(backend, model, inputs):
        calls.append((backend, model, list(inputs)))
        return [[1.0] * _STUB_DIM for _ in inputs]

    monkeypatch.setattr("graphify.embed._call_embeddings", recording_seam)

    out = enrich_embeddings(_mixed_graph(), tmp_path / "graph.json")

    with np.load(out) as data:
        text_meta = json.loads(str(data["text_meta"]))
        code_meta = json.loads(str(data["code_meta"]))

    assert text_meta["backend"] == "ollama"
    assert text_meta["model"] == "nomic-embed-text"
    assert code_meta["backend"] == "ollama"
    assert code_meta["model"] == "nomic-embed-code"

    text_calls = [c for c in calls if c[1] == "nomic-embed-text"]
    code_calls = [c for c in calls if c[1] == "nomic-embed-code"]
    assert len(text_calls) == 1
    assert len(code_calls) == 1
    assert text_calls[0][0] == "ollama"
    assert code_calls[0][0] == "ollama"


# Override arm: text space is overridden, code space stays default.


def test_embed_override_text_backend_model(tmp_path, monkeypatch):
    """With ``GRAPHIFY_EMBED_TEXT_BACKEND=openai`` and
    ``GRAPHIFY_EMBED_TEXT_MODEL=text-embedding-3-small`` the text space uses the
    override values while the code space retains its default."""
    monkeypatch.setenv("GRAPHIFY_EMBED_TEXT_BACKEND", "openai")
    monkeypatch.setenv("GRAPHIFY_EMBED_TEXT_MODEL", "text-embedding-3-small")
    monkeypatch.delenv("GRAPHIFY_EMBED_CODE_BACKEND", raising=False)
    monkeypatch.delenv("GRAPHIFY_EMBED_CODE_MODEL", raising=False)

    calls: list[tuple[str, str, list[str]]] = []

    def recording_seam(backend, model, inputs):
        calls.append((backend, model, list(inputs)))
        return [[1.0] * _STUB_DIM for _ in inputs]

    monkeypatch.setattr("graphify.embed._call_embeddings", recording_seam)

    out = enrich_embeddings(_mixed_graph(), tmp_path / "graph.json")

    with np.load(out) as data:
        text_meta = json.loads(str(data["text_meta"]))
        code_meta = json.loads(str(data["code_meta"]))

    assert text_meta["backend"] == "openai"
    assert text_meta["model"] == "text-embedding-3-small"
    assert code_meta["backend"] == "ollama"
    assert code_meta["model"] == "nomic-embed-code"

    text_calls = [c for c in calls if c[1] == "text-embedding-3-small"]
    code_calls = [c for c in calls if c[1] == "nomic-embed-code"]
    assert len(text_calls) == 1
    assert len(code_calls) == 1
    assert text_calls[0][0] == "openai"
    assert code_calls[0][0] == "ollama"
