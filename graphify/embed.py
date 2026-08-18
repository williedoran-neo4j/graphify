"""Embedding enrichment — R1 minimal text-space sidecar.

C1.1 implements build_node_text, a pure and deterministic text-family selector.
The real text family is fixed here in R1 as the source of truth; ``code`` and
``image`` are excluded (code is reserved for R10, image for every space).

R3 (C3.2): the ``_call_embeddings`` seam is real — it calls the backend's
OpenAI-compatible embeddings API in batches of ``_EMBED_BATCH_SIZE`` (100)
inputs per request (I11).
"""
from __future__ import annotations

import os
from pathlib import Path

from graphify.embeddings import EMBEDDING_BACKENDS, resolve_embedding_backend
from graphify.llm import (
    _backend_pkg_hint,
    _get_backend_api_key,
    _resolve_api_timeout,
    _resolve_max_retries,
)

_TEXT_FILE_TYPES = frozenset({"document", "paper", "rationale", "concept"})


def build_node_text(node: dict) -> str | None:
    """Return ``"{label}\n{source_file}"`` for a text-family node, else ``None``."""
    if node.get("file_type") not in _TEXT_FILE_TYPES:
        return None
    return f"{str(node.get('label', ''))}\n{str(node.get('source_file', ''))}"


_STUB_DIM = 8

# Embedding batch size (I11): inputs are chunked into requests of this many
# texts each, so a huge graph never rides a single oversized embeddings call.
_EMBED_BATCH_SIZE = 100


def _l2_normalize(rows: np.ndarray) -> np.ndarray:
    """L2-normalize each row so ``‖v‖ == 1.0``; zero-norm rows stay zero."""
    import numpy as np

    arr = np.asarray(rows, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return np.divide(arr, norms, out=np.zeros_like(arr), where=norms != 0)


def _call_embeddings(backend: str, model: str, inputs: list[str]) -> list[list[float]]:
    """Embed ``inputs`` through the backend's OpenAI-compatible API (R3/C3.2).

    Non-supported backends fail fast naming the supported set (RT2); the
    supported backends resolve their base URL live and call the real endpoint
    (``openai.OpenAI(...).embeddings.create``). Inputs are chunked into
    ``_EMBED_BATCH_SIZE``-sized requests; the returned vectors are concatenated
    in input order, one vector per input. If a response ever mixes vector
    dimensions, ``ValueError`` is raised (I5) — a dimension mixture never flows
    onward to the sidecar writer.
    """
    if backend not in EMBEDDING_BACKENDS:
        raise ValueError(
            f"unsupported embedding backend {backend!r}; "
            f"supported: {sorted(EMBEDDING_BACKENDS)}"
            " (chat-only backends like claude do not serve embeddings)"
        )
    base_url = resolve_embedding_backend(backend)
    key = _get_backend_api_key(backend) or ("ollama" if backend == "ollama" else "")
    try:
        from openai import OpenAI
    except ImportError as exc:
        extra = backend if backend in ("kimi", "gemini", "openai", "ollama") else "openai"
        raise ImportError(_backend_pkg_hint("openai", extra)) from exc

    # Same 0-retry-ollama rule as the chat path (llm.py:1198-1200): a local
    # server does not rate-limit, and 6 SDK retries would turn a hung request
    # into a multi-minute block (#1686). An explicit GRAPHIFY_MAX_RETRIES wins.
    _retries = _resolve_max_retries()
    if backend == "ollama" and not os.environ.get("GRAPHIFY_MAX_RETRIES", "").strip():
        _retries = 0
    client = OpenAI(
        api_key=key,
        base_url=base_url,
        timeout=_resolve_api_timeout(),
        max_retries=_retries,
    )

    vectors: list[list[float]] = []
    for start in range(0, len(inputs), _EMBED_BATCH_SIZE):
        batch = inputs[start : start + _EMBED_BATCH_SIZE]
        resp = client.embeddings.create(model=model, input=batch)
        data = resp.model_dump()["data"] if hasattr(resp, "model_dump") else resp.data
        vectors.extend(item["embedding"] for item in data)
    _guard_dim_consistency(vectors)
    return vectors


def _guard_dim_consistency(vectors: list[list[float]]) -> None:
    """I5 (write-side): never pass a dimension mix onward — a response whose
    vectors do not all share the first vector's length is a corrupt backend."""
    if not vectors:
        return
    dim = len(vectors[0])
    for vec in vectors:
        if len(vec) != dim:
            raise ValueError(
                "embedding backend returned inconsistent dims; "
                f"expected {dim}, got {len(vec)}"
            )


def enrich_embeddings(graph, graph_path: str | os.PathLike) -> os.PathLike:
    """Write the text-space ``embeddings.npz`` sidecar next to ``graph_path``.

    Text-family nodes (sorted by id) are embedded via ``_call_embeddings`` and
    written with ``numpy.savez`` as ``text_ids`` / ``text_vecs`` (float32) /
    ``text_meta`` (JSON string). Returns the written sidecar path.
    """
    import json
    from datetime import datetime, timezone
    from importlib.metadata import version as _pkg_version

    import numpy as np

    texts: list[str] = []
    ids: list[str] = []
    for nid in sorted(graph.nodes):
        attrs = graph.nodes[nid]
        text = build_node_text(attrs)
        if text is not None:
            ids.append(str(nid))
            texts.append(text)

    try:
        graphify_version = _pkg_version("graphifyy")
    except Exception:
        graphify_version = "unknown"

    vecs = np.asarray(
        _call_embeddings(backend="ollama", model="nomic-embed-text", inputs=texts),
        dtype=np.float32,
    )
    meta = {
        "model": "nomic-embed-text",
        "backend": "ollama",
        "dim": vecs.shape[1],
        "graphify_version": graphify_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    npz_path = Path(graph_path).parent / "embeddings.npz"
    np.savez(
        npz_path,
        text_ids=np.array(ids, dtype=str),
        text_vecs=_l2_normalize(vecs),
        text_meta=json.dumps(meta),
    )
    return npz_path
