"""Embedding enrichment — R1 minimal text-space sidecar.

C1.1 implements build_node_text, a pure and deterministic text-family selector.
The real text family is fixed here in R1 as the source of truth; ``code`` and
``image`` are excluded (code is reserved for R10, image for every space).
"""
from __future__ import annotations

import os
from pathlib import Path

_TEXT_FILE_TYPES = frozenset({"document", "paper", "rationale", "concept"})


def build_node_text(node: dict) -> str | None:
    """Return ``"{label}\n{source_file}"`` for a text-family node, else ``None``."""
    if node.get("file_type") not in _TEXT_FILE_TYPES:
        return None
    return f"{str(node.get('label', ''))}\n{str(node.get('source_file', ''))}"


_STUB_DIM = 8


def _l2_normalize(rows: np.ndarray) -> np.ndarray:
    """L2-normalize each row so ``‖v‖ == 1.0``; zero-norm rows stay zero."""
    import numpy as np

    arr = np.asarray(rows, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return np.divide(arr, norms, out=np.zeros_like(arr), where=norms != 0)


def _call_embeddings(backend: str, model: str, inputs: list[str]) -> list[list[float]]:
    """C2 seam (stub): deterministic per-input vectors, no HTTP.

    ``backend``/``model`` are opaque — recorded later in sidecar meta.
    """
    del backend, model  # opaque in the stub; surfaced via ``text_meta``
    return [[float(i + 1)] * _STUB_DIM for i in range(len(inputs))]


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
