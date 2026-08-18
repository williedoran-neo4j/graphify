"""Semantic search over the text-space sidecar — R2 C2.2.

``search_vectors`` is the pure compute core: load the sidecar, embed the query
via the injected seam (``_stub_query_embed`` is the deterministic C4 query-embed
stub, replaced by the real backend in R3), score rows as ``text_vecs @ q``
(stored rows are L2-normalized from R1 I3, so cosine is a plain dot product),
and return ranked result rows. The absent-sidecar case returns ``None`` — the
seam the serve handler keys off.
"""
from __future__ import annotations

import json
import os
from collections import OrderedDict
from collections.abc import Callable


def _stub_query_embed(query: str, *, space: str, meta: dict) -> list[float]:
    # C4 query-embed stub: deterministic per query, satisfying q[1] > q[0] > q[2]
    # (the ranking premise pinned in tests/test_embed_search.py). The real
    # backend lands in R3; until then the query string is deliberately ignored.
    del query, space, meta
    return [1.0, 2.0, 0.5, 0, 0, 0, 0, 0]


# C3.3/RT6: bounded (model, text) -> vector cache over the injected query_embed
# seam. Keyed on CONTENTS (the sidecar's recorded model + the query string),
# never on a graph object: a query embed is a pure function of (model, text),
# so the entry survives — and stays correct across — hot-reloaded graphs, just
# as preloaded/file_type_lookup cross that boundary. The LRU bound keeps the
# cache from growing without limit.
_QUERY_EMBED_CACHE: "OrderedDict[tuple[object, str], list[float]]" = OrderedDict()
_QUERY_EMBED_MAXSIZE = 128


def _embed_query_cached(
    query_embed: Callable[..., list[float]],
    query: str,
    *,
    space: str,
    meta: dict,
) -> list[float]:
    """Query-embed invocation wrapped in a bounded ``(model, text)`` LRU (RT6).

    The first search for a distinct ``(meta["model"], query)`` pair calls the
    injected ``query_embed`` and memoizes the vector; an identical pair on a
    later ``search_vectors`` call — same recorded model, same text — is served
    from the cache without re-calling the seam. A changed text or a different
    ``meta["model"]`` re-calls.
    """
    key = (meta.get("model"), query)
    cached = _QUERY_EMBED_CACHE.get(key)
    if cached is not None:
        _QUERY_EMBED_CACHE.move_to_end(key)
        return cached
    cached = query_embed(query, space=space, meta=meta)
    _QUERY_EMBED_CACHE[key] = cached
    _QUERY_EMBED_CACHE.move_to_end(key)
    if len(_QUERY_EMBED_CACHE) > _QUERY_EMBED_MAXSIZE:
        _QUERY_EMBED_CACHE.popitem(last=False)
    return cached


def load_sidecar(path: str | os.PathLike) -> dict | None:
    """Load the embeddings sidecar ``.npz``, or ``None`` when it is absent.

    Returns the stored arrays: ``text_ids`` (unicode ids), ``text_vecs``
    (float32 rows) and ``text_meta`` (a JSON string under R1's writer).
    numpy is imported inside the function so module import never pulls it in
    (serve.py's lazy-load guard depends on that).

    Enforces invariant I5 at the single load chokepoint (RT7): when the stored
    ``text_meta["dim"]`` disagrees with ``text_vecs.shape[1]``, raises
    ``ValueError`` instead of letting a dimension mix flow onward.
    """
    import numpy as np

    npz_path = os.fspath(path)
    if not os.path.exists(npz_path):
        return None
    with np.load(npz_path, allow_pickle=False) as data:
        text_ids = data["text_ids"]
        text_vecs = data["text_vecs"]
        text_meta = data["text_meta"]
    meta = json.loads(str(text_meta))
    if int(meta["dim"]) != int(text_vecs.shape[1]):
        raise ValueError(
            f"sidecar embeddings.npz meta dim {int(meta['dim'])} "
            f"does not match stored text_vecs width {int(text_vecs.shape[1])}"
        )
    return {
        "text_ids": text_ids,
        "text_vecs": text_vecs,
        "text_meta": text_meta,
    }


def node_file_type(nid: str) -> str:
    """``search_vectors`` file-type helper seam (C2.5 will own this lookup)."""
    del nid
    return ""


def search_vectors(
    path: str | os.PathLike,
    query: str,
    *,
    space: str,
    top_k: int = 10,
    file_type: list[str] | None = None,
    min_score: float = 0.0,
    query_embed=_stub_query_embed,
    file_type_lookup: Callable[[str], str] = node_file_type,
    preloaded: dict | None = None,
) -> list[dict] | None:
    """Rank the sidecar rows against the query, score-descending.

    Returns a list of ``{"id", "score", "space"}`` rows (the handler joins them
    to the graph's nodes for label/file_type), or ``None`` when the sidecar is
    absent. ``min_score`` drops rows strictly below the threshold before the
    ``file_type`` allow-set (each row's type resolved through the injected
    ``file_type_lookup``), then results are sorted score-descending and cut
    to ``top_k``.``. ``preloaded`` supplies already-loaded sidecar arrays so a
    caller can memoize them across calls; when absent the sidecar is loaded
    from ``path`` on every call (the graph-agnostic default).
    """
    import numpy as np

    sidecar = preloaded if preloaded is not None else load_sidecar(path)
    if sidecar is None:
        return None
    text_ids = sidecar["text_ids"]
    text_vecs = sidecar["text_vecs"]
    try:
        meta = json.loads(str(sidecar["text_meta"]))
    except (ValueError, TypeError):
        meta = {}
    q = np.asarray(
        _embed_query_cached(query_embed, query, space=space, meta=meta),
        dtype=np.float32,
    )
    scores = text_vecs @ q
    allow = set(file_type) if file_type else None
    rows = (
        {"id": str(nid), "score": float(score), "space": space}
        for nid, score in zip(text_ids, scores, strict=False)
        if score >= min_score
        if allow is None or file_type_lookup(nid) in allow
    )
    ranked = sorted(rows, key=lambda r: r["score"], reverse=True)
    return ranked[:top_k]
