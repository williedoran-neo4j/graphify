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
from concurrent.futures import ThreadPoolExecutor

from graphify.embed import _call_embeddings, _l2_normalize


def _stub_query_embed(query: str, *, space: str, meta: dict) -> list[float]:
    # C4 query-embed stub: deterministic per query, satisfying q[1] > q[0] > q[2]
    # (the ranking premise pinned in tests/test_embed_search.py). The real
    # backend lands in R3; until then the query string is deliberately ignored.
    del query, space, meta
    return [1.0, 2.0, 0.5, 0, 0, 0, 0, 0]


def _real_query_embed(query: str, *, space: str, meta: dict) -> list[float]:
    # Resolve backend/model from meta, falling back to space defaults
    backend = meta.get("backend")
    model = meta.get("model")
    if backend is None or model is None:
        backend = "ollama"
        model = "nomic-embed-text" if space == "text" else "nomic-embed-code"
    vectors = _call_embeddings(backend, model, [query])
    vec = vectors[0]
    import numpy as np
    normalized = _l2_normalize(np.array([vec], dtype=np.float32))[0]
    return normalized.tolist()


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
    (float32 rows) and ``text_meta`` (a JSON string under the writer). A group
    absent from the archive (the writer omits an empty group) returns ``None``
    for its keys, so a pre-code text-only sidecar loads unchanged and a
    code-only sidecar loads with ``text_*`` == ``None``. numpy is imported
    inside the function so module import never pulls it in (serve.py's lazy-load
    guard depends on that).

    Enforces the dim-consistency invariant at the single load chokepoint
    (RT7): when a present group's ``meta["dim"]`` disagrees with its
    ``vecs.shape[1]``, raises ``ValueError`` instead of letting a dimension
    mix flow onward.
    """
    import numpy as np

    npz_path = os.fspath(path)
    if not os.path.exists(npz_path):
        return None
    with np.load(npz_path, allow_pickle=False) as data:
        text_ids = data["text_ids"] if "text_ids" in data.files else None
        text_vecs = data["text_vecs"] if "text_vecs" in data.files else None
        text_meta = data["text_meta"] if "text_meta" in data.files else None
        code_ids = data["code_ids"] if "code_ids" in data.files else None
        code_vecs = data["code_vecs"] if "code_vecs" in data.files else None
        code_meta = data["code_meta"] if "code_meta" in data.files else None
    for name, meta, vecs in (
        ("text", text_meta, text_vecs),
        ("code", code_meta, code_vecs),
    ):
        if meta is not None and vecs is not None:
            meta_dict = json.loads(str(meta))
            if int(meta_dict["dim"]) != int(vecs.shape[1]):
                raise ValueError(
                    f"sidecar embeddings.npz meta dim {int(meta_dict['dim'])} "
                    f"does not match stored {name}_vecs width "
                    f"{int(vecs.shape[1])}"
                )
    return {
        "text_ids": text_ids,
        "text_vecs": text_vecs,
        "text_meta": text_meta,
        "code_ids": code_ids,
        "code_vecs": code_vecs,
        "code_meta": code_meta,
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
    absent. Each present group (``text`` and ``code``) is scored against a query
    embed from its own space's model, ranked score-descending, and the groups'
    rows then merge into ONE global score-descending list. ``min_score`` drops
    rows strictly below the threshold and ``top_k`` cuts the count — both apply
    across the merged list, after the per-space rank (``file_type`` resolves
    each row's type through the injected ``file_type_lookup``). ``preloaded``
    supplies already-loaded sidecar arrays so a caller can memoize them across
    calls; when absent the sidecar is loaded from ``path`` on every call (the
    graph-agnostic default).
    """
    import numpy as np

    sidecar = preloaded if preloaded is not None else load_sidecar(path)
    if sidecar is None:
        return None
    allow = set(file_type) if file_type else None
    groups = (
        ("text", sidecar["text_ids"], sidecar["text_vecs"], sidecar["text_meta"]),
        ("code", sidecar["code_ids"], sidecar["code_vecs"], sidecar["code_meta"]),
    )
    present = [
        g for g in groups if g[1] is not None and g[2] is not None
    ]

    def score_group(group: str, group_ids, group_vecs, raw_meta) -> list[dict]:
        """Embed the query for one space and rank that space's rows."""
        try:
            meta = json.loads(str(raw_meta))
        except (ValueError, TypeError):
            meta = {}
        q = np.asarray(
            _embed_query_cached(query_embed, query, space=group, meta=meta),
            dtype=np.float32,
        )
        scores = group_vecs @ q
        rows = (
            {"id": str(nid), "score": float(score), "space": group}
            for nid, score in zip(group_ids, scores, strict=False)
            if score >= min_score
            if allow is None or file_type_lookup(nid) in allow
        )
        return sorted(rows, key=lambda r: r["score"], reverse=True)

    if len(present) == 1:
        per_space = [score_group(*present[0])]
    else:
        with ThreadPoolExecutor(max_workers=2) as pool:
            per_space = list(pool.map(score_group, *zip(*present)))
    merged = sorted(
        (r for rows in per_space for r in rows),
        key=lambda r: r["score"],
        reverse=True,
    )
    return merged[:top_k]
