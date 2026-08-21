"""Embedding enrichment — text-space sidecar.

C2 (R5) composes ``build_node_text(graph, nid, node)`` into a deterministic
4-line text — ``{label}``, ``{source_file}`` optionally qualified with
``:{source_location}``, the ``{rationale}`` when present (else an empty line),
and a space-joined line of up to 10 neighbour labels sorted by
``(-degree, id)`` — while ``image`` (and every other non-text-family type) is
excluded from every space.

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

_TEXT_FILE_TYPES = frozenset({"document", "paper", "rationale", "concept", "code"})

# file_type → embedding space: the table the code/text vector index DDL and
# the per-space node props / `:Embedded` label branch on.
_EMBED_SPACE_BY_FILE_TYPE: dict = {
    "code": "code",
    "document": "text",
    "paper": "text",
    "rationale": "text",
    "concept": "text",
}


def _embed_space(file_type: str | None) -> str | None:
    """Return the embedding space ("code"/"text") for a file_type, or None."""
    return _EMBED_SPACE_BY_FILE_TYPE.get(file_type)


def _neighbour_text(graph, nid: str, limit: int = 10) -> str:
    """Return the space-joined neighbour labels of ``nid``, highest-degree first.

    Sorted by ``(-graph.degree[nb], nb)`` and truncated to ``limit`` labels;
    each label is the neighbour node's own ``label`` attribute (not its id).
    The line is emitted even when there are no neighbours (an empty string).
    """
    neighbours = sorted(
        graph.neighbors(nid),
        key=lambda nb: (-graph.degree[nb], nb),
    )[:limit]
    return " ".join(graph.nodes[nb].get("label", "") for nb in neighbours)

# RT14 (C5.3) — cap on the FULL constructed build_node_text string. Only the
# neighbour block shrinks (trailing whole labels dropped, character-trim as a
# last resort); the label/path/rationale lines stay byte-intact.
_NODE_TEXT_CAP_CHARS = 2000


def _cap_neighbour_line(line: str, cap_chars: int) -> str:
    """Cap the neighbour line to ``cap_chars`` characters.

    Drops TRAILING whole labels while the space-joined remainder exceeds the
    cap; if a single remaining label alone is still over cap, character-trims
    that label's tail (whole-label drop first, char-trim last resort).
    """
    if len(line) <= cap_chars:
        return line
    tokens = line.split(" ")
    while len(tokens) > 1:
        kept = tokens[:-1]
        if len(" ".join(kept)) <= cap_chars:
            return " ".join(kept)
        tokens = kept
    return tokens[0][:cap_chars]

# R4/C4.1 write-cache counters: corrupt entries are counted misses, hits are
# counted hits (incremented by load_embedding only).
_embed_cache_hits = 0
_embed_cache_corrupt = 0


def _read_code_source(source_path: Path) -> str | None:
    """Read a code source file as text, or ``None`` if it cannot be read.

    Missing, unreadable, and undecodable files all return ``None`` (never
    raise): the caller falls through to the attribute-only text.
    """
    try:
        return source_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def build_node_text(graph, nid: str, node: dict, root: str | os.PathLike | None = None) -> str | None:
    """Compose a text-family node's deterministic embedding text.

    ``{label}`` / ``{source_file}[:{source_location}]`` / ``{rationale}`` (only
    when present) / the neighbour line from ``_neighbour_text`` — joined with
    ``"\\n"``. Returns ``None`` for any non-text-family type.

    A ``code`` node's text additionally appends the raw contents of
    ``source_file`` (resolved against ``root``) after the skeleton, with the
    whole string truncated to ``_NODE_TEXT_CAP_CHARS``; an unreadable source
    file falls through to the attribute-only text.
    """
    if node.get("file_type") not in _TEXT_FILE_TYPES:
        return None
    path = str(node.get("source_file", ""))
    loc = node.get("source_location")
    if loc:
        path = f"{path}:{str(loc)}"
    lines = [str(node.get("label", "")), path]
    rationale = node.get("rationale")
    lines.append(str(rationale) if rationale else "")
    neighbours = _neighbour_text(graph, nid, limit=10)
    budget = max(0, _NODE_TEXT_CAP_CHARS - sum(len(line) for line in lines) - len(lines))
    lines.append(_cap_neighbour_line(neighbours, budget))
    text = "\n".join(lines)
    if node.get("file_type") == "code" and root is not None:
        source_name = node.get("source_file")
        if source_name:
            source_path = Path(source_name) if Path(source_name).is_absolute() else Path(root) / str(source_name)
            source_text = _read_code_source(source_path)
        else:
            source_text = None
        if source_text is not None:
            text = f"{text}\n{source_text}"
            if len(text) > _NODE_TEXT_CAP_CHARS:
                text = text[:_NODE_TEXT_CAP_CHARS]
    return text


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


def _embed_group(
    graph, cache_root: Path, space: str, root: Path | None
) -> tuple[list[str], np.ndarray, dict]:
    """Embed one space's nodes and return ``(ids, vecs, meta)`` for the sidecar.

    Nodes whose space maps to ``space`` (``_embed_space``) are sorted by id and
    their C1-constructed texts are embedded through ``_call_embeddings`` under
    the space's own ``(backend, model)`` namespace, serving hits from the
    write-cache first. An empty group yields ``([], empty_vecs, {})`` and never
    touches the seam.
    """
    import numpy as np

    backend, model = ("ollama", "nomic-embed-code") if space == "code" else (
        "ollama",
        "nomic-embed-text",
    )

    texts: list[str] = []
    ids: list[str] = []
    for nid in sorted(graph.nodes):
        attrs = graph.nodes[nid]
        if _embed_space(attrs.get("file_type")) != space:
            continue
        text = build_node_text(graph, str(nid), attrs, root=root)
        if text is not None:
            ids.append(str(nid))
            texts.append(text)

    if not texts:
        return [], np.array([], dtype=np.float32), {}

    cached: list[list[float] | None] = []
    missing_texts: list[str] = []
    for text in texts:
        vec = load_embedding(backend, model, text, cache_root)
        cached.append(vec)
        if vec is None:
            missing_texts.append(text)

    # R4/C4.2 — only MISSING texts reach the seam; an all-hit set calls nothing.
    if missing_texts:
        rebuilt = _call_embeddings(backend=backend, model=model, inputs=missing_texts)
    else:
        rebuilt = []

    vec_rows: list[list[float]] = []
    rebuilt_iter = iter(rebuilt)
    for vec in cached:
        if vec is not None:
            vec_rows.append(vec)
        else:
            vec_rows.append(next(rebuilt_iter))
    _guard_dim_consistency(vec_rows)
    for text, vec in zip(missing_texts, rebuilt):
        save_embedding(backend, model, text, vec, cache_root)

    vecs = np.asarray(vec_rows, dtype=np.float32)
    return ids, vecs, {
        "model": model,
        "backend": backend,
        "dim": vecs.shape[1],
    }


def enrich_embeddings(graph, graph_path: str | os.PathLike) -> os.PathLike:
    """Write the ``embeddings.npz`` sidecar next to ``graph_path``.

    Nodes are partitioned by their embedding space (``_embed_space``): the
    ``text_*`` group (``text_ids`` / ``text_vecs`` float32 L2-normalized /
    ``text_meta`` JSON) holds every ``"text"``-space node, and the ``code_*``
    group (``code_ids`` / ``code_vecs`` / ``code_meta``) holds every
    ``"code"``-space node under its own ``(ollama, nomic-embed-code)`` model.
    Nodes whose file_type maps to no space are written to neither group. An
    empty group is omitted entirely. Returns the written sidecar path.
    """
    import json
    from datetime import datetime, timezone
    from importlib.metadata import version as _pkg_version

    import numpy as np

    cache_root = Path(graph_path).parent
    try:
        graphify_version = _pkg_version("graphifyy")
    except Exception:
        graphify_version = "unknown"

    groups = {}
    for space in ("text", "code"):
        ids, vecs, meta = _embed_group(graph, cache_root, space, root=cache_root)
        if not ids:
            continue
        meta["graphify_version"] = graphify_version
        meta["created_at"] = datetime.now(timezone.utc).isoformat()
        groups[space] = (ids, vecs, meta)

    npz_path = Path(graph_path).parent / "embeddings.npz"
    arrays: dict[str, object] = {}
    for space, (ids, vecs, meta) in groups.items():
        arrays[f"{space}_ids"] = np.array(ids, dtype=str)
        arrays[f"{space}_vecs"] = _l2_normalize(vecs)
        arrays[f"{space}_meta"] = json.dumps(meta)
    np.savez(npz_path, **arrays)
    return npz_path


# ---- R4/C4.1: two-namespace (backend, model) embedding write-cache ----
#
# Pinned by tests/test_embed_cache.py (RT9 namespace independence, RT11
# corrupt-entry counted miss). Cache dir layout reuses cache.cache_dir's kind
# namespacing; entries are ``{sha256(text)}.npy`` files holding a pickled raw
# float vector. Write discipline mirrors cache.py's save_cached (mkstemp +
# os.replace); corrupt entries mirror load_cached's counted-miss handling.

def _embed_texts_key(backend: str, model: str) -> str:
    """Return the cache ``kind``: ``f"embed-{backend}-{model}"`` (C4.1).
    Single source of truth for BOTH the namespace separation and the prune glob."""
    return f"embed-{backend}-{model}"


def _embedding_cache_dir(root: Path, backend: str, model: str) -> Path:
    """C4.1 — thin wrapper over ``cache.cache_dir`` for the embed kind."""
    from graphify.cache import cache_dir

    return cache_dir(root, kind=_embed_texts_key(backend, model))


def load_embedding(
    backend: str, model: str, text: str, root: Path
) -> list[float] | None:
    """C4.1 — load a cached embedding; miss/corrupt -> None (corrupt counted).

    The entry is ``cache_dir(root, kind=embed-{backend}-{model}) /
    {sha256(text)}.npy``. A missing entry is a plain miss; a present-but-
    unparseable entry is a counted miss (``_embed_cache_corrupt``) that does
    not raise and is left in place (mirrors cache.py's JSONDecodeError
    discipline, #2405). A hit increments ``_embed_cache_hits``.
    """
    global _embed_cache_hits, _embed_cache_corrupt
    import hashlib
    import pickle

    entry = _embedding_cache_dir(root, backend, model) / (
        f"{hashlib.sha256(text.encode()).hexdigest()}.npy"
    )
    if not entry.exists():
        return None
    try:
        vec = pickle.loads(entry.read_bytes())
    except (pickle.UnpicklingError, EOFError, OSError):
        _embed_cache_corrupt += 1
        return None
    if not isinstance(vec, (list, tuple)) or not all(
        isinstance(x, (int, float)) and not isinstance(x, bool) for x in vec
    ):
        # Payload unpickled but is not a flat sequence of numbers: not a
        # usable vector, count it as corrupt.
        _embed_cache_corrupt += 1
        return None
    _embed_cache_hits += 1
    return list(vec)


def save_embedding(
    backend: str, model: str, text: str, vec: list[float], root: Path
) -> None:
    """C4.1 — atomically save an embedding (mkstemp + os.replace, pickle).

    Stored bytes are the pickled RAW float vector (not numpy), so the entry
    keyed by ``sha256(constructed text)`` stays valid regardless of a future
    normalization or dtype change."""
    import hashlib
    import pickle
    import tempfile

    target_dir = _embedding_cache_dir(root, backend, model)
    entry = target_dir / f"{hashlib.sha256(text.encode()).hexdigest()}.npy"
    fd, tmp_path = tempfile.mkstemp(dir=target_dir, prefix=entry.name, suffix=".tmp")
    try:
        os.write(fd, pickle.dumps(vec))
        os.close(fd)
        try:
            os.replace(tmp_path, entry)
        except PermissionError:
            # Windows: os.replace can fail with WinError 5 if the target is
            # briefly locked. Fall back to copy-then-delete.
            import shutil

            shutil.copy2(tmp_path, entry)
            os.unlink(tmp_path)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def prune_embedding_cache(
    root: Path, backend: str, model: str, live_texts: set[str]
) -> int:
    """R4/C4.2 — remove orphaned embed-cache entries, returning the count pruned.

    Mirror of ``prune_semantic_cache`` (cache.py), applied to the
    ``embed-{backend}-{model}`` namespace: base-anchored at ``_GRAPHIFY_OUT``,
    glob ``**/*.npy``, delete any entry whose stem (the ``sha256(text)`` key)
    is not in ``{sha256(t) for t in live_texts}``. ``*.tmp`` atomic-write
    temporaries are skipped. Best-effort per unlink (``try/except OSError``).
    NOT wired into ``enrich_embeddings`` — the standalone re-embed CLI owns the
    call site."""
    import hashlib

    from graphify.cache import _GRAPHIFY_OUT

    _out = Path(_GRAPHIFY_OUT)
    base = _out if _out.is_absolute() else Path(root).resolve() / _out
    live_hashes = {hashlib.sha256(t.encode()).hexdigest() for t in live_texts}
    cache_dir = base / "cache" / _embed_texts_key(backend, model)
    if not cache_dir.is_dir():
        return 0
    pruned = 0
    for entry in cache_dir.glob("**/*.npy"):
        if entry.stem in live_hashes:
            continue
        try:
            entry.unlink()
            pruned += 1
        except OSError:
            pass
    return pruned
