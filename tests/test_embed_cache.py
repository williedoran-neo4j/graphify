"""R4/C4.1 — two-namespace ``(backend, model)`` embedding write-cache (RT9/RT11).

C4.1 pins the write-cache contract before it is wired into
``enrich_embeddings`` (that wiring is C4.2/RT10). The two behaviors here:

- RT9 — ``save_embedding`` / ``load_embedding`` key the cache by ``(backend,
  model)``: a vector saved under one namespace round-trips there, while the
  same text under a different model or backend is a separate namespace (a
  miss), and the original entry is neither moved nor invalidated. The in-file
  key is ``sha256(constructed text)`` (NOT a source-file hash).
- RT11 — a ``.npy`` entry at the computed path containing corrupt/unparseable
  bytes is a COUNTED miss: ``load_embedding`` returns None, nothing raises,
  ``_embed_cache_corrupt`` increments, and the corrupt entry stays on disk
  (mirrors cache.py's JSONDecodeError discipline, #2405). This covers TWO
  corruption families: (a) bytes pickle.loads cannot parse, and (b) a payload
  that unpickles fine but is not a valid embedding shape (rejected by
  load_embedding's shape guard).
"""
from __future__ import annotations

import hashlib
import pickle

import pytest

from graphify import cache as _cache


def test_embed_cache_namespace_independence(tmp_path):
    """RT9 — a save under (ollama, nomic-embed-text) with the constructed text
    round-trips under the SAME namespace; switching the model (or backend) must
    not move or invalidate the entry — the same text in another namespace is a
    miss while the original namespace still returns the vector.

    The fixture has NO source file, so a source-file-hash key could never store
    or hit even a single entry: the round-trip itself proves the key is
    sha256(constructed text) in an embed-{backend}-{model}/ directory.
    """
    import graphify.embed as _embed
    from graphify.embed import _embed_texts_key, load_embedding, save_embedding

    text = "Embedding math\ndocs/embeddings.md"
    caller_vec = [0.25, -0.5, 0.75]

    hits_before = _embed._embed_cache_hits

    save_embedding("ollama", "nomic-embed-text", text, caller_vec, root=tmp_path)

    # Same-namespace hit returns the exact stored vector.
    assert (
        load_embedding("ollama", "nomic-embed-text", text, root=tmp_path) == caller_vec
    )

    # Model switch: the new namespace is a miss, the original stays hittable.
    assert (
        load_embedding("ollama", "snowflake-arctic-embed", text, root=tmp_path) is None
    )
    assert (
        load_embedding("ollama", "nomic-embed-text", text, root=tmp_path) == caller_vec
    )

    # Backend switch: the same pair of (label/value) collisions is a miss too.
    assert load_embedding("openai", "nomic-embed-text", text, root=tmp_path) is None

    # Layout pin: the entry lives at
    # graphify-out/cache/embed-{backend}-{model}/sha256(text).npy — a
    # source-file-hash or unnamespaced write cannot satisfy the round-trip.
    entry = _cache.cache_dir(tmp_path, kind="embed-ollama-nomic-embed-text") / (
        f"{hashlib.sha256(text.encode()).hexdigest()}.npy"
    )
    assert entry.is_file()
    assert _embed_texts_key("ollama", "nomic-embed-text") == "embed-ollama-nomic-embed-text"

    # Two successful loads = two recorded hits; misses are not hits.
    assert _embed._embed_cache_hits == hits_before + 2


def test_embed_cache_corrupt_entry_counted_miss_left_in_place(tmp_path):
    """RT11 — BOTH corruption families at the computed path are a counted miss.

    Family 1: unparseable bytes (truncated pickle -> EOFError from
    pickle.loads). Family 2: a payload that unpickles fine but is NOT a valid
    embedding shape — bool/str members, rejected by load_embedding's shape
    guard at embed.py:212-218 — which a shape-guard-removed mutant would return
    as a counted HIT. For EACH: load returns None, no exception escapes,
    _embed_cache_corrupt increments by exactly one, and the corrupt file is
    left in place (not deleted, not rewritten). The two payloads live at
    distinct sha256 paths (distinct text) so the classes are independent."""
    import graphify.embed as _embed
    from graphify.embed import load_embedding

    truncated_text = "API Tokens\ndocs/security.md"
    truncated_path = _cache.cache_dir(
        tmp_path, kind="embed-ollama-nomic-embed-text"
    ) / f"{hashlib.sha256(truncated_text.encode()).hexdigest()}.npy"
    truncated_bytes = (
        b"\x80\x05"  # truncated pickle stream -> EOFError from pickle.loads
    )
    truncated_path.write_bytes(truncated_bytes)
    assert truncated_path.is_file()

    shape_bad_text = "Shape Guard\ndocs/cache.md"
    shape_bad_path = _cache.cache_dir(
        tmp_path, kind="embed-ollama-nomic-embed-text"
    ) / f"{hashlib.sha256(shape_bad_text.encode()).hexdigest()}.npy"
    shape_bad_bytes = pickle.dumps([True, "x"])  # bool + str, invalid shape
    shape_bad_path.write_bytes(shape_bad_bytes)
    assert shape_bad_path.is_file()

    corrupt_before = _embed._embed_cache_corrupt

    # Family 1: unparseable pickle -> counted miss, left in place.
    loaded = load_embedding("ollama", "nomic-embed-text", truncated_text, root=tmp_path)
    assert loaded is None
    assert _embed._embed_cache_corrupt == corrupt_before + 1
    assert truncated_path.is_file()
    assert truncated_path.read_bytes() == truncated_bytes

    # Family 2: parseable-but-invalid-shape payload -> counted miss, left in place.
    loaded = load_embedding("ollama", "nomic-embed-text", shape_bad_text, root=tmp_path)
    assert loaded is None
    assert _embed._embed_cache_corrupt == corrupt_before + 2
    assert shape_bad_path.is_file()
    assert shape_bad_path.read_bytes() == shape_bad_bytes
