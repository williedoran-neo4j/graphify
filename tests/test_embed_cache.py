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
import shutil
from shutil import rmtree

import numpy as np
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


# ---- R5/C5.2 — RT15: the rationale line (line 3) PRECEDES the neighbour
# block, and a rationale-absent node falls back to label + source_file — as
# they flow through the REAL enrich_embeddings + sha256(text) cache-key path.
#
# Expected GREEN-ON-ARRIVAL (revalidation): C5.1's accepted build_node_text
# composes exactly this, so the lock passes on first run against the C5.2
# pre-state. TEETH is about a naive constructor: a "rationale concatenated
# AFTER the neighbour block" build would compose a DIFFERENT text for the
# rationale node — the pre-seeded entry below (keyed on the constructed text)
# would not hit, and the warm run's seam inputs would then contain the
# rationale text, not just the fallback neighbour's. A "label+source_file only
# when rationale absent" constructor would drop n-f's neighbour line, making
# its expected full-string equality fail. Both mutations flip the test.


def test_rt15_rationale_precedes_neighbour_block_via_cache_key(tmp_path, monkeypatch):
    """RT15 — the rationale-when-present text and the rationale-absent
    fallback, as observed at the cache-key seam of the REAL enrich_embeddings.

    Two connected text-family documents: n-r (has ``rationale``) and n-f
    (none). Cold run must call the seam with BOTH texts. Then the cache dir is
    rmtree'd and ONLY n-r's entry is re-seeded — keyed on the constructed text
    that carries the rationale on line 3. The warm run must:
      - serve n-r from cache (its text — rationale BEFORE the neighbour block —
        is the exact cache key), recording ZERO seam calls for it;
      - miss n-f's entry and issue the ONLY warm seam call to its text, which
        equals the exact fallback form
        ``{label}\n{source_file}\n\n{_neighbour_text(...)}``.

    The two behaviors are SEPARATELY pinned: deleting the n-r pre-write would
    still leave the cold-rationale-order assertions exact, while any
    rationale-after-neighbour (or fallback dropping the neighbour line)
    constructor breaks the n-r cache hit and/or the n-f exact text.
    """
    import graphify.embed as _embed
    from graphify.embed import (
        _embedding_cache_dir,
        _neighbour_text,
        build_node_text,
        enrich_embeddings,
        save_embedding,
    )

    g = _graph(
        {
            "id": "n-r",
            "label": "API Tokens",
            "source_file": "docs/security.md",
            "rationale": "credentials policy guidance",
            "file_type": "document",
        },
        {
            "id": "n-f",
            "label": "Embedding math",
            "source_file": "docs/embeddings.md",
            "file_type": "document",
        },
    )
    g.add_edge("n-r", "n-f")
    graph_path = tmp_path / "graph.json"

    # Reference construction (documented C2 lines): L3 = rationale (line 3,
    # BEFORE the neighbour line) for n-r; L3 = "" for n-f.
    ref_rationale_text = (
        f"API Tokens\ndocs/security.md\ncredentials policy guidance\n{_neighbour_text(g, 'n-r')}"
    )
    ref_fallback_text = f"Embedding math\ndocs/embeddings.md\n\n{_neighbour_text(g, 'n-f')}"

    # The seam records inputs only; the real embedding skeleton (dim 1) stays.
    calls: list = []
    _broadcast_spy_embeddings(monkeypatch, calls)
    enrich_embeddings(g, graph_path)

    # Cold run: BOTH texts reach the cache-key seam — the cache held nothing
    # (list order is enrich's sorted-id iteration, an irrelevant detail here).
    assert len(calls) == 1
    cold_inputs = set(list(kwargs["inputs"] for _, kwargs in calls)[0])
    assert cold_inputs == {ref_rationale_text, ref_fallback_text}

    # Kill the cache, re-seed ONLY n-r's entry — keyed on the constructed text
    # with the rationale on line 3.
    rmtree(_embedding_cache_dir(tmp_path, "ollama", "nomic-embed-text"))
    save_embedding("ollama", "nomic-embed-text", ref_rationale_text, [1.0], root=tmp_path)

    # Warm run: n-r hits (exact key), n-f misses — the ONLY seam call is n-f's
    # exact fallback text. A rationale-after-neighbour constructor's n-r text
    # would NOT be the pre-write's key and would therefore leak into this list.
    calls.clear()
    enrich_embeddings(g, graph_path)
    assert len(calls) == 1, calls
    warm_inputs = list(kwargs["inputs"] for _, kwargs in calls)[0]
    assert warm_inputs == [ref_fallback_text]

    # Sanity: the constructor on disk composes the exact texts above.
    assert build_node_text(g, "n-r", g.nodes["n-r"]) == ref_rationale_text
    assert build_node_text(g, "n-f", g.nodes["n-f"]) == ref_fallback_text
# prune_embedding_cache (RT12), and clear_cache's embed kinds (C4.3 folded).
#
# RT10's discriminating leg: with the cache dir rm -rf'd between runs, the
# second run MUST call the seam again — proving HIT LOGIC (not the fixture)
# suppresses the warm re-call. Every leg must run through the real seam's
# callable (a delegating spy) so the zero-call assert is a measured side
# effect of the real path, and the vectors driven into the cache are the real
# backend's — never a fixture-only value a dumb `not text` check could fake.


def _graph(*node_attrs) -> "object":
    """A real networkx.Graph (test_embed.py's fixture shape): each node dict
    carries id/label/source_file/file_type so build_node_text (R1) can select
    the text family."""
    import networkx as nx

    g = nx.Graph()
    g.add_nodes_from((a["id"], a) for a in node_attrs)
    return g


def _broadcast_spy_embeddings(monkeypatch, recorded):
    """Real seam + delegating spy: records inputs then embeds via _call_embeddings.

    ``*args``/``**kwargs`` so the keyword call shape
    (``backend=..., model=..., inputs=...``) of the real seam is recorded, not
    mangled. Dimension 1 keeps every vector unit-norm out of the box.
    """
    import graphify.embed as _embed

    def spy(*args, **kwargs):
        recorded.append((args, kwargs))
        return [[1.0] for _ in kwargs["inputs"]]

    monkeypatch.setattr(_embed, "_call_embeddings", spy)


def test_enrich_zero_call_re_run_warm_cache(tmp_path, monkeypatch):
    """RT10 — a warm-cache re-run must issue ZERO ``_call_embeddings`` calls,
    and the two sidecars must be identical in ``text_ids`` and ``text_vecs``.

    Plan's pinned shape: cold run records exactly ONE call (sidecar written);
    ``calls.clear()``; warm run records ZERO (the cache suppressed it); the
    second ``embeddings.npz`` byte-matches on ids and allclose() on vecs.
    Real-red discriminator: delete the cache dir, run a THIRD time — the seam
    must be called again (exactly one), proving HIT LOGIC, not the fixture's
    short-circuited inputs, suppresses the warm call. ``root`` is
    ``Path(graph_path).parent`` per plan. Every leg needs both a hit register
    and a fresh counter block.
    """
    import graphify.embed as _embed
    from graphify.embed import enrich_embeddings

    g = _graph(
        {
            "id": "n-a-01",
            "label": "API Tokens",
            "source_file": "docs/security.md",
            "file_type": "document",
        },
        {
            "id": "n-b-02",
            "label": "Embedding math",
            "source_file": "docs/embeddings.md",
            "file_type": "document",
        },
        {
            "id": "n-IMG-07",
            "label": "architecture diagram",
            "source_file": "assets/arch.png",
            "file_type": "image",
        },
    )
    graph_path = tmp_path / "graph.json"

    # Cold run: the cache lets nothing through; the seam must be hit exactly once.
    calls: list = []
    _broadcast_spy_embeddings(monkeypatch, calls)
    hits_before = _embed._embed_cache_hits
    out1 = enrich_embeddings(g, graph_path)
    assert len(calls) == 1, calls

    # Warm run over the unchanged graph: zero calls, identical sidecar state.
    calls.clear()
    out2 = enrich_embeddings(g, graph_path)
    assert len(calls) == 0, calls
    with np.load(out2) as data2, np.load(out1) as data1:
        assert [str(s) for s in data2["text_ids"]] == [str(s) for s in data1["text_ids"]]
        assert np.allclose(data2["text_vecs"], data1["text_vecs"])
    # The cache swallowed both texts — not a side-effect-free re-embed.
    assert _embed._embed_cache_hits >= hits_before + 2

    # Real-red discriminator: rm -rf the cache dir, run again — hit logic must
    # be the suppressed part, so the seam is called exactly once more.
    from graphify.embed import _embedding_cache_dir

    cache_root = _embedding_cache_dir(tmp_path, "ollama", "nomic-embed-text")
    shutil.rmtree(cache_root)
    calls.clear()
    out3 = enrich_embeddings(g, graph_path)
    assert len(calls) == 1, calls
    with np.load(out3) as data3, np.load(out1) as data1:
        assert [str(s) for s in data3["text_ids"]] == [str(s) for s in data1["text_ids"]]
        assert np.allclose(data3["text_vecs"], data1["text_vecs"])


def test_prune_embedding_cache_and_namespace_coverage(tmp_path):
    """RT12 + C4.3-folded — the prune helper removes orphans and nothing else.

    Builds two (backend, model) namespaces directly (save_embedding) and prunes
    the first. ``prune_embedding_cache`` does not exist yet, so this test's
    FAILURE is the right-reason red (attribute error on import). Three closing
    assertions fold in C4.3:

    - pruning under the FIRST kind returns the count >= 1 and the orphaned
      entry is gone;
    - the SECOND namespace's entries are left intact (its key's own regex would
      never match ``embed-ollama-nomic-embed-text``, so only a real prune can
      produce the removal);
    - ``clear_cache`` removes embed-kind entries too — today its kind loop stops
      at ``semantic-deep``, so the orphaned entry survives (fails).
    """
    from graphify import cache as _cache
    from graphify.embed import (
        _embed_texts_key,
        load_embedding,
        prune_embedding_cache,
        save_embedding,
    )

    # Two namespaces, three texts. The transient text is an orphan of kind1.
    kind1 = ("ollama", "nomic-embed-text")
    kind2 = ("openai", "nomic-embed-text")
    live1 = "Embedding math\ndocs/embeddings.md"
    live2 = "API Tokens\ndocs/security.md"
    orphan1 = "Cargo Graph\ndocs/other.md"
    for text in (live1, live2):
        save_embedding(*kind1, text, [0.25], root=tmp_path)
        save_embedding(*kind2, text, [0.5, -0.5], root=tmp_path)
    save_embedding(*kind1, orphan1, [0.125], root=tmp_path)

    kind1_dir = _cache.cache_dir(tmp_path, kind=_embed_texts_key(*kind1))

    # The orphans are concretely present (2 files in kind1 before pruning).
    assert list(kind1_dir.glob("*.npy"))
    assert len(list(kind1_dir.glob("*.npy"))) == 3

    # RT12 — the helper prunes the first namespace's orphaned entry.
    pruned = prune_embedding_cache(
        root=tmp_path, backend="ollama", model="nomic-embed-text",
        live_texts={live1, live2},
    )
    assert pruned >= 1
    assert not (kind1_dir / f"{hashlib.sha256(orphan1.encode()).hexdigest()}.npy").exists()

    # C4.3 — the second namespace kept its orphan-of-kind2 (different hash),
    # and pruning never touched the actually-live vectors (they still hit).
    kind2_dir = _cache.cache_dir(tmp_path, kind=_embed_texts_key(*kind2))
    assert (kind2_dir / f"{hashlib.sha256(live1.encode()).hexdigest()}.npy").is_file()
    assert (kind2_dir / f"{hashlib.sha256(live2.encode()).hexdigest()}.npy").is_file()
    assert load_embedding("ollama", "nomic-embed-text", live1, root=tmp_path) == [0.25]
    assert load_embedding("ollama", "nomic-embed-text", live2, root=tmp_path) == [0.25]

    # C4.3 — clear_cache must remove embed entries under the embed kinds too
    # (glob **/*.npy). Today cache.py's loop stops at semantic-deep, so the
    # still-alive kind1 entry survives — making this leg fail (red).
    _cache.clear_cache(tmp_path)
    assert not (kind1_dir / f"{hashlib.sha256(live1.encode()).hexdigest()}.npy").exists()
