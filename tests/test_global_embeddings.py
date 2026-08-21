"""Global re-embed on the post-`global_add` (namespaced) graph.

The sidecar lands as `~/.graphify/embeddings-global.npz` (a sibling of
`global-graph.json`, NOT a per-repo `graphify-out/embeddings.npz`), and every
sidecar id row — in the `text_ids` group AND the `code_ids` group (the code
nodes' own namespace) — carries `repo_tag::local_id`, never a bare local id.

The cross-repo cache dedup lock (cache-dedup-across-repos fixture): two repos'
identical external-library node texts hash to ONE cached vector under the code
namespace, so a re-embed after both `global_add`s issues ZERO additional
`_call_embeddings` calls for repoB's contribution.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import networkx as nx
from unittest.mock import patch

from tests.test_embed_cache import _broadcast_spy_embeddings


def _inject_external(
    G: "nx.Graph", nid: str, attrs: dict[str, object]
) -> "nx.Graph":
    G.add_node(nid, **attrs)
    return G


def test_global_reembed_writes_namespaced_sidecar_to_global_dir(
    tmp_path, monkeypatch
):
    """Re-embedding the POST-`global_add` graph (a) records exactly one seam
    call carrying repoA's NAMESPACED text, (b) a re-run over repoA's frozen
    subgraph issues ZERO calls (warm-cache semantics against the global graph —
    "skipped"-fast-path-independent provenance), (c) ``global_add`` of repoB
    with a fresh source hash issues exactly one new seam call carrying repoB's
    NAMESPACED text, and (d) the written ``embeddings-global.npz`` sits in the
    GLOBAL dir (sibling of ``global-graph.json``, never a per-repo
    ``graphify-out/embeddings.npz``) with ``text_ids`` ==
    ``["repoA::a-doc", "repoB::b-doc"]`` — the namespaced ids, in sorted order."""
    from graphify.global_graph import global_add, global_reembed

    # Two repos, each a text-family graph whose ids are BARE in the source file.
    for repo, nid in (("repoA", "a-doc"), ("repoB", "b-doc")):
        G = nx.Graph()
        G.add_node(
            nid,
            label=f"{repo}_{nid}",
            source_file=f"docs/{nid}.md",
            file_type="document",
        )
        src = tmp_path / f"{repo}-graph.json"
        src.write_text(
            json.dumps(nx.node_link_data(G, edges="links")), encoding="utf-8"
        )

    global_dir = tmp_path / "global"
    recorded: list = []

    with patch("graphify.global_graph._GLOBAL_DIR", global_dir), \
         patch("graphify.global_graph._GLOBAL_GRAPH", global_dir / "global-graph.json"), \
         patch("graphify.global_graph._GLOBAL_MANIFEST", global_dir / "global-manifest.json"):
        import graphify.embed as _embed
        # Real `enrich_embeddings` from the START — the seam's cache is live so
        # the re-run and hence repoB's add are all measured cache behavior.
        _broadcast_spy_embeddings(monkeypatch, recorded)

        global_add(tmp_path / "repoA-graph.json", "repoA")
        result1 = global_reembed()

        assert result1["written"] is True
        assert result1["nodes"] == 1
        assert str(result1["sidecar"]) == str(
            global_dir / "embeddings-global.npz"
        )
        assert len(recorded) == 1
        assert recorded[0][1]["inputs"] == ["repoA_a-doc\ndocs/a-doc.md\n\n"]

        # The seam request's text came from the REAL `build_node_text` pipeline,
        # with zero fixture-built input strings.
        assert recorded[0][1]["inputs"][0] == _embed.build_node_text(
            _embed_enriched_global_graph(),
            "repoA::a-doc",
            _embed_enriched_global_graph().nodes["repoA::a-doc"],
        )

        recorded.clear()
        result2 = global_reembed()
        assert result2["written"] is True
        assert len(recorded) == 0  # warm cache, real hit

        recorded.clear()
        global_add(tmp_path / "repoB-graph.json", "repoB")
        result3 = global_reembed()
        assert result3["written"] is True
        assert result3["nodes"] == 2
        assert len(recorded) == 1
        assert recorded[0][1]["inputs"] == [
            "repoB_b-doc\ndocs/b-doc.md\n\n"
        ]

    assert (global_dir / "embeddings-global.npz").is_file()
    with np.load(global_dir / "embeddings-global.npz") as data:
        assert list(data["text_ids"]) == ["repoA::a-doc", "repoB::b-doc"]
    assert not (tmp_path / "embeddings.npz").exists()


def _embed_enriched_global_graph() -> "nx.Graph":
    """Rebuild the post-add global graph's data for the pure-function
    construction check.

    The global node is the graph node in ``_load_global_graph()``; the composed
    node text is a pure function of that graph. The enrichment machinery reads
    the SAME text the ``_load_global_graph()`` round-trip produces per graph —
    those values are the ones the sidecar pins.
    """
    from graphify.global_graph import _load_global_graph

    return _load_global_graph()


def test_cache_dedup_across_repos_shared_external_library_node(
    tmp_path, monkeypatch
):
    """Cross-repo cache dedup under the two-space partition.

    The fixture's shared external node is code-space under the partition, so
    its text is embedded under the ``(ollama, nomic-embed-code)`` namespace
    while the two document nodes ride ``(ollama, nomic-embed-text)``. The cache
    key stays ``sha256(constructed text)`` — repoB's external code node carries
    an IDENTICAL constructed text to repoA's, so the code namespace legitimately
    holds ONE entry after repoA's add. ``global_add`` then dedups repoB's
    external node OUT of the merged graph (its bare id is remapped to
    ``repoA::requests``), and a re-embed over the post-``global_add`` graph
    serves the shared text from the warm cache with ZERO seam calls.

    Sequence pinned through the real ``enrich_embeddings`` + real
    ``sha256(text)`` cache + real seam spy (``_broadcast_spy_embeddings``):

    1. Both ``global_add``s MUST run before the first re-embed (the skip guard
       destroys add-ordering if the source is unchanged). ``global_reembed()``
       then issues one seam call per space: the text call carries the two
       documents' texts, the code call carries only the shared external text.
    2. The code text's cache entry exists at the exactly derived path
       ``{global-dir}/graphify-out/cache/embed-ollama-nomic-embed-code/
       sha256("HTTP requests\\n\\n\\n").npy`` — the repo-agnostic key.
    3. A SECOND ``global_reembed()`` over the unchanged graph issues ZERO calls
       total. THIS is the dedup observation: repoB's contribution adds no NEW
       embedding — its id was canonicalized to ``repoA::requests`` by
       ``global_add`` and its text's vector is served from the shared cache
       entry.
    4. The final written sidecar pins ``text_ids`` == ``["repoA::acceptance",
       "repoB::api"]`` (the documents) and ``code_ids`` == ``["repoA::requests"]``
       (the shared external node, canonicalized to repoA's id — never a second
       copy under repoB's bare ``http``). Every id is ``repo_tag::local_id``.
    """
    from graphify.global_graph import global_add, global_reembed
    from graphify.embed import _embed_texts_key
    from graphify import cache as _cache

    # Per repo: a distinct text-family document node (with a ``source_file``)
    # plus an external-library code node whose ID is DIFFERENT across the two
    # repos ("requests" vs "http") but whose constructed text is IDENTICAL
    # (secure label, empty path, empty rationale/neighbour lines). The external
    # node carries NO ``source_file`` -- absent it, ``global_add``'s
    # label-based dedup key matches across repos.
    for repo, doc_id, src_file, doc_label, ext_id in (
        ("repoA", "acceptance", "docs/acceptance.md", "Acceptance flow", "requests"),
        ("repoB", "api", "docs/api.md", "API flow", "http"),
    ):
        G = nx.Graph()
        G.add_node(
            doc_id,
            label=doc_label,
            source_file=src_file,
            file_type="document",
        )
        G.add_node(
            ext_id,
            label="HTTP requests",
            file_type="code",
        )
        src = tmp_path / f"{repo}-graph.json"
        src.write_text(
            json.dumps(nx.node_link_data(G, edges="links")), encoding="utf-8"
        )

    global_dir = tmp_path / "global"

    with patch("graphify.global_graph._GLOBAL_DIR", global_dir), \
         patch("graphify.global_graph._GLOBAL_GRAPH", global_dir / "global-graph.json"), \
         patch("graphify.global_graph._GLOBAL_MANIFEST", global_dir / "global-manifest.json"):
        import graphify.embed as _embed
        recorded: list = []
        _broadcast_spy_embeddings(monkeypatch, recorded)

        # BOTH adds first, so the re-embed sees the full post-add state at once
        # (the skip guard destroys add-ordering if the source is unchanged).
        global_add(tmp_path / "repoA-graph.json", "repoA")
        global_add(tmp_path / "repoB-graph.json", "repoB")

        result1 = global_reembed()
        assert result1["written"] is True
        assert result1["nodes"] == 3  # acceptance + requests + api
        # Cold run over the 3 post-add nodes, ONE seam call per space: the text
        # call carries the two documents' constructed texts; the code call
        # carries ONLY the shared external text — under the canonical
        # ``repoA::requests`` id (repoB's bare ``http`` was deduped out by
        # ``global_add``, so it contributes no row). The two calls are
        # SET-compared: the space iteration order is an implementation detail.
        assert len(recorded) == 2, recorded
        assert {k["model"] for _, k in recorded} == {
            "nomic-embed-text",
            "nomic-embed-code",
        }
        by_model = {k["model"]: k["inputs"] for _, k in recorded}
        assert set(by_model["nomic-embed-text"]) == {
            "Acceptance flow\ndocs/acceptance.md\n\n",
            "API flow\ndocs/api.md\n\n",
        }
        assert by_model["nomic-embed-code"] == ["HTTP requests\n\n\n"]

        # The repo-agnostic cache key: the shared code text's entry exists at
        # the sha256(constructed text) path under the CODE namespace (the
        # ``(ollama, nomic-embed-code)`` kind). A source-file-hash key, a
        # repo-tagged key, or a text-namespace write would place the vector
        # elsewhere and the warm re-run below would MISS it — recording a call.
        cache_entry = (
            _cache.cache_dir(global_dir, kind=_embed_texts_key("ollama", "nomic-embed-code"))
            / f"{hashlib.sha256('HTTP requests\n\n\n'.encode()).hexdigest()}.npy"
        )
        assert cache_entry.is_file()

        # WARM re-run over the unchanged post-add global graph: repoB's
        # external node was dedup-canonicalized to repoA::requests (its text
        # is the SAME string), so the vector is served from the shared cache
        # entry — ZERO additional seam calls. This is the dedup observation.
        recorded.clear()
        result2 = global_reembed()
        assert result2["written"] is True
        assert result2["nodes"] == 3
        assert len(recorded) == 0, recorded

    sidecar = global_dir / "embeddings-global.npz"
    assert sidecar.is_file()
    with np.load(sidecar) as data:
        assert list(data["text_ids"]) == ["repoA::acceptance", "repoB::api"]
        assert list(data["code_ids"]) == ["repoA::requests"]


def test_global_sidecar_no_bare_local_ids(tmp_path, monkeypatch):
    """Structural lock: sidecar ids are namespaced, on write and fetch.

    Under the two-space partition the code node no longer rides ``text_ids``:
    the sidecar carries the text group for the documents and a code group for
    the external code node, and EVERY row of BOTH groups keeps the
    ``repo_tag::local_id`` form — never a bare local id, with no special-casing.
    Sole-reason fixture design per leg (every bare id in the SOURCE graphs is a
    different string from both its namespaced row and its constructed text, so
    no leg can pass unless the real prefixing is at work):

    1. ``text_ids`` equals the exact NAMESPACED document set — sorted rows
       ``["repoA::architect", "repoB::gateway"]`` (set equality with the
       namespaced ids, plus the exact-sorted-list shape), and ``code_ids``
       equals ``["repoA::requests"]``. The source graphs are written with BARE
       ids; only ``global_add`` + ``enrich_embeddings`` can produce these rows.
    2. No row of either group equals ANY repo's bare local id — membership
       absence of ``{"architect", "requests", "gateway"}`` across both groups.
       This is a separate signal from leg 1: a drop-the-prefix mutant (bare
       rows) passes NOTHING here, while a wrong-prefix mutant ("x::architect")
       passes leg 2 but fails leg 1's exact equality — both mutants are needed
       for the sandwich.
    3. Fetch-side proxy — the ``"repoA::requests" != "requests" != the text``
       invariant made observable at the code cache: the constructed text of the
       namespaced external element ``repoA::requests`` is ``"HTTP requests\\n
       \\n\\n"`` (never its bare id, never its namespaced id — the id never
       enters the embedding path). Querying the REAL text-keyed code cache
       (``load_embedding``, which computes sha256(text) internally — the test
       does NOT re-derive the key) proves a bare-id-shaped input and even a
       namespaced-id-shaped input fetch NOTHING, while the text fetches the
       cached vector. Then the code cache is cleared and a re-embed re-fetches
       the namespaced code element under its text — the seam receives the exact
       constructed code text (never an id-shaped input), and the re-written
       sidecar is still the namespaced set.
    """
    from graphify.global_graph import global_add, global_reembed
    from graphify.embed import _embedding_cache_dir, build_node_text, load_embedding
    import graphify.embed as _embed
    from shutil import rmtree

    # Two repos, three nodes whose ids are BARE in the source files.
    # The external code node carries NO source_file (matches global_add's
    # external-label dedup shape) but its label is unique, so it is not deduped.
    # One graph FILE per repo — each contains all of that repo's nodes.
    repo_nodes: dict[str, list[tuple]] = {
        "repoA": [
            ("architect", "docs/arch.md", "Architecture", "document"),
            ("requests", None, "HTTP requests", "code"),
        ],
        "repoB": [
            ("gateway", "docs/gateway.md", "API gateway", "document"),
        ],
    }
    for repo, nodes in repo_nodes.items():
        G = nx.Graph()
        for nid, src_file, label, ftype in nodes:
            attrs = {"label": label, "file_type": ftype}
            if src_file is not None:
                attrs["source_file"] = src_file
            G.add_node(nid, **attrs)
        src = tmp_path / f"{repo}-graph.json"
        src.write_text(
            json.dumps(nx.node_link_data(G, edges="links")), encoding="utf-8"
        )

    global_dir = tmp_path / "global"
    recorded: list = []

    with patch("graphify.global_graph._GLOBAL_DIR", global_dir), \
         patch("graphify.global_graph._GLOBAL_GRAPH", global_dir / "global-graph.json"), \
         patch("graphify.global_graph._GLOBAL_MANIFEST", global_dir / "global-manifest.json"):
        _broadcast_spy_embeddings(monkeypatch, recorded)

        for repo in ("repoA", "repoB"):
            global_add(tmp_path / f"{repo}-graph.json", repo)
        result = global_reembed()
        assert result["written"] is True
        # One seam call per space: text carries the two documents, code carries
        # the external code node's text (call order is an impl detail — the
        # per-space inputs are SEPARATELY asserted, never order-linked).
        assert len(recorded) == 2, recorded
        assert {k["model"] for _, k in recorded} == {
            "nomic-embed-text",
            "nomic-embed-code",
        }
        by_model = {k["model"]: k["inputs"] for _, k in recorded}

        # Leg 1 — the exact namespaced rows: two text rows, one code row (set
        # equality vs the namespaced sets, plus the exact sorted-list shapes
        # the sidecar pins).
        expected_ids = ["repoA::architect", "repoB::gateway"]
        code_id = "repoA::requests"

        # Leg 3 fixture premise, cross-checked against the REAL pipeline: the
        # constructed texts are neither the bare ids nor the namespaced ids, so
        # so the later id-shaped cache-input negatives are meaningfully distinct.
        G_live = _embed_enriched_global_graph()
        expected_texts = [
            build_node_text(G_live, nid, G_live.nodes[nid]) for nid in expected_ids
        ]
        code_text = build_node_text(G_live, code_id, G_live.nodes[code_id])
        for nid, text in zip(expected_ids, expected_texts):
            assert nid != text
            assert nid.split("::", 1)[1] != text  # bare form != text too
        assert code_id != code_text
        assert code_id.split("::", 1)[1] != code_text  # bare form != text too
        assert set(by_model["nomic-embed-text"]) == set(expected_texts)
        assert by_model["nomic-embed-code"] == [code_text]

        # Leg 2 — a text-keyed cache has no id dimension: a bare id can never
        # fetch the namespaced element, and neither can the namespaced id, in
        # EITHER namespace.
        assert (
            load_embedding("ollama", "nomic-embed-text", "architect", root=global_dir)
            is None
        )
        assert (
            load_embedding("ollama", "nomic-embed-code", "requests", root=global_dir)
            is None
        )
        assert (
            load_embedding("ollama", "nomic-embed-code", code_id, root=global_dir)
            is None
        )
        # …only the constructed TEXT fetches it — the code model for the code
        # row, the text model for a document row.
        assert (
            load_embedding("ollama", "nomic-embed-code", code_text, root=global_dir)
            == [1.0]
        )
        assert (
            load_embedding(
                "ollama", "nomic-embed-text", expected_texts[0], root=global_dir
            )
            == [1.0]
        )

        # Cold re-fetch after the CODE cache alone is cleared: the text group
        # stays warm (document vectors served from their entries), so the re-run
        # issues exactly ONE call — the code-model seam re-deriving the
        # namespaced code element from its TEXT, never an id-shaped input.
        rmtree(_embedding_cache_dir(global_dir, "ollama", "nomic-embed-code"))
        recorded.clear()
        assert global_reembed()["written"] is True
        assert len(recorded) == 1, recorded
        code_fetched = recorded[0][1]["inputs"]
        assert recorded[0][1]["model"] == "nomic-embed-code"
        assert code_fetched == [code_text]
        assert set(code_fetched).isdisjoint({"architect", "requests", "gateway"})

    sidecar = global_dir / "embeddings-global.npz"
    assert sidecar.is_file()
    with np.load(sidecar) as data:
        rows = list(data["text_ids"])
        code_rows = list(data["code_ids"])
        # Leg 1 — equality with the exact namespaced sets, both groups.
        assert set(rows) == set(expected_ids)
        assert rows == expected_ids
        assert list(code_rows) == [code_id]
        # Leg 2 — no row of EITHER group equals any repo's bare local id
        # (membership absence).
        for bare in ("architect", "requests", "gateway"):
            assert bare not in rows
            assert bare not in code_rows
