"""R6/C6.1 — global re-embed: the post-`global_add` (namespaced) graph is what
gets embedded, the sidecar lands as `~/.graphify/embeddings-global.npz` (a
sibling of `global-graph.json`, NOT a per-repo `graphify-out/embeddings.npz`),
and every `text_ids` row carries `repo_tag::local_id` — never a bare local id
(C12 bullets 1/2/3, I14)."""
from __future__ import annotations

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
    """C6.1 — re-embedding the POST-`global_add` graph (a) records exactly one
    seam call carrying repoA's NAMESPACED text, (b) a re-run over repoA's frozen
    subgraph issues ZERO calls (R4 RT10 semantics against the global graph —
    "skipped"-fast-path-independent provenance), (c) ``global_add`` of repoB
    with a fresh source hash issues exactly one new seam call carrying repoB's
    NAMESPACED text, and (d) the written ``embeddings-global.npz`` sits in the
    GLOBAL dir (sibling of ``global-graph.json``, never a per-repo
    ``graphify-out/embeddings.npz``) with ``text_ids`` ==
    ``["repoA::a-doc", "repoB::b-doc"]`` — the namespaced ids, in sorted order
    (I14 guard)."""
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
         patch("graphify.global_graph._GLOBAL_MANIFEST", global_dir / "global-manifest.json"), \
         patch("graphify.global_graph._GLOBAL_EMBEDDINGS", global_dir / "embeddings-global.npz"):
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

        # Dependency layout invariant: the seam request whose text came from the
        # REAL `build_node_text` pipeline, with zero fixture-built input strings.
        assert recorded[0][1]["inputs"][0] == _embed.build_node_text(
            _embed_enriched_global_graph(),
            "repoA::a-doc",
            _embed_enriched_global_graph().nodes["repoA::a-doc"],
        )

        recorded.clear()
        result2 = global_reembed()
        assert result2["written"] is True
        assert len(recorded) == 0  # R4 RT10 — warm cache, real hit

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
    construction check (identity-case law enforcement).

    R6's self-consistency anchor (plan "Execution notes"): global_node = the
    graph node in ``_load_global_graph()``; the composed node text is a pure
    function of that graph. The enrichment machinery reads the SAME text the
    ``_load_global_graph()`` round-trip produces per graph — those values are
    the ones the sidecar pins.
    """
    from graphify.global_graph import _load_global_graph

    return _load_global_graph()
