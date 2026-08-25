"""R9 C2 — the semantic fold into the query-graph ranking, gated behind the
blend weight, with lexical priority.

Pins the behavior the query-graph path must have once a nonzero gate weight is
injected and an embeddings sidecar is present beside the graph:
- a vector near-match with no exact lexical label is lifted into the ranking,
- an exact lexical label keeps rank 1 at any weight (lexical priority),
- the default weight (0.0) leaves the semantic sidecar unread, so the render is
  byte-identical to the fold-off baseline.
"""
import asyncio
import json
import math

import networkx as nx
import numpy as np
from mcp.types import CallToolRequest, CallToolRequestParams

from graphify import serve as serve_mod
from graphify.embed import _STUB_DIM
from graphify.search import _stub_query_embed
from graphify.serve import _score_query


def _semantics(sidecar_path, query):
    """Per-node text-space dot-product scores for a query under the stub embed."""
    sidecar = np.load(sidecar_path, allow_pickle=False)
    try:
        q = np.asarray(_stub_query_embed(query, space="text", meta={}), dtype=np.float32)
        scores = sidecar["text_vecs"] @ q
        return {str(nid): float(score) for nid, score in zip(sidecar["text_ids"], scores)}
    finally:
        sidecar.close()


def _render(server, question, semantic_weight=None):
    """Render `query_graph` for a question on a server; returns the text."""
    if semantic_weight is None:
        arguments = {"question": question, "mode": "bfs", "depth": 1}
    else:
        arguments = {
            "question": question,
            "mode": "bfs",
            "depth": 1,
            "semantic_weight": str(semantic_weight),
        }
    result = server.request_handlers[CallToolRequest](
        CallToolRequest(params=CallToolRequestParams(name="query_graph", arguments=arguments))
    )
    if asyncio.iscoroutine(result):
        result = asyncio.run(result)
    return result.root.content[0].text


def test_query_graph_blend_semantic_near_miss_joins_ranking(tmp_path, monkeypatch):
    """A semantically-near node without an exact lexical label joins the ranked
    list once a nonzero blend weight is injected and the sidecar's scores are
    pulled, while the exact-label node keeps rank 1 at the same weight.

    The weight is deliberately far above the value where a linear (uncapped)
    fold — lift = semantic score x weight — flips rank 1 onto the near-miss
    (score_analog x w > score_widget_lexical + score_widget x w happens near
    w ~ 7000 for this fixture), so the exact-label node's rank-1 pin only
    holds if the fold is capped beneath the exact-match tier.
    """
    import graphify.serve as serve

    G = nx.Graph()
    G.add_node("n-widget", label="Widget", source_file="widget.py",
               source_location="L1", community=0)
    G.add_node("n-analog", label="Analog Mechanism", source_file="analog.py",
               source_location="L1", community=0)
    G.add_node("n-other", label="Unrelated Leaf", source_file="other.py",
               source_location="L1", community=1)
    G.add_edge("n-widget", "n-analog", relation="calls", confidence="EXTRACTED")

    ianalog = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    vectors = math.sqrt(sum(c * c for c in ianalog))
    rows = np.array(
        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         ianalog,
         [0.1, 0.1, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0]],
        dtype=np.float32,
    ) / vectors
    meta = {"model": "nomic-embed-text", "backend": "ollama", "dim": _STUB_DIM,
            "graphify_version": "test", "created_at": "2026-08-18T00:00:00+00:00"}
    np.savez(corpus / "embeddings.npz",
             text_ids=np.array(["n-widget", "n-analog", "n-other"], dtype=str),
             text_vecs=rows, text_meta=json.dumps(meta))
    semantic = _semantics(corpus / "embeddings.npz", "widget")
    assert semantic["n-analog"] > semantic["n-widget"], "fixture must be a semantic near-miss"
    graph_file = corpus / "graph.json"
    graph_file.write_text(
        json.dumps({
            "directed": False,
            "nodes": [
                {"id": "n-widget", "label": "Widget", "source_file": "widget.py",
                 "source_location": "L1", "community": 0, "file_type": "concept"},
                {"id": "n-analog", "label": "Analog Mechanism", "source_file": "analog.py",
                 "source_location": "L1", "community": 0, "file_type": "concept"},
                {"id": "n-other", "label": "Unrelated Leaf", "source_file": "other.py",
                 "source_location": "L1", "community": 1, "file_type": "document"},
            ],
            "edges": [
                {"source": "n-widget", "target": "n-analog", "relation": "calls",
                 "confidence": "EXTRACTED"},
            ],
        }),
        encoding="utf-8",
    )

    baseline = _score_query(G, ["widget"], collect_per_term_seeds=True)
    assert baseline.ranked[0][1] == "n-widget", "exact-label node must be the rank-1 seed at weight 0"

    original = serve._score_query
    recorded = []

    def recording(*args, **kwargs):
        recorded.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr("graphify.serve._score_query", recording)
    monkeypatch.setattr(
        "graphify.search._real_query_embed",
        lambda query, *, space, meta: [1.0, 2.0, 0.5, 0, 0, 0, 0, 0],
    )
    server = serve_mod._build_server(str(graph_file))
    text = _render(server, "widget", semantic_weight=10000)

    assert len(recorded) == 1
    _call_args, call_kwargs = recorded[0]
    assert call_kwargs.get("semantic_weight") == 10000
    got_semantic = call_kwargs.get("semantic_scores")
    assert got_semantic is not None, "a nonzero gate weight must obtain the sidecar's semantic scores"
    near_lift = got_semantic.get("n-analog", 0.0)
    widget_lift = got_semantic.get("n-widget", 0.0)
    assert near_lift > 0.0, "the near-miss node carries no semantic lift"
    assert near_lift > widget_lift

    open_ = _score_query(
        G, ["widget"], collect_per_term_seeds=True,
        semantic_weight=10000, semantic_scores=got_semantic,
    )
    closed = _score_query(G, ["widget"], collect_per_term_seeds=True)
    assert "n-analog" not in closed.best_seed_by_term, (
        "the near-miss must be invisible to the closed gate (weight 0)"
    )

    # The open gate admits the near-miss to the ranking, the closed gate does
    # not — the fold's observable reach, not a BFS render artefact (n-analog is
    # a depth-1 neighbour of the seeded n-widget and renders regardless).
    open_ranked_ids = [nid for _score, nid in open_.ranked]
    closed_ranked_ids = [nid for _score, nid in closed.ranked]
    assert "n-analog" in open_ranked_ids, (
        "the lifted near-miss must enter the ranking behind the open gate "
        f"(10000): {open_ranked_ids}"
    )
    assert "n-analog" not in closed_ranked_ids, (
        "the near-miss must not enter the ranking with the real sidecar still "
        f"beside the graph at the closed gate (0): {closed_ranked_ids}"
    )
    # The fold's lift must stay beneath the exact-match tier: the exact-label
    # node keeps rank 1 even though the linear lift from this weight alone
    # hands the near-miss a higher combined score.
    assert open_.ranked[0][1] == "n-widget", (
        "the exact-label node must keep rank 1 behind any gate value; the "
        "uncapped linear fold flips it to the near-miss at this weight "
        f"(open rank: {open_ranked_ids})"
    )


def test_query_graph_blend_default_weight_unchanged_rendering(tmp_path, monkeypatch):
    """query_graph at the default blend weight (0.0) must never consult the
    semantic sidecar.

    The blend gate lives at the tool boundary: a query without an explicit
    semantic_weight must leave the sidecar unread (functions serially, so a call
    even to pull-and-ignore would keep it off — or a stray nonzero weight would
    fold scores in) and render byte-identically to the no-sidecar baseline. A
    spy on the sidecar-pull seam is the assertion of that gate; a real sibling
    embeddings.npz stands beside the graph so the seam has a pull to take.
    """
    import graphify.serve as serve

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    vectors = math.sqrt(3)
    rows = np.array(
        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.1, 0.1, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0]],
        dtype=np.float32,
    ) / vectors
    meta = {"model": "nomic-embed-text", "backend": "ollama", "dim": _STUB_DIM,
            "graphify_version": "test", "created_at": "2026-08-18T00:00:00+00:00"}
    # Rows exist precisely so that, were the gate ever bypassed, the seam would
    # have real scores to return (on the stub embed, "Analog Mechanism" scores a
    # near-miss over "Widget" for the query "widget"). The gate must make that
    # pull unreachable.
    np.savez(corpus / "embeddings.npz",
             text_ids=np.array(["n-widget", "n-analog", "n-other"], dtype=str),
             text_vecs=rows, text_meta=json.dumps(meta))
    g = {
        "directed": True,
        "nodes": [
            {"id": "n-widget", "label": "Widget", "source_file": "widget.py",
             "source_location": "L1", "community": 0, "file_type": "concept"},
            {"id": "n-analog", "label": "Analog Mechanism", "source_file": "analog.py",
             "source_location": "L1", "community": 0, "file_type": "concept"},
            {"id": "n-other", "label": "Unrelated Leaf", "source_file": "other.py",
             "source_location": "L1", "community": 1, "file_type": "document"},
        ],
        "edges": [],
    }
    graph_file = corpus / "graph.json"
    graph_file.write_text(json.dumps(g), encoding="utf-8")
    bare = corpus.parent / "bare"
    bare.mkdir()
    (bare / "graph.json").write_text(json.dumps(g), encoding="utf-8")

    calls = []
    original = serve._query_graph_semantic_scores

    def recording(question, graph_path, **kwargs):
        calls.append((question, graph_path))
        return original(question, graph_path, **kwargs)

    monkeypatch.setattr(serve, "_query_graph_semantic_scores", recording)
    with_sidecar = serve_mod._build_server(str(graph_file))
    without_sidecar = serve_mod._build_server(str(bare / "graph.json"))
    blended = _render(with_sidecar, "widget")
    baseline = _render(without_sidecar, "widget")
    assert calls == [], (
        "query_graph at the default blend weight (0.0) must leave the semantic "
        "sidecar unread — zero pulls of the semantic scores, not a pull that "
        f"happens to stay inert: {calls}"
    )
    assert blended == baseline, (
        "query_graph at the default blend weight must render byte-identically "
        "with and without a sidecar beside the graph"
    )
    assert "Start:" in blended, "the query path must have rendered its normal header"


def test_query_graph_blend_semantic_reaches_trigram_prefilter(tmp_path):
    """When the blend gate is open, a purely semantic near-miss whose text
    shares no trigram with the query still enters the ranking, because the
    candidate superset is widened to include every node carrying a semantic
    lift. At the closed gate the prefilter remains unchanged and the near-
    miss stays out, preserving the bit-identical baseline.
    """
    import graphify.serve as serve

    # Build a 16-node graph so the trigram index is selective (thresh = 1).
    # Only n-widget's search text contains the query trigrams; n-analog is
    # excluded by the prefilter.
    G = nx.Graph()
    G.add_node("n-widget", label="Widget", source_file="widget.py",
               source_location="L1", community=0)
    G.add_node("n-analog", label="Analog Mechanism", source_file="analog.py",
               source_location="L1", community=0)
    other_nodes = [
        ("n-alpha", "Alpha", "alpha.py"),
        ("n-beta", "Beta", "beta.py"),
        ("n-charlie", "Charlie", "charlie.py"),
        ("n-delta", "Delta", "delta.py"),
        ("n-echo", "Echo", "echo.py"),
        ("n-foxtrot", "Foxtrot", "foxtrot.py"),
        ("n-golf", "Golf", "golf.py"),
        ("n-hotel", "Hotel", "hotel.py"),
        ("n-india", "India", "india.py"),
        ("n-juliet", "Juliet", "juliet.py"),
        ("n-kilo", "Kilo", "kilo.py"),
        ("n-lima", "Lima", "lima.py"),
        ("n-mike", "Mike", "mike.py"),
        ("n-november", "November", "november.py"),
    ]
    for nid, label, source in other_nodes:
        G.add_node(nid, label=label, source_file=source,
                   source_location="L1", community=0)
    G.add_edge("n-widget", "n-analog", relation="calls", confidence="EXTRACTED")

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    ianalog = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    vectors = math.sqrt(sum(c * c for c in ianalog))
    rows = np.array(
        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]] +
        [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]] * 14,
        dtype=np.float32,
    ) / vectors
    meta = {"model": "nomic-embed-text", "backend": "ollama", "dim": _STUB_DIM,
            "graphify_version": "test", "created_at": "2026-08-18T00:00:00+00:00"}
    all_ids = ["n-widget", "n-analog"] + [nid for nid, _, _ in other_nodes]
    np.savez(corpus / "embeddings.npz",
             text_ids=np.array(all_ids, dtype=str),
             text_vecs=rows, text_meta=json.dumps(meta))

    semantic = _semantics(corpus / "embeddings.npz", "widget")
    assert semantic["n-analog"] > semantic["n-widget"], "fixture must be a semantic near-miss"

    # Verify the prefilter is selective and excludes n-analog
    cands = serve._trigram_candidates(G, ["widget"])
    assert cands is not None, "prefilter must return a proper subset, not None"
    assert cands == ["n-widget"], (
        "only n-widget should pass the trigram prefilter; got %r" % cands
    )

    closed = _score_query(G, ["widget"], collect_per_term_seeds=True)
    closed_ids = [nid for _s, nid in closed.ranked]
    assert "n-widget" in closed_ids
    assert "n-analog" not in closed_ids, (
        "n-analog must be excluded from the ranking at weight 0 "
        "(no semantic union, prefilter excludes it)"
    )

    open_ = _score_query(
        G, ["widget"], collect_per_term_seeds=True,
        semantic_weight=10000, semantic_scores=semantic,
    )
    open_ids = [nid for _s, nid in open_.ranked]
    assert "n-analog" in open_ids, (
        "with the gate open, the semantic near-miss must join the ranking "
        "even though the trigram prefilter excludes it: %r" % open_ids
    )
    assert open_.ranked[0][1] == "n-widget", (
        "the exact-label node must keep rank 1 behind the open gate"
    )


def test_query_graph_blend_multiterm_exact_keeps_rank_one(tmp_path):
    """On a multi-term query, the exact-label node matches only one term, so its
    exact tier is coverage-scaled to a fraction. A rival with NO full-label
    equality but partial lexical matches (prefix + substring + source) and a
    saturated semantic lift must still rank below the exact-label node. The sort
    must be two-class: any node that fires the exact/joined tier outranks every
    node that did not, regardless of the semantic fold value.
    """
    G = nx.Graph()
    G.add_node("n-exact", label="Widget", source_file="widget.py",
               source_location="L1", community=0)
    G.add_node("n-rival", label="Widget Analog Repeater",
               source_file="widget-analog-repeater.py",
               source_location="L1", community=0)
    G.add_edge("n-exact", "n-rival", relation="calls", confidence="EXTRACTED")

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    # Rival vector aligned with the stub query embed (high semantic score);
    # exact-node vector orthogonal (semantic score ~0).
    qnorm = math.sqrt(1.0 ** 2 + 2.0 ** 2 + 0.5 ** 2)  # sqrt(5.25) ≈ 2.291
    rows = np.array(
        [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],   # exact: orthogonal
         [1.0, 2.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]],  # rival: aligned
        dtype=np.float32,
    ) / qnorm
    meta = {"model": "nomic-embed-text", "backend": "ollama", "dim": _STUB_DIM,
            "graphify_version": "test", "created_at": "2026-08-18T00:00:00+00:00"}
    np.savez(corpus / "embeddings.npz",
             text_ids=np.array(["n-exact", "n-rival"], dtype=str),
             text_vecs=rows, text_meta=json.dumps(meta))

    semantic = _semantics(corpus / "embeddings.npz", "widget repeater")
    assert semantic["n-rival"] > 0.0, "rival must carry a semantic lift"
    assert semantic["n-exact"] == 0.0, "exact node must have zero semantic score"

    open_ = _score_query(
        G, ["widget", "repeater"], collect_per_term_seeds=True,
        semantic_weight=10000, semantic_scores=semantic,
    )
    open_ids = [nid for _s, nid in open_.ranked]
    assert "n-rival" in open_ids, "rival must join the ranking via semantic lift"
    assert open_.ranked[0][1] == "n-exact", (
        "the exact-label node must keep rank 1 even when a saturated semantic rival "
        "outscores it arithmetically on a multi-term query: %r" % open_ids
    )


def test_query_graph_semantic_scores_absent_sidecar_returns_empty_dict(tmp_path):
    """When the embeddings sidecar is absent, `_query_graph_semantic_scores` must
    return an empty dict so the blend fold is an exact no-op.

    The helper resolves the sidecar path from the graph json's parent directory.
    With no sibling ``embeddings.npz``, ``search_vectors`` returns ``None``; the
    helper's ``if not rows: return {}`` short-circuit turns that into ``{}``.
    """
    import graphify.serve as serve

    graph_dir = tmp_path / "graph_dir"
    graph_dir.mkdir()
    graph_file = graph_dir / "graph.json"
    graph_file.write_text("{}", encoding="utf-8")

    # Ensure no embeddings.npz sibling exists
    assert not (graph_dir / "embeddings.npz").exists()

    result = serve._query_graph_semantic_scores("some question", str(graph_file))
    assert result == {}, (
        "absent sidecar must return an empty dict; got %r" % result
    )