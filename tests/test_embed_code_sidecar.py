"""The code_* sidecar group written by enrich_embeddings.

The second embedding space (nomic-embed-code) is written alongside the existing
text_* group in the same embeddings.npz: ``code_ids`` / ``code_vecs`` /
``code_meta``. ``code_ids`` holds exactly the nodes whose file_type maps to the
"code" space (sorted by nid), the rows are float32 and L2-normalized like the
text group, meta carries the code model/backend/dim, and each node's constructed
text reaches the ``_call_embeddings`` seam under its own space's model — the
code texts including the source-content block from the root-aware code-text
path. On the two-model cold cache the seam is called exactly once per namespace.
"""
from __future__ import annotations

import json

import networkx as nx
import numpy as np

from graphify.embed import build_node_text, enrich_embeddings

_DIM = 8


def _code_text_graph(tmp_path) -> nx.Graph:
    """Two code nodes (each with a real source file under tmp_path) plus one
    document and one concept node, inserted in NON-sorted order so a writer
    that skips the id sort is visibly wrong."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "parser.py").write_text(
        "def parse_token(text: str):\n    return text.split()\n", encoding="utf-8"
    )
    (src / "graph_build.py").write_text(
        "def build_index(graph):\n    return sorted(graph)\n", encoding="utf-8"
    )
    g = nx.Graph()
    g.add_nodes_from(
        [
            (
                "n-c-03",
                {
                    "id": "n-c-03",
                    "label": "IndexBuilder",
                    "source_file": "src/graph_build.py",
                    "file_type": "code",
                },
            ),
            (
                "n-a-01",
                {
                    "id": "n-a-01",
                    "label": "TokenParser",
                    "source_file": "src/parser.py",
                    "file_type": "code",
                },
            ),
            (
                "n-d-04",
                {
                    "id": "n-d-04",
                    "label": "Graph Index",
                    "source_file": "docs/index.md",
                    "file_type": "concept",
                },
            ),
            (
                "n-b-02",
                {
                    "id": "n-b-02",
                    "label": "Embedding math",
                    "source_file": "docs/embeddings.md",
                    "file_type": "document",
                },
            ),
        ]
    )
    return g


def test_enrich_writes_code_group_alongside_text_group(tmp_path, monkeypatch):
    """enrich_embeddings writes code_ids/code_vecs/code_meta next to text_*,
    and routes each node's text to the seam under its own space's model."""
    g = _code_text_graph(tmp_path)
    calls: list[tuple[str, str, list[str]]] = []

    def recording_seam(backend, model, inputs):
        calls.append((backend, model, list(inputs)))
        return [[0.25] * _DIM for _ in inputs]

    monkeypatch.setattr("graphify.embed._call_embeddings", recording_seam)

    out = enrich_embeddings(g, tmp_path / "graph.json")

    with np.load(out) as data:
        assert {"text_ids", "text_vecs", "text_meta",
                "code_ids", "code_vecs", "code_meta"} <= set(data.files)

        # code_ids: exactly the two code nodes, sorted by nid (nodes were
        # inserted in the reverse of that order).
        code_ids = [str(s) for s in data["code_ids"]]
        assert code_ids == ["n-a-01", "n-c-03"]

        # code_vecs: float32, (2, D), every row L2-normalized (I3).
        assert data["code_vecs"].dtype == np.float32
        assert data["code_vecs"].shape == (2, _DIM)
        norms = np.linalg.norm(data["code_vecs"], axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

        # code_meta names the code model and stores the real stored dim.
        code_meta = json.loads(str(data["code_meta"]))
        assert code_meta["model"] == "nomic-embed-code"
        assert code_meta["backend"] == "ollama"
        assert code_meta["dim"] == data["code_vecs"].shape[1]

        # Partition: the code nodes are OUT of the text group and the text
        # nodes are out of the code group (no double-write, no drop).
        text_ids = [str(s) for s in data["text_ids"]]
        assert text_ids == ["n-b-02", "n-d-04"]

    # On the cold cache each (backend, model) namespace calls the seam once.
    code_calls = [c for c in calls if c[1] == "nomic-embed-code"]
    text_calls = [c for c in calls if c[1] == "nomic-embed-text"]
    assert len(code_calls) == 1
    assert len(text_calls) == 1

    # The code-model inputs are the C1 code-text path's output (source-content
    # block included, since root is wired through) and never a text-family
    # text; the text-model inputs are exactly the two non-code texts.
    code_inputs = code_calls[0][2]
    text_inputs = text_calls[0][2]
    for nid in ("n-a-01", "n-c-03"):
        built = build_node_text(g, nid, g.nodes[nid], tmp_path)
        assert built in code_inputs
        assert built not in text_inputs
    for nid in ("n-b-02", "n-d-04"):
        built = build_node_text(g, nid, g.nodes[nid], tmp_path)
        assert built in text_inputs
        assert built not in code_inputs
