"""C1.2 — enrich_embeddings writes the text-space ``embeddings.npz`` sidecar.

Contract C3 (thread form): the sidecar is space-keyed and holds exactly the
``text_*`` group — ``text_ids`` (unicode node ids, sorted id order), ``text_vecs``
(float32, ``(N_t, D)``), ``text_meta`` (json string). ``image`` nodes are
excluded (the enrich half of T3). ``dim`` in the meta must equal
``text_vecs.shape[1]`` (invariant I5).

The graph fixture is a real ``networkx.Graph`` whose per-node attrs dicts carry
``id``/``label``/``source_file``/``file_type``, matching the shape
``graphify.build.build_from_json`` produces and what ``build_node_text`` reads.
"""
from __future__ import annotations

import json

import networkx as nx
import numpy as np

from graphify.embed import _STUB_DIM, enrich_embeddings


def _small_graph() -> nx.Graph:
    """Two text-family nodes plus one image node (excluded from the text space)."""
    g = nx.Graph()
    g.add_nodes_from(
        [
            (
                "n-IMG-07",
                {
                    "id": "n-IMG-07",
                    "label": "architecture diagram",
                    "source_file": "assets/arch.png",
                    "file_type": "image",
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
            (
                "n-a-01",
                {
                    "id": "n-a-01",
                    "label": "API Tokens",
                    "source_file": "docs/security.md",
                    "file_type": "concept",
                },
            ),
        ]
    )
    return g


def test_enrich_writes_sidecar_keys_and_dtype(tmp_path):
    """T4 — ``enrich_embeddings`` writes sibling ``embeddings.npz`` exactly for
    the text family: keys are exactly ``text_ids``/``text_vecs``/``text_meta``,
    vectors are float32, and the image node's id is absent from ``text_ids``
    (the enrich half of T3)."""
    out = enrich_embeddings(_small_graph(), tmp_path / "graph.json")

    npz_path = tmp_path / "embeddings.npz"
    assert out == npz_path
    assert npz_path.is_file()

    with np.load(npz_path) as data:
        assert set(data.files) == {"text_ids", "text_vecs", "text_meta"}
        assert data["text_vecs"].dtype == np.float32
        ids = [str(s) for s in data["text_ids"]]
        assert ids == ["n-a-01", "n-b-02"]
        assert "n-IMG-07" not in ids


def test_sidecar_meta_fields(tmp_path):
    """T6 — ``text_meta`` parses to a JSON object carrying model, backend, dim,
    graphify_version and created_at; ``dim`` matches ``text_vecs.shape[1]``
    (I5), proving the writer records the dimension it actually stored."""
    enrich_embeddings(_small_graph(), tmp_path / "graph.json")

    with np.load(tmp_path / "embeddings.npz") as data:
        meta = json.loads(str(data["text_meta"]))
        assert meta["model"] == "nomic-embed-text"
        assert meta["backend"] == "ollama"
        assert meta["dim"] == data["text_vecs"].shape[1]
        assert isinstance(meta["graphify_version"], str) and meta["graphify_version"]
        assert isinstance(meta["created_at"], str) and meta["created_at"]


def test_normalize_on_write(tmp_path, monkeypatch):
    """T5 — every stored ``text_vecs`` row is L2-normalized (‖v‖ == 1.0),
    regardless of what the embedding seam returns. The seam is faked to emit
    norm-5.0 vectors so the writer MUST divide magnitudes down to 1.0."""
    vecs = [np.full(_STUB_DIM, 5.0 / np.sqrt(_STUB_DIM))] * 2
    monkeypatch.setattr(
        "graphify.embed._call_embeddings",
        lambda backend, model, inputs: vecs,
    )

    enrich_embeddings(_small_graph(), tmp_path / "graph.json")

    with np.load(tmp_path / "embeddings.npz") as data:
        norms = np.linalg.norm(data["text_vecs"], axis=1)
        assert norms.shape == (2,)
        assert np.allclose(norms, 1.0, atol=1e-6)
