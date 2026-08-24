"""Tests for `graphify extract --embed` flag."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import graphify.__main__ as mainmod


def _code_corpus(tmp_path: Path) -> Path:
    """Minimal code-only corpus so no LLM key is needed."""
    (tmp_path / "app.py").write_text("def hello():\n    return 1\n")
    return tmp_path


def _fake_enrich_embeddings(graph, graph_path, root=None):
    """Stub that writes a minimal embeddings.npz sidecar next to graph_path."""
    npz_path = Path(graph_path).parent / "embeddings.npz"
    np.savez(
        npz_path,
        text_ids=np.array(["n1"], dtype=str),
        text_vecs=np.array([[1.0, 2.0]], dtype=np.float32),
        text_meta="{}",
    )


def test_extract_embed_flag_writes_sidecar(monkeypatch, tmp_path):
    """`graphify extract <path> --embed` writes an embeddings.npz sidecar
    next to graph.json after the graph is written."""
    corpus = _code_corpus(tmp_path)
    out_dir = tmp_path / "out"

    monkeypatch.setattr(mainmod, "_check_skill_version", lambda _: None)
    monkeypatch.setattr("graphify.embed.enrich_embeddings", _fake_enrich_embeddings)
    monkeypatch.setattr(
        mainmod.sys,
        "argv",
        [
            "graphify",
            "extract",
            str(corpus),
            "--code-only",
            "--no-cluster",
            "--embed",
            "--out",
            str(out_dir),
        ],
    )

    try:
        mainmod.main()
    except SystemExit as exc:
        assert exc.code in (None, 0), f"unexpected exit code {exc.code}"

    graph_json = out_dir / "graphify-out" / "graph.json"
    embeddings_npz = out_dir / "graphify-out" / "embeddings.npz"

    assert graph_json.exists(), "graph.json must be written"
    assert embeddings_npz.exists(), (
        "embeddings.npz sidecar must be written when --embed is passed"
    )

    # Structural sanity: the stub wrote recognisable arrays.
    data = np.load(embeddings_npz)
    assert "text_ids" in data
    assert len(data["text_ids"]) > 0
