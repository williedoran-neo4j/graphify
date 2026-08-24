"""CLI tests for the standalone `graphify embed` command."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import pytest
from networkx.readwrite import json_graph

import graphify.__main__ as mainmod


def _write_graph(tmp_path, nodes):
    G = nx.Graph()
    for nid, attrs in nodes:
        G.add_node(nid, **attrs)
    graph_path = tmp_path / "graphify-out" / "graph.json"
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path.write_text(json.dumps(json_graph.node_link_data(G, edges="links")))
    return graph_path


def test_embed_cli_success(monkeypatch, tmp_path, capsys):
    """`graphify embed .` loads an existing graph.json, calls enrich_embeddings,
    writes the embeddings.npz sidecar, and prints a success message."""
    nodes = [
        (
            "n-a-01",
            {
                "id": "n-a-01",
                "label": "API Tokens",
                "source_file": "docs/security.md",
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
    graph_path = _write_graph(tmp_path, nodes)
    monkeypatch.setattr(mainmod, "_check_skill_version", lambda _: None)
    monkeypatch.setattr(
        "graphify.embed._call_embeddings",
        lambda backend, model, inputs: [[1.0] * 768] * len(inputs),
    )
    monkeypatch.setattr(mainmod.sys, "argv", ["graphify", "embed", str(tmp_path)])
    monkeypatch.setenv("GRAPHIFY_OUT", str(graph_path.parent.relative_to(tmp_path)))
    mainmod.main()
    out = capsys.readouterr().out
    assert graph_path.with_name("embeddings.npz").is_file()
    assert "Embedded" in out or "embed" in out.lower()


def test_embed_cli_zero_node_no_op(monkeypatch, tmp_path, capsys):
    """A zero-node graph prints a clear no-op message and exits 0 without
    calling the embedding backend."""
    graph_path = _write_graph(tmp_path, [])
    monkeypatch.setattr(mainmod, "_check_skill_version", lambda _: None)
    backend_called = []
    monkeypatch.setattr(
        "graphify.embed._call_embeddings",
        lambda backend, model, inputs: (backend_called.append(True) or [[1.0] * 768] * len(inputs)),
    )
    monkeypatch.setattr(mainmod.sys, "argv", ["graphify", "embed", str(tmp_path)])
    monkeypatch.setenv("GRAPHIFY_OUT", str(graph_path.parent.relative_to(tmp_path)))
    with pytest.raises(SystemExit) as exc_info:
        mainmod.main()
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert not backend_called
    assert "no-op" in out.lower() or "no nodes" in out.lower() or "empty" in out.lower() or "nothing" in out.lower()


def test_embed_in_help_text(capsys, monkeypatch):
    """`graphify --help` must list the `embed` command with its description."""
    monkeypatch.setattr(mainmod, "_check_skill_version", lambda _: None)
    monkeypatch.setattr(mainmod.sys, "argv", ["graphify", "--help"])
    mainmod.main()
    out = capsys.readouterr().out
    assert "embed <path>" in out, "`embed <path>` command line missing from help text"
    assert "re-embed" in out, "`re-embed` description missing from embed help line"


def test_embed_help_guard_redirects(monkeypatch, tmp_path, capsys):
    """`graphify embed --help` hits the universal help guard because `embed`
    is not in `_FREE_TEXT_CMDS`, prints the redirect message, and does not
    run the embed command or write any sidecar files."""
    nodes = [
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
    graph_path = _write_graph(tmp_path, nodes)
    monkeypatch.setattr(mainmod, "_check_skill_version", lambda _: None)
    monkeypatch.setattr(
        "graphify.embed._call_embeddings",
        lambda backend, model, inputs: [[1.0] * 768] * len(inputs),
    )
    monkeypatch.setattr(mainmod.sys, "argv", ["graphify", "embed", str(tmp_path), "--help"])
    monkeypatch.setenv("GRAPHIFY_OUT", str(graph_path.parent.relative_to(tmp_path)))
    mainmod.main()
    out = capsys.readouterr().out
    assert "Run 'graphify --help' for full usage." in out
    assert not graph_path.with_name("embeddings.npz").is_file()
    assert "Embedded" not in out
