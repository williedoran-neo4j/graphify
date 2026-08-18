"""Serve-side tests for the R2 semantic_search tool (T13's list_tools half).

Mirrors ``tests/test_serve_http.py`` conventions: ``pytest.importorskip("mcp")``
at module top, a minimal node-link graph JSON in ``tmp_path``. Asserting the
built server's ``list_tools()`` directly keeps this fast and offline; the HTTP
round trip is already covered by ``test_serve_http.py``.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mcp")

from mcp.types import ListToolsRequest  # noqa: E402

from graphify import serve as serve_mod  # noqa: E402
from graphify.embed import _STUB_DIM  # noqa: E402

SAMPLE_GRAPH = {
    "directed": True,
    "nodes": [
        {"id": "a", "label": "Alpha", "community": 0},
        {"id": "b", "label": "Beta", "community": 0},
    ],
    "edges": [
        {"source": "a", "target": "b", "relation": "calls", "confidence": "EXTRACTED"},
    ],
}


def _build_tools(tmp_path) -> list:
    graph_file = tmp_path / "graph.json"
    graph_file.write_text(json.dumps(SAMPLE_GRAPH), encoding="utf-8")
    server = serve_mod._build_server(str(graph_file))
    return asyncio.run(server.request_handlers[ListToolsRequest](None)).root.tools


def test_semantic_search_list_tools_registration_and_schema(tmp_path):
    tools = _build_tools(tmp_path)
    search = [t for t in tools if t.name == "semantic_search"]
    assert search, "semantic_search tool not registered in list_tools()"
    props = search[0].inputSchema["properties"]
    # Exactly the four spec'd params (C2.1) plus the pre-existing injection's
    # project_path — nothing accidental leaking into the tool surface.
    assert set(props) == {"query", "top_k", "file_type", "min_score", "project_path"}
    assert search[0].inputSchema["required"] == ["query"]
    assert "project_path" not in search[0].inputSchema["required"]
    assert props["query"] == {"type": "string"}
    assert props["top_k"] == {"type": "integer", "default": 10}
    assert props["file_type"] == {"type": "array", "items": {"type": "string"},
                                  "description": "Optional: restrict to these file_types"}
    assert props["min_score"] == {"type": "number", "default": 0.3}


def _write_project(tmp_path: Path, prefix: str, *ids: str) -> Path:
    """Build a project dir with ``<prefix>-NN`` nodes: ``graph.json`` under
    ``graphify-out/`` (R1's layout) whose node ids are ``ids`` (no labels), and
    the sidecar ``graphify-out/embeddings.npz`` next to it — the sibling the
    handler resolves (serve.py:1924). Every vector is an identity-basis unit
    row, so under the canonical stub embed (q[1] > q[0] > q[2] > 0) score_j ==
    q[j] and the rendered ids are the top-scoring ones. Returns the project
    DIRECTORY (the value the spec's project_path must carry)."""
    out = tmp_path / "graphify-out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "graph.json").write_text(
        json.dumps({"directed": True, "nodes": [{"id": i} for i in ids], "edges": []})
    )
    meta = {
        "model": "nomic-embed-text",
        "backend": "ollama",
        "dim": _STUB_DIM,
        "graphify_version": "test",
        "created_at": "2026-08-18T00:00:00+00:00",
    }
    rows = np.zeros((len(ids), _STUB_DIM), dtype=np.float32)
    for j, _ in enumerate(ids):
        rows[j, j] = 1.0
    np.savez(
        out / "embeddings.npz",
        text_ids=np.array(list(ids), dtype=str),
        text_vecs=rows,
        text_meta=json.dumps(meta),
    )
    return tmp_path


def _semantic_search_text(
    server, query: str = "query", **arguments
) -> str:
    """Invoke semantic_search on the built server via call_tool and return the
    rendered tool text — the ACTIVE MCP shape, so project_path flows through
    call_tool's pop + _select_graph rebind (serve.py:2056-2061)."""
    from mcp.types import CallToolRequest, CallToolRequestParams

    future = server.request_handlers[CallToolRequest](
        CallToolRequest(
            params=CallToolRequestParams(name="semantic_search", arguments={"query": query, **arguments})
        )
    )
    if asyncio.iscoroutine(future):
        result = asyncio.run(future)
    else:  # newer mcp majors may return a wrapped result directly
        result = future
    return result.root.content[0].text


def test_semantic_search_project_path_routes_to_that_projects_sidecar(tmp_path):
    """C2.9 — T13's project-path half: a semantic_search call carrying
    project_path resolves sidecar candidates from THAT PROJECT's graph dir, so
    the rendered ids come from that project's embeddings.npz — and the default
    graph's sidecar is used when project_path is omitted.

    Sole-reason lock (per the spec's caution, the handler's effect is asserted
    — never the generic injection loop): the ONLY way ``p-b-01`` can render in
    the result_text is if project_path reached the handler's sidecar resolution
    (serve.py:1924 builds sidecar_path from the _select_graph-set
    active_graph_path, serve.py:2061). If call_tool dropped project_path, the
    search would run against the DEFAULT graph A's sidecar — which holds a
    DIFFERENT, overlapping-but-distinct id set (n-a-01, n-b-02, n-c-03) that
    does NOT contain ``p-b-01`` — so the "p-b-01 present" assert fires and the
    "n-a-01 absent" assert is a packet-level guard against the vacuous "found
    something" pass (n-b-02 collides across BOTH sidecars and is deliberately
    not asserted). Conversely, the no-project_path arm proves the default sidecar
    is what the omission resolves to (its ids render) while B's distinct id does
    not — the routing is a swap, not an append of both sidecars.

    Fixtures: project B = tmp_path/B with the single row p-b-01 storing
    identity-basis e0; default graph A = project-like tmp_path/A/GRAPHIFY_OUT
    with rows n-a-01 -> e0, n-b-02 -> e1, n-c-03 -> e2. Before the B call,
    call_tool re-arms the default context (serve.py:2061) so A is loaded
    regardless of default-path type or cache state.
    """
    default_A = _write_project(tmp_path / "A", "n", "n-a-01", "n-b-02", "n-c-03")
    project_B = _write_project(tmp_path / "B", "p", "p-b-01")
    server = serve_mod._build_server(str(default_A / "graphify-out" / "graph.json"))

    # The default-context call re-arms A, then the default arm asserts A's ids.
    no_path_text = _semantic_search_text(server)
    assert "n-b-02" in no_path_text and "n-a-01" in no_path_text, (
        "omitting project_path must search the server default graph's sidecar; "
        f"got {no_path_text!r}"
    )
    assert "n-c-03" in no_path_text, (
        "the default sidecar's third row must render too; got {no_path_text!r}"
    )
    assert "p-b-01" not in no_path_text, (
        "project B's sidecar must NOT leak into a default-graph search; "
        f"got {no_path_text!r}"
    )

    # The routed call resolves B's sidecar — p-b-01 (B's distinct id) renders...
    routed_text = _semantic_search_text(server, project_path=str(project_B))
    assert "p-b-01" in routed_text, (
        "project_path must route the search to THAT project's sidecar — its "
        f"distinct id must render; got {routed_text!r}"
    )
    assert "[text]" in routed_text, (
        f"routed search must render a real ranked row, got {routed_text!r}"
    )
    # ...and the default sidecar's id set does not (routing is a swap, not a merge).
    assert "n-a-01" not in routed_text, (
        "the routed search must use project B's sidecar, NOT the default graph "
        f"A's — an id unique to A must be absent; got {routed_text!r}"
    )
