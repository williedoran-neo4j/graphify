"""Serve-side tests for the R2 semantic_search tool (T13's list_tools half).

Mirrors ``tests/test_serve_http.py`` conventions: ``pytest.importorskip("mcp")``
at module top, a minimal node-link graph JSON in ``tmp_path``. Asserting the
built server's ``list_tools()`` directly keeps this fast and offline; the HTTP
round trip is already covered by ``test_serve_http.py``.
"""
from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("mcp")

from mcp.types import ListToolsRequest  # noqa: E402

from graphify import serve as serve_mod  # noqa: E402

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
