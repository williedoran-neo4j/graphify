"""R7 Neo4j/FalkorDB tier — shared prop-filter helper.

The four push sites in ``graphify/exporters/graphdb.py`` each wrote an inline
dict comprehension that kept only scalar values (``str``/``int``/``float``/
``bool``) under non-``_`` keys. Neo4j can store ``LIST<FLOAT>``, and the
embedding-vector props planned for the same push are numpy rows of int/float —
so the filter that feeds the driver must keep an all-numeric list. ``True`` is
a subclass of ``int`` in Python, so the membership test must exclude ``bool``
before it accepts ``int`` (a ``True`` member must be dropped, never coerced to
``1``). Any nested/object element (e.g. a ``dict``) must still drop the whole
list. This file is the R7 test home; the other slices live in
``test_embed_graphdb.py`` alongside this one.
"""
from __future__ import annotations

import inspect
import sys
import types

import networkx as nx

from graphify.embed import _EMBED_SPACE_BY_FILE_TYPE, _embed_space
from graphify.exporters.graphdb import (
    _pushable_props,
    push_to_falkordb,
    push_to_neo4j,
)


def test_pushable_props_float_list_survives_but_bool_first_and_dict_list_dropped():
    props = _pushable_props(
        {
            "evals": [0.717, 1.0, 2.5],          # all float — must survive
            "counts": [1, 2, 3],                 # all int — must survive
            "mixed_num": [1, 2.5, 3],            # int+float mix — must survive
            "flags": [True, False, 1],           # bool member — whole list dropped
            "records": [{"role": "doc"}],        # dict element — whole list dropped
            "label": "doc",                      # scalar str — survives
            "rank": 3,                           # scalar int — survives
            "score": 0.5,                        # scalar float — survives
            "active": True,                      # scalar bool — survives
            "_internal": "secret",               # _-prefixed key — dropped
        }
    )

    # All-numeric lists survive with their members intact.
    assert props["evals"] == [0.717, 1.0, 2.5]
    assert props["counts"] == [1, 2, 3]
    assert props["mixed_num"] == [1, 2.5, 3]

    # A bool member is not a number: dropped, never coerced to 1. A
    # list[dict] is nested/object data the driver cannot store: dropped.
    assert "flags" not in props
    assert "records" not in props

    # Scalars survive exactly as today; _-prefixed keys are filtered out.
    assert props["label"] == "doc"
    assert props["rank"] == 3
    assert props["score"] == 0.5
    assert props["active"] is True
    assert "_internal" not in props


class _RecordingNeo4jSession:
    """Recording stand-in for the neo4j driver's session (context manager)."""

    def __init__(self, log: list):
        self._log = log

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query, **params):
        self._log.append(("neo4j", query, params))


class _RecordingNeo4jDriver:
    def __init__(self, log: list):
        self._log = log

    def session(self):
        return _RecordingNeo4jSession(self._log)

    def close(self):
        pass


class _RecordingFalkorQuery:
    def __init__(self, log: list):
        self._log = log

    def query(self, cypher, params):
        self._log.append(("falkordb", cypher, params))


def _install_fake_neo4j(monkeypatch, log: list) -> None:
    """Point sys.modules['neo4j'] at a recording driver so push_to_neo4j runs
    without a live server."""
    fake = types.ModuleType("neo4j")

    class _GraphDatabase:
        @staticmethod
        def driver(uri, auth=None):
            return _RecordingNeo4jDriver(log)

    fake.GraphDatabase = _GraphDatabase
    monkeypatch.setitem(sys.modules, "neo4j", fake)


def _install_fake_falkordb(monkeypatch, log: list) -> None:
    """Point sys.modules['falkordb'] at a recording client so push_to_falkordb
    runs without a live server."""
    fake = types.ModuleType("falkordb")

    class _FalkorDB:
        def __init__(self, *args, **kwargs):
            self._log = log

        def select_graph(self, name):
            return _RecordingFalkorQuery(self._log)

    fake.FalkorDB = _FalkorDB
    monkeypatch.setitem(sys.modules, "falkordb", fake)


def test_all_four_push_sites_filter_through_shared_pushable_props(monkeypatch):
    """Both push functions must forward their node AND edge property filtering
    through the shared module-level ``_pushable_props`` helper instead of each
    re-deriving the predicate inline, and the edge merge must receive exactly
    the helper's output — never an ``id``/``community`` key spliced in next to
    the edge's own attributes.

    Real edge data is scalar-only, so the fold is invisible to real graphs.
    The fixture below therefore hangs an all-numeric list off both a node and
    an edge — the one value shape the old inline comprehension dropped and the
    shared helper keeps — so the recorded driver payloads can tell the two
    filters apart.
    """
    G = nx.Graph()
    G.add_node("n1", label="doc", file_type="document", evals=[0.5, 1.25])
    G.add_node("n2", label="target")
    G.add_edge("n1", "n2", relation="imports", weights=[2, 3])

    # Structural lock: each push function calls the helper once for its node
    # path and once for its edge path, and no inline comprehension survives.
    for fn in (push_to_neo4j, push_to_falkordb):
        src = inspect.getsource(fn)
        assert src.count("_pushable_props(") == 2, (
            f"{fn.__name__} must filter node and edge props via _pushable_props"
        )
        assert "k: v for k, v in data.items()" not in src

    expected_node = dict(_pushable_props(dict(G.nodes["n1"])))
    expected_node["id"] = "n1"
    expected_edge = _pushable_props(dict(G.edges["n1", "n2"]))
    assert "id" not in expected_edge  # edge props are the filter output alone

    neo4j_log = []
    _install_fake_neo4j(monkeypatch, neo4j_log)
    push_to_neo4j(G, uri="bolt://localhost:7687", user="neo4j", password="pw")

    neo4j_nodes = {
        params["id"]: params["props"]
        for kind, query, params in neo4j_log
        if kind == "neo4j" and "SET n +=" in query
    }
    neo4j_edges = [
        params["props"]
        for kind, query, params in neo4j_log
        if kind == "neo4j" and "SET r +=" in query
    ]
    assert neo4j_nodes["n1"] == expected_node
    assert neo4j_edges == [expected_edge]

    falkordb_log = []
    _install_fake_falkordb(monkeypatch, falkordb_log)
    push_to_falkordb(G, uri="redis://localhost:6379")

    falkordb_nodes = {
        params["id"]: params["props"]
        for kind, query, params in falkordb_log
        if kind == "falkordb" and "SET n +=" in query
    }
    falkordb_edges = [
        params["props"]
        for kind, query, params in falkordb_log
        if kind == "falkordb" and "SET r +=" in query
    ]
    assert falkordb_nodes["n1"] == expected_node
    assert falkordb_edges == [expected_edge]


def test_vector_space_of_file_types_document_absent():
    """A node's embedding space is derived from its file_type: code-family
    types land in the "code" space, document-family types in the "text" space,
    and an absent/None file_type maps to NO space — so the sidecar, the
    per-space embedding props, and the `:Embedded` label can all be derived
    from this same one-token decision.
    """
    assert _embed_space("code") == "code"
    assert _embed_space("document") == "text"
    assert _embed_space("paper") == "text"
    assert _embed_space("rationale") == "text"
    assert _embed_space("concept") == "text"

    # Non-text-family and absent/None file types never get a vector space.
    assert _embed_space("image") is None
    assert _embed_space(None) is None
    assert _embed_space("") is None

    # The per-family table drives the decision: each family member resolves
    # through it, and an absent key (image / None) resolves to no space.
    assert _EMBED_SPACE_BY_FILE_TYPE["code"] == "code"
    assert _EMBED_SPACE_BY_FILE_TYPE["document"] == "text"
    assert _EMBED_SPACE_BY_FILE_TYPE["paper"] == "text"
    assert _EMBED_SPACE_BY_FILE_TYPE["rationale"] == "text"
    assert _EMBED_SPACE_BY_FILE_TYPE["concept"] == "text"
    assert _EMBED_SPACE_BY_FILE_TYPE.get("image") is None
    assert _EMBED_SPACE_BY_FILE_TYPE.get(None) is None
