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
import re
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
    without a live server. The fake also installs a minimal ``neo4j.exceptions``
    module so a push that guards vector-index creation can import the driver's
    exception types."""
    fake = types.ModuleType("neo4j")
    exceptions_mod = types.ModuleType("neo4j.exceptions")

    class _ClientError(Exception):
        pass

    exceptions_mod.ClientError = _ClientError
    monkeypatch.setitem(sys.modules, "neo4j.exceptions", exceptions_mod)
    fake.exceptions = exceptions_mod

    class _GraphDatabase:
        @staticmethod
        def driver(uri, auth=None):
            return _RecordingNeo4jDriver(log)

    fake.GraphDatabase = _GraphDatabase
    monkeypatch.setitem(sys.modules, "neo4j", fake)


def _install_fake_neo4j_ddl_failure(monkeypatch, log: list):
    """Point sys.modules['neo4j'] at a recording driver whose session RAISES on
    every CREATE VECTOR INDEX statement (after recording the query), so a test
    can prove an index-creation failure never fails the push."""
    fake = types.ModuleType("neo4j")
    exceptions_mod = types.ModuleType("neo4j.exceptions")

    class _ClientError(Exception):
        pass

    exceptions_mod.ClientError = _ClientError
    monkeypatch.setitem(sys.modules, "neo4j.exceptions", exceptions_mod)
    fake.exceptions = exceptions_mod

    class _FailSession(_RecordingNeo4jSession):
        def run(self, query, **params):
            super().run(query, **params)
            if "CREATE VECTOR INDEX" in query:
                raise _ClientError("vector index not supported (simulated)")

    class _FailDriver(_RecordingNeo4jDriver):
        def session(self):
            return _FailSession(self._log)

    class _GraphDatabase:
        @staticmethod
        def driver(uri, auth=None):
            return _FailDriver(log)

    fake.GraphDatabase = _GraphDatabase
    monkeypatch.setitem(sys.modules, "neo4j", fake)
    return _ClientError


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


def test_neo4j_embedding_props_L2_EQ_sidecar(monkeypatch, tmp_path):
    """Pushing a graph with an ``embeddings_path`` sidecar must land the
    sidecar's vector rows on the merged node props, split into per-space
    ``embedding_text``/``embedding_code`` keys by the node's file_type.

    The pushed array must equal the sidecar's own row for the same id
    (both come from one ``enrich_embeddings`` writer, so they cannot drift).
    Only nodes whose id is actually in the sidecar get an embedding prop, and
    a node with no file_type gets none even when its id is present.
    """
    import numpy as np

    G = nx.Graph()
    G.add_node("doc1", label="report", file_type="document")
    G.add_node("code1", label="module.py", file_type="code")
    G.add_node("no_vec", label="dashboard", file_type="concept")
    G.add_node("no_type", label="untyped")
    G.add_edge("doc1", "code1", relation="references")

    sidecar = tmp_path / "embeddings.npz"
    np.savez(
        sidecar,
        text_ids=np.array(["doc1", "code1"]),
        text_vecs=np.array(
            [[0.5, -0.5, 1.0, 0.0], [0.0, 1.0, 0.5, -1.0]], dtype=np.float32
        ),
        text_meta=np.str_('{"dim": 4}'),
    )

    log = []
    _install_fake_neo4j(monkeypatch, log)
    push_to_neo4j(
        G,
        uri="bolt://localhost:7687",
        user="neo4j",
        password="pw",
        embeddings_path=sidecar,
    )

    pushed = {
        params["id"]: params["props"]
        for kind, query, params in log
        if kind == "neo4j" and "SET n +=" in query
    }
    doc_row = np.array([0.5, -0.5, 1.0, 0.0], dtype=np.float32)
    code_row = np.array([0.0, 1.0, 0.5, -1.0], dtype=np.float32)
    # The driver payload is a plain list of floats, so it round-trips through
    # the shared prop filter instead of being dropped as a numpy array.
    assert np.allclose(pushed["doc1"]["embedding_text"], doc_row)
    assert np.allclose(pushed["code1"]["embedding_code"], code_row)
    assert pushed["doc1"]["embedding_text"] == doc_row.tolist()
    assert pushed["code1"]["embedding_code"] == code_row.tolist()
    assert "embedding_text" not in pushed["no_vec"]
    assert "embedding_code" not in pushed["no_vec"]
    assert "embedding_text" not in pushed["no_type"]
    assert "embedding_code" not in pushed["no_type"]

    # I1: without a sidecar the push keeps today's exact behavior — no
    # embedding_* prop anywhere, code or text.
    no_sidecar_log = []
    _install_fake_neo4j(monkeypatch, no_sidecar_log)
    push_to_neo4j(
        G, uri="bolt://localhost:7687", user="neo4j", password="pw"
    )
    no_sidecar_props = {
        params["id"]: params["props"]
        for kind, query, params in no_sidecar_log
        if kind == "neo4j" and "SET n +=" in query
    }
    for pid, props in no_sidecar_props.items():
        assert not [k for k in props if k.startswith("embedding_")], (
            f"node {pid} must carry no embedding prop without a sidecar"
        )


def test_falkordb_embedding_props_L2_EQ_sidecar(monkeypatch, tmp_path):
    """The FalkorDB push must splice sidecar vectors exactly like the Neo4j
    push: a node whose id is in the sidecar gets its row under
    ``embedding_<space>`` (text for document-family, code for code), and a
    node whose id is absent -- or which has no file_type at all -- gets no
    embedding prop. The pushed values must equal the sidecar's own row, so a
    delete of the FalkorDB splice site (graphdb.py's call to
    ``_embedding_prop``) would fail this test."""
    import numpy as np

    G = nx.Graph()
    G.add_node("doc1", label="report", file_type="document")
    G.add_node("code1", label="module.py", file_type="code")
    G.add_node("no_vec", label="dashboard", file_type="concept")
    G.add_node("no_type", label="untyped")
    G.add_edge("doc1", "code1", relation="references")

    sidecar = tmp_path / "embeddings.npz"
    np.savez(
        sidecar,
        text_ids=np.array(["doc1", "code1"]),
        text_vecs=np.array(
            [[0.5, -0.5, 1.0, 0.0], [0.0, 1.0, 0.5, -1.0]], dtype=np.float32
        ),
        text_meta=np.str_('{"dim": 4}'),
    )

    log = []
    _install_fake_falkordb(monkeypatch, log)
    push_to_falkordb(
        G, uri="redis://localhost:6379", embeddings_path=sidecar
    )

    pushed = {
        params["id"]: params["props"]
        for kind, query, params in log
        if kind == "falkordb" and "SET n +=" in query
    }
    doc_row = np.array([0.5, -0.5, 1.0, 0.0], dtype=np.float32)
    code_row = np.array([0.0, 1.0, 0.5, -1.0], dtype=np.float32)
    assert np.allclose(pushed["doc1"]["embedding_text"], doc_row)
    assert np.allclose(pushed["code1"]["embedding_code"], code_row)
    assert pushed["doc1"]["embedding_text"] == doc_row.tolist()
    assert pushed["code1"]["embedding_code"] == code_row.tolist()
    assert "embedding_text" not in pushed["no_vec"]
    assert "embedding_code" not in pushed["no_vec"]
    assert "embedding_text" not in pushed["no_type"]
    assert "embedding_code" not in pushed["no_type"]

    no_sidecar_log = []
    _install_fake_falkordb(monkeypatch, no_sidecar_log)
    push_to_falkordb(G, uri="redis://localhost:6379")
    no_sidecar_props = {
        params["id"]: params["props"]
        for kind, query, params in no_sidecar_log
        if kind == "falkordb" and "SET n +=" in query
    }
    for pid, props in no_sidecar_props.items():
        assert not [k for k in props if k.startswith("embedding_")], (
            f"node {pid} must carry no embedding prop without a sidecar"
        )


def _integer_tokens(query: str) -> set[int]:
    """All whole-number tokens in a query (grammar-agnostic)."""
    return {int(m) for m in re.findall(r"\d+", query)}


def test_vector_index_emitted_once_per_space_failure_non_fatal(
    monkeypatch, tmp_path
):
    """A push with a sidecar must (a) mark each vector-carrying node with an
    :Embedded label, (b) emit exactly one CREATE VECTOR INDEX ... IF NOT EXISTS
    per embedding space, each reading its dimension from the sidecar meta, and
    (c) never fail the push when an index creation errors -- the node MERGEs
    must be recorded before any index DDL is attempted."""
    import numpy as np

    G = nx.Graph()
    G.add_node("doc1", label="report", file_type="document")
    G.add_node("code1", label="module.py", file_type="code")
    G.add_node("no_vec", label="dashboard", file_type="concept")
    G.add_edge("doc1", "code1", relation="references")

    sidecar = tmp_path / "embeddings.npz"
    np.savez(
        sidecar,
        text_ids=np.array(["doc1", "code1"]),
        text_vecs=np.array(
            [[0.5, -0.5, 1.0, 0.0], [0.0, 1.0, 0.5, -1.0]], dtype=np.float32
        ),
        text_meta=np.str_('{"dim": 4}'),
    )

    # Labels are carried on the same MERGE statements as the props the driver
    # will execute; the node-MERGE query for a vector-carrying id must say
    # SET n:Embedded, the absent-id node must not.
    log = []
    _install_fake_neo4j(monkeypatch, log)
    counts = push_to_neo4j(
        G,
        uri="bolt://localhost:7687",
        user="neo4j",
        password="pw",
        embeddings_path=sidecar,
    )
    push_log = [(q, p) for kind, q, p in log if kind == "neo4j"]

    label_queries = {
        p["id"]: q
        for q, p in push_log
        if "SET n +=" in q and "SET n:Embedded" in q
    }
    assert "doc1" in label_queries
    assert "code1" in label_queries

    node_queries = {
        p["id"]: q for q, p in push_log if "SET n +=" in q
    }
    assert "no_vec" in node_queries
    assert "SET n:Embedded" not in node_queries["no_vec"]

    # Exactly two index statements, one per space, each once, idempotent and
    # dimensioned from the sidecar meta.
    ddl = [q for q, _ in push_log if "CREATE VECTOR INDEX" in q]
    assert ddl == [q for q, _ in push_log if "IF NOT EXISTS" in q]
    assert len([q for q in ddl if "embedding_text" in q]) == 1
    assert len([q for q in ddl if "embedding_code" in q]) == 1
    text_ddl = next(q for q in ddl if "embedding_text" in q)
    code_ddl = next(q for q in ddl if "embedding_code" in q)
    assert 4 in _integer_tokens(text_ddl)
    assert 4 in _integer_tokens(code_ddl)
    assert "FOR (n:Embedded)" in text_ddl
    assert "FOR (n:Embedded)" in code_ddl

    # Non-fatal ordering: the DDL runs after the node MERGEs, and a DDL
    # failure must not abort the push (counts still returned).
    first_ddl_idx = min(
        next(i for i, (qq, _) in enumerate(push_log) if qq == q) for q in ddl
    )
    last_node_merge_idx = max(
        i for i, (qq, _) in enumerate(push_log) if "SET n +=" in qq
    )
    assert first_ddl_idx > last_node_merge_idx

    fail_log = []
    _install_fake_neo4j_ddl_failure(monkeypatch, fail_log)
    fail_counts = push_to_neo4j(
        G,
        uri="bolt://localhost:7687",
        user="neo4j",
        password="pw",
        embeddings_path=sidecar,
    )
    fail_push = [(q, p) for kind, q, p in fail_log if kind == "neo4j"]
    fail_node_queries = {p["id"]: q for q, p in fail_push if "SET n +=" in q}
    fail_node_labels = {
        pid for pid, q in fail_node_queries.items() if "SET n:Embedded" in q
    }
    assert "doc1" in fail_node_labels and "code1" in fail_node_labels
    # The node pushes already completed (and the labels landed); the index
    # failures were swallowed rather than aborting the push.
    assert fail_counts == counts

    # Without a sidecar, neither the label nor any index statement appears.
    no_sidecar_log = []
    _install_fake_neo4j(monkeypatch, no_sidecar_log)
    push_to_neo4j(G, uri="bolt://localhost:7687", user="neo4j", password="pw")
    no_sidecar_push = [(q, p) for kind, q, p in no_sidecar_log if kind == "neo4j"]
    assert not [q for q, _ in no_sidecar_push if "CREATE VECTOR INDEX" in q]
    assert not [q for q, _ in no_sidecar_push if "SET n:Embedded" in q]


def test_embedding_code_reads_code_group_row_not_text_row(monkeypatch, tmp_path):
    """A code node's ``embedding_code`` must carry the sidecar's ``code_vecs``
    row for its id -- never the ``text_vecs`` row -- while a text-family node's
    ``embedding_text`` keeps its ``text_vecs`` row.

    The exporter builds every sidecar entry from the ``text_*`` group alone and
    reuses that one row for both the ``text`` and ``code`` slots, so today a
    code node's ``embedding_code`` silently carries the text-space vector. This
    fixture therefore writes the code node into BOTH groups with DIFFERENT rows:
    the pushed ``embedding_code`` must equal the group's own row, not the text
    row the old build splices in. The code node must also carry no
    ``embedding_text`` (the per-space split), and each space's vector index
    must be dimensioned by its own group. The no-sidecar push and a
    pre-code-group (``text_*``-only) sidecar must keep pushing exactly as they
    do today.
    """
    import numpy as np

    G = nx.Graph()
    G.add_node("doc1", label="report", file_type="document")
    G.add_node("code1", label="module.py", file_type="code")
    G.add_edge("doc1", "code1", relation="references")

    doc_text_row = np.array([0.5, -0.5, 1.0, 0.0], dtype=np.float32)
    code_text_row = np.array([1.0, 1.0, -1.0, 0.5], dtype=np.float32)
    # The code group is a DIFFERENT model with a DIFFERENT width (3 vs the
    # text group's 4): the code node's pushed vector must be this 3-wide row --
    # never the 4-wide text row -- and the code-space index must be dimensioned
    # 3, not the text group's 4.
    code_code_row = np.array([0.3, -0.2, 0.7], dtype=np.float32)
    # The two rows that could populate the code slot must genuinely differ;
    # a matching-vector fixture could not tell the two read paths apart.
    # (Different widths here: the text row is 4-wide, the code row 3-wide.)
    assert code_text_row.shape != code_code_row.shape

    sidecar = tmp_path / "embeddings.npz"
    np.savez(
        sidecar,
        text_ids=np.array(["doc1", "code1"]),
        text_vecs=np.array([doc_text_row, code_text_row], dtype=np.float32),
        text_meta=np.str_('{"dim": 4}'),
        code_ids=np.array(["code1"]),
        code_vecs=np.array([code_code_row], dtype=np.float32),
        code_meta=np.str_('{"dim": 3}'),
    )

    log = []
    _install_fake_neo4j(monkeypatch, log)
    push_to_neo4j(
        G,
        uri="bolt://localhost:7687",
        user="neo4j",
        password="pw",
        embeddings_path=sidecar,
    )
    pushed = {
        params["id"]: params["props"]
        for kind, query, params in log
        if kind == "neo4j" and "SET n +=" in query
    }

    # The code node's embedding_code is the code group's own row -- never the
    # text row the old text-only build spliced into the code slot. (The text
    # row is 4-wide, the code row 3-wide, so the lists differ on length too.)
    assert pushed["code1"]["embedding_code"] == code_code_row.tolist()
    assert pushed["code1"]["embedding_code"] != code_text_row.tolist()
    # Per-space split: a code node carries no text-space prop.
    assert "embedding_text" not in pushed["code1"]

    # The text-family node keeps its text group's row, and only that.
    assert pushed["doc1"]["embedding_text"] == doc_text_row.tolist()
    assert "embedding_code" not in pushed["doc1"]

    # Each space's vector index is dimensioned by that space's own group: the
    # code group's 3-wide rows must not be indexed under the text group's
    # 4-dim setting (a shared dim would silently misdimension one search).
    ddl = [
        query
        for kind, query, params in log
        if kind == "neo4j" and "CREATE VECTOR INDEX" in query
    ]
    text_ddl = next(q for q in ddl if "embedding_text" in q)
    code_ddl = next(q for q in ddl if "embedding_code" in q)
    assert _integer_tokens(text_ddl) == {4}
    assert _integer_tokens(code_ddl) == {3}

    # No sidecar: today's exact behavior -- no embedding prop anywhere.
    no_sidecar_log = []
    _install_fake_neo4j(monkeypatch, no_sidecar_log)
    push_to_neo4j(G, uri="bolt://localhost:7687", user="neo4j", password="pw")
    no_sidecar_props = {
        params["id"]: params["props"]
        for kind, query, params in no_sidecar_log
        if kind == "neo4j" and "SET n +=" in query
    }
    for pid, props in no_sidecar_props.items():
        assert not [k for k in props if k.startswith("embedding_")], (
            f"node {pid} must carry no embedding prop without a sidecar"
        )

    # A text_*-only sidecar (pre-code-group shape): the code slot falls back to
    # the text row exactly as it has always been pushed -- no regression.
    legacy = tmp_path / "legacy.npz"
    np.savez(
        legacy,
        text_ids=np.array(["doc1", "code1"]),
        text_vecs=np.array([doc_text_row, code_text_row], dtype=np.float32),
        text_meta=np.str_('{"dim": 4}'),
    )
    legacy_log = []
    _install_fake_neo4j(monkeypatch, legacy_log)
    push_to_neo4j(
        G,
        uri="bolt://localhost:7687",
        user="neo4j",
        password="pw",
        embeddings_path=legacy,
    )
    legacy_pushed = {
        params["id"]: params["props"]
        for kind, query, params in legacy_log
        if kind == "neo4j" and "SET n +=" in query
    }
    assert legacy_pushed["code1"]["embedding_code"] == code_text_row.tolist()
    assert legacy_pushed["doc1"]["embedding_text"] == doc_text_row.tolist()


def test_k8s_node_merge_label_is_K8s(monkeypatch):
    """A graph node with file_type='k8s' must produce a MERGE query whose label
    token is exactly ``K8s`` — capitalized, not ``Entity``, not lowercase ``k8s``,
    and not stripped by the sanitizer.
    """
    G = nx.Graph()
    raw_id = "k8s://payments/Deployment/api-server"
    G.add_node(raw_id, label="api-server", file_type="k8s", source_file="payments/deployment.yaml")

    log = []
    _install_fake_neo4j(monkeypatch, log)
    push_to_neo4j(G, uri="bolt://localhost:7687", user="neo4j", password="pw")

    merges = [
        (query, params)
        for kind, query, params in log
        if kind == "neo4j" and "SET n +=" in query
    ]
    assert len(merges) == 1
    query, params = merges[0]
    assert params["id"] == raw_id
    # The label token must be exactly K8s, never Entity or lowercase k8s.
    assert "MERGE (n:K8s {id: $id})" in query


def test_neo4j_push_code_only_sidecar_absent_text_group(monkeypatch, tmp_path):
    """A code-only first-run sidecar -- ``code_ids``/``code_vecs``/``code_meta``
    present, NO ``text_*`` group -- must still push cleanly. The embedding
    index and per-space dims are built from the groups that exist, so the code
    node's ``embedding_code`` carries its own ``code_vecs`` row, exactly the
    code-space vector index is emitted dimensioned from ``code_meta["dim"]``,
    and no text-space index statement appears. A text-family node cannot carry
    ``embedding_text`` (the sidecar holds no text rows), and the absent group
    must never raise ``KeyError``.
    """
    import numpy as np

    G = nx.Graph()
    G.add_node("code1", label="module.py", file_type="code")
    G.add_node("doc1", label="report", file_type="document")
    G.add_edge("code1", "doc1", relation="references")

    code_code_row = np.array([0.3, -0.2, 0.7], dtype=np.float32)
    sidecar = tmp_path / "embeddings.npz"
    np.savez(
        sidecar,
        code_ids=np.array(["code1"]),
        code_vecs=np.array([code_code_row], dtype=np.float32),
        code_meta=np.str_('{"dim": 3}'),
    )
    # The fixture shapes of the two groups are irrelevant here -- a code-only
    # sidecar has no text group at all, so no text row can leak anywhere.

    log = []
    _install_fake_neo4j(monkeypatch, log)
    counts = push_to_neo4j(
        G,
        uri="bolt://localhost:7687",
        user="neo4j",
        password="pw",
        embeddings_path=sidecar,
    )

    # The push completes -- the absent text group never raises KeyError.
    assert counts == {"nodes": 2, "edges": 1}

    push_log = [(q, p) for kind, q, p in log if kind == "neo4j"]
    pushed = {p["id"]: p["props"] for q, p in push_log if "SET n +=" in q}
    # The code node's embedding_code is its own code_vecs row...
    assert pushed["code1"]["embedding_code"] == code_code_row.tolist()
    # ...and it carries no text-space prop.
    assert "embedding_text" not in pushed["code1"]
    # A text-family node carries no embedding_text either: the sidecar has no
    # text rows to draw from.
    assert "embedding_text" not in pushed["doc1"]
    assert "embedding_code" not in pushed["doc1"]

    # Exactly one vector index, for the code space only, dimensioned from the
    # code group's meta -- never a text-space index.
    ddl = [q for q, _ in push_log if "CREATE VECTOR INDEX" in q]
    assert len(ddl) == 1
    code_ddl = ddl[0]
    assert "embedding_code" in code_ddl
    assert "embedding_text" not in code_ddl
    assert _integer_tokens(code_ddl) == {3}


def test_hostile_punctuation_parameterized_not_interpolated(monkeypatch):
    """Punctuation-heavy scalar values (embedded quotes, backslashes, newlines,
    braces) must travel inside the Cypher ``$id`` and ``$props`` parameters,
    never be string-interpolated into the query text itself.
    """
    G = nx.Graph()
    # Sentinel punctuation that would break or alter a naive f-string.
    hostile_id = 'k8s://payments/ConfigMap/ha-"x\\y\n{z'
    hostile_label = 'ConfigMap/name-with-"quotes-\\backslash'
    hostile_source = 'deployments/ha-"x\\y\n{z.yaml'
    G.add_node(
        hostile_id,
        label=hostile_label,
        file_type="k8s",
        source_file=hostile_source,
        source_location="line_42\nline_43",
    )

    log = []
    _install_fake_neo4j(monkeypatch, log)
    push_to_neo4j(G, uri="bolt://localhost:7687", user="neo4j", password="pw")

    merges = [
        (query, params)
        for kind, query, params in log
        if kind == "neo4j" and "SET n +=" in query
    ]
    assert len(merges) == 1
    query, params = merges[0]

    # The raw values must be preserved exactly in the parameter dict.
    assert params["id"] == hostile_id
    assert params["props"]["id"] == hostile_id
    assert params["props"]["label"] == hostile_label
    assert params["props"]["source_file"] == hostile_source
    assert params["props"]["source_location"] == "line_42\nline_43"

    # The query text must be free of the hostile literal values in any value
    # position; if they were f-string-interpolated they would appear here.
    assert hostile_id not in query
    assert hostile_label not in query
    assert hostile_source not in query
    assert "line_42\nline_43" not in query

    # The parameterized placeholders must still be present in the query text.
    assert "{id: $id}" in query
    assert "SET n += $props" in query
