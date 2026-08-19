"""C5.1 — build_node_text(graph, nid, node) composes the deterministic 4-line text.

Contract C2 (R5 thread form): for a text-family node (document/paper/rationale/
concept/code) ``build_node_text`` returns exactly

    1. ``{label}``
    2. ``{source_file}:{source_location}`` when ``source_location`` is truthful,
       else ``source_file`` alone
    3. ``{rationale}`` when truthful, else NO line
    4. one line of space-joined neighbour labels (sorted by ``(-degree, id)``,
       head 10), emitted even when empty

joined with ``"\n"``; for ``image`` (and every other non-text-family type) it
returns ``None`` (excluded from every space). Invariant I2: two runs over the
same unchanged graph produce byte-identical strings.

This file's purpose is pinned by STRUCTURE AND CONTENT, not by a literal
full-string golden (the done T1 exact-literal rewrite per C1/C2).
"""
from __future__ import annotations

import networkx as nx
import pytest

from graphify.embed import _NODE_TEXT_CAP_CHARS, _neighbour_text, build_node_text


@pytest.fixture
def text_graph() -> nx.Graph:
    """Four text-family nodes with distinct known degrees.

    Edges are added in an order chosen so that neighbour insertion order does
    NOT match id order, keeping the ``(-degree, id)`` sort the sole reason the
    neighbour lines below hold.

    Degrees: n-doc-01 = 3, n-code-02 = 2, n-concept-03 = 2, n-paper-04 = 1.
    """
    g = nx.Graph()
    g.add_nodes_from(
        [
            (
                "n-doc-01",
                {
                    "id": "n-doc-01",
                    "label": "API Tokens",
                    "source_file": "docs/security.md",
                    "source_location": "L42",
                    "rationale": "credentials policy guidance",
                    "file_type": "document",
                },
            ),
            (
                "n-code-02",
                {
                    "id": "n-code-02",
                    "label": "TokenParser",
                    "source_file": "src/parser.py",
                    "file_type": "code",
                },
            ),
            (
                "n-concept-03",
                {
                    "id": "n-concept-03",
                    "label": "Graph Index",
                    "source_file": "docs/index.md",
                    "file_type": "concept",
                },
            ),
            (
                "n-paper-04",
                {
                    "id": "n-paper-04",
                    "label": "Codegen Runbook",
                    "source_file": "docs/codegen.md",
                    "file_type": "paper",
                },
            ),
        ]
    )
    g.add_edges_from(
        [
            ("n-doc-01", "n-concept-03"),
            ("n-doc-01", "n-code-02"),
            ("n-code-02", "n-concept-03"),
            ("n-doc-01", "n-paper-04"),
        ]
    )
    return g


def test_build_node_text_text_family(text_graph):
    """T1 (rewritten per C1/C2) — the composed text has exactly the 4-line
    structure, in order: label / path:L42 / rationale / neighbour line, with
    the neighbours sorted highest-degree first and ties broken by id."""
    lines = build_node_text(
        text_graph, "n-doc-01", text_graph.nodes["n-doc-01"]
    ).split("\n")

    assert len(lines) == 4
    assert lines[0] == "API Tokens"
    assert lines[1] == "docs/security.md:L42"
    assert lines[2] == "credentials policy guidance"
    # n-code-02 and n-concept-03 are tied at degree 2 (id order wins), and the
    # degree-1 n-paper-04 sorts last.
    assert lines[3] == "TokenParser Graph Index Codegen Runbook"


def test_build_node_text_neighbour_order_highest_degree_first(text_graph):
    """C2/§2 — the neighbour line sorts by (-degree, id): the HIGHER-degree
    neighbour comes first (a degree-blind or id-only sort would flip it)."""
    code_attr = text_graph.nodes["n-code-02"]
    lines = build_node_text(text_graph, "n-code-02", code_attr).split("\n")

    assert len(lines) == 4
    assert lines[0] == "TokenParser"          # code node is in the text family
    assert lines[1] == "src/parser.py"        # truthful source_file, no location
    assert lines[2] == ""                      # no rationale -> no line
    # n-doc-01 (degree 3) sorts before n-concept-03 (degree 2); insertion and
    # bare-id orders both give "Graph Index API Tokens" — the result is "the
    # sole reason" the degree-first sort.
    assert lines[3] == "API Tokens Graph Index"


def test_build_node_text_no_rationale_no_location():
    """T1 arm — a solo text-family node with neither rationale nor
    source_location yields ``label / path / empty / empty`` (the neighbour line
    is still emitted, proving the 4-line structure survives neighbour-free
    calls)."""
    g = nx.Graph()
    g.add_node(
        "n-plain-04",
        id="n-plain-04",
        label="Retry Policy",
        source_file="docs/retries.md",
        file_type="paper",
    )
    text = build_node_text(g, "n-plain-04", g.nodes["n-plain-04"])
    assert text == "Retry Policy\ndocs/retries.md\n\n"


def test_build_node_text_deterministic(text_graph):
    """T2 — two calls over the same unchanged graph are byte-identical (I2)."""
    first = build_node_text(text_graph, "n-doc-01", text_graph.nodes["n-doc-01"])
    second = build_node_text(text_graph, "n-doc-01", text_graph.nodes["n-doc-01"])
    assert first == second
    assert first.encode("utf-8") == second.encode("utf-8")


def test_image_node_excluded(text_graph):
    """T3 (first half) — an image node produces no text: build_node_text is
    None even when the graph has edges among text-family nodes."""
    image_node = {
        "id": "n-IMG-07",
        "label": "architecture diagram",
        "source_file": "assets/arch.png",
        "file_type": "image",
    }
    assert build_node_text(text_graph, "n-IMG-07", image_node) is None


def test_build_node_text_rt14_cap_truncates_neighbour_tail():
    """RT14 (C5.3) — the full constructed text is capped at
    ``_NODE_TEXT_CAP_CHARS``; the cap truncates the NEIGHBOUR TAIL
    (lowest-degree whole labels dropped) and never touches the first three
    lines (label, path, empty rationale).

    Fixture: short-label text-family node with 10 neighbours whose labels are
    each ``NbLabel{k}_`` + ``"x"*200`` (209 chars) and whose degrees are
    distinct (10..1), so the ``(-degree, id)`` sort is DEGREE-driven —
    NbLabel0 is the highest-degree HEAD, NbLabel9 the lowest-degree TAIL. The
    uncapped join (2099 chars) is computed through the C5.1-accepted
    ``_neighbour_text(graph, nid, limit=10)`` seam, never hand-concatenated.
    """
    g = nx.Graph()
    g.add_node(
        "n-rt14",
        id="n-rt14",
        label="L",
        source_file="docs/notes.md",
        file_type="document",
    )
    for k in range(10):
        g.add_node(
            f"nb-{k}",
            id=f"nb-{k}",
            label=f"NbLabel{k}_" + ("x" * 200),
            file_type="document",
        )
        g.add_edge("n-rt14", f"nb-{k}")

    uncapped = build_node_text(g, "n-rt14", g.nodes["n-rt14"])
    uncapped_neighbours = _neighbour_text(g, "n-rt14", limit=10)
    assert len(uncapped_neighbours) > _NODE_TEXT_CAP_CHARS  # uncapped join 2099 chars

    text = build_node_text(g, "n-rt14", g.nodes["n-rt14"])
    lines = text.split("\n")

    assert len(text) <= _NODE_TEXT_CAP_CHARS
    assert text.split("\n")[:3] == uncapped.split("\n")[:3]
    assert lines[3] != uncapped_neighbours and len(lines[3]) < len(uncapped_neighbours)
    assert "NbLabel9" not in lines[3]
    assert "NbLabel0_" in lines[3]
