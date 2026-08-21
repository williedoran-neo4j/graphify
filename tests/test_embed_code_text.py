"""C1 — a ``file_type == "code"`` node's text embeds the contents of its
``source_file``.

``build_node_text`` keeps the existing 4-line attribute skeleton (label /
source_file[:loc] / rationale-or-empty / capped neighbour line) and appends a
block of the source file's raw text, the whole string still capped at
``_NODE_TEXT_CAP_CHARS``. A resolution root is passed so the test builds its
own fixture file instead of depending on a real repo tree.
"""
from __future__ import annotations

import networkx as nx
import pytest

from graphify.embed import _NODE_TEXT_CAP_CHARS, _neighbour_text, build_node_text

_WHITESPACE = " \t\f\v\r\xa0　\v\t"
_SOURCE_MARKER = "def parse_token(text: str):"

_TINY_SOURCE = (
    'def parse_token(text: str):\n    """Extract a token from the text input."""\n'
)
_HUGE_SOURCE = (
    # Escape a tab so the heredoc below does not expand it; the tab is only
    # whitespace and never part of an assertion token.
    "\\\\t".join(
        [
            'def parse_token(text: str):\n    x = 0\n',
            "print('\\\"escaped quote\\\"')\n",
        ]
    )
    * 100
)


@pytest.fixture
def code_graph_root(tmp_path):
    """A ``code`` node whose source file exists under the resolution root."""
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    source = source_dir / "parser.py"
    source.write_text(_TINY_SOURCE, encoding="utf-8")
    graph = nx.Graph()
    graph.add_node(
        "n-code-02",
        id="n-code-02",
        label="TokenParser",
        source_file="src/parser.py",
        file_type="code",
    )
    graph.add_node(
        "n-doc-01",
        id="n-doc-01",
        label="API Tokens",
        source_file="docs/security.md",
        file_type="document",
    )
    graph.add_edge("n-code-02", "n-doc-01")
    return graph, tmp_path


@pytest.fixture
def huge_code_graph_root(tmp_path):
    """A ``code`` node whose source file's raw text (with a long tail block
    after a candidate marker) far exceeds any full-text cap."""
    source = tmp_path / "parser.py"
    source.write_text(
        f"{_SOURCE_MARKER}\n{'# filler ' * 400}\ngit tail marker az09 xyz\n",
        encoding="utf-8",
    )
    graph = nx.Graph()
    graph.add_node(
        "n-code-huge",
        id="n-code-huge",
        label="TokenParser",
        source_file="parser.py",
        file_type="code",
    )
    return graph, tmp_path


def test_build_node_text_code_reads_source_file(code_graph_root):
    """The code node's text keeps the 4-line attribute skeleton (label /
    source_file / empty rationale / neighbour line) and appends the file's raw
    source content, all within the full-text cap."""
    graph, root = code_graph_root
    text = build_node_text(graph, "n-code-02", graph.nodes["n-code-02"], root)

    assert len(text) <= _NODE_TEXT_CAP_CHARS
    lines = text.split("\n")
    # The existing attribute structure survives verbatim: label line first,
    # path; empty rationale line, degree-ordered neighbour line.
    assert lines[0] == "TokenParser"
    assert lines[1] == "src/parser.py"
    assert lines[2] == ""
    assert lines[3] == _neighbour_text(graph, "n-code-02", limit=10)
    # The file's raw source text is present in the returned text.
    assert _SOURCE_MARKER in text
    assert _TINY_SOURCE.strip() in text


def test_build_node_text_code_truncated_long_file_never_exceeds_cap(huge_code_graph_root):
    """A source file far larger than the cap yields a text of at most
    ``_NODE_TEXT_CAP_CHARS`` characters — the raw source block is truncated —
    while the attribute skeleton (label line first) survives the truncation.
    The candidate marker sits well inside the cap, so the seeded attribute text
    is never empty and the (absent) marker is never cut."""

    graph, root = huge_code_graph_root
    text = build_node_text(graph, "n-code-huge", graph.nodes["n-code-huge"], root)

    assert len(text) <= _NODE_TEXT_CAP_CHARS
    assert text.split("\n")[0] == "TokenParser"
    assert _SOURCE_MARKER in text
    assert "git tail marker az09 xyz" not in text


def test_build_node_text_code_deterministic(code_graph_root):
    """Two calls over the same unchanged graph produce byte-identical text (I2)."""
    graph, root = code_graph_root
    first = build_node_text(graph, "n-code-02", graph.nodes["n-code-02"], root)
    second = build_node_text(graph, "n-code-02", graph.nodes["n-code-02"], root)
    assert first == second
    assert first.encode("utf-8") == second.encode("utf-8")


def test_build_node_text_code_missing_source_file_does_not_raise(tmp_path):
    """A code node whose source file does not exist must not raise: it falls
    through to the attribute-only text (label / path / empty rationale /
    neighbour line), within the full-text cap."""
    graph = nx.Graph()
    graph.add_node(
        "n-code-gone",
        id="n-code-gone",
        label="Gone parser",
        source_file="src/missing.py",
        file_type="code",
    )
    graph.add_node(
        "n-doc-01",
        id="n-doc-01",
        label="API Tokens",
        source_file="docs/security.md",
        file_type="document",
    )
    graph.add_edge("n-code-gone", "n-doc-01")

    text = build_node_text(graph, "n-code-gone", graph.nodes["n-code-gone"], tmp_path)

    assert len(text) <= _NODE_TEXT_CAP_CHARS
    assert text == "\n".join(
        [
            "Gone parser",
            "src/missing.py",
            "",
            _neighbour_text(graph, "n-code-gone", limit=10),
        ]
    )
